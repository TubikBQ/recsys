import numpy as np
import scipy.sparse as sp
import torch


def _pad(rows_ids, rows_w):
    L = max(len(x) for x in rows_ids)
    ids = np.zeros((len(rows_ids), L), dtype=np.int64)
    w = np.zeros((len(rows_ids), L), dtype=np.float32)
    for r, (a, b) in enumerate(zip(rows_ids, rows_w)):
        ids[r, :len(a)], w[r, :len(b)] = a, b
    return torch.from_numpy(ids), torch.from_numpy(w)


class FM:
    """Факторизационная машина второго порядка, обучение BPR.
    y(x) = w0 + sum_f w_f x_f + sum_{f<g} <v_f, v_g> x_f x_g.
    Признаки: one-hot user/item ID и метапризнаки из ts.user_features, ts.item_features."""

    def __init__(self, dim=64, lr=2e-3, reg=1e-4, epochs=15, batch=2048,
                 use_meta=True, seed=42):
        self.dim, self.lr, self.reg, self.epochs = dim, lr, reg, epochs
        self.batch, self.use_meta, self.seed = batch, use_meta, seed

    def _side(self, n, F, offset_id, offset_meta):
        ids, ws = [], []
        F = F.tocsr() if (F is not None and self.use_meta) else None
        for r in range(n):
            a, b = [offset_id + r], [1.0]
            if F is not None:
                s, e = F.indptr[r], F.indptr[r + 1]
                a += (F.indices[s:e] + offset_meta).tolist()
                b += F.data[s:e].tolist()
            ids.append(a); ws.append(b)
        return _pad(ids, ws)

    def _repr(self, ids, w):
        v = self.V(ids) * w[..., None]
        lin = (self.W(ids).squeeze(-1) * w).sum(1)
        e = v.sum(1)
        sq = (v ** 2).sum((1, 2))
        return e, lin, sq

    def fit(self, ts):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        nuf = ts.user_features.shape[1] if ts.user_features is not None else 0
        nif = ts.item_features.shape[1] if ts.item_features is not None else 0
        o_ui, o_um = 0, ts.n_users
        o_ii = o_um + nuf
        o_im = o_ii + ts.n_items
        n_feat = o_im + nif
        self.uid, self.uw = self._side(ts.n_users, ts.user_features, o_ui, o_um)
        self.iid, self.iw = self._side(ts.n_items, ts.item_features, o_ii, o_im)
        self.V = torch.nn.Embedding(n_feat, self.dim)
        torch.nn.init.normal_(self.V.weight, 0, 0.01)
        self.W = torch.nn.Embedding(n_feat, 1)
        torch.nn.init.zeros_(self.W.weight)
        opt = torch.optim.Adam(list(self.V.parameters()) + list(self.W.parameters()), lr=self.lr)
        u_all, i_all = torch.from_numpy(ts.u), torch.from_numpy(ts.i)
        n = len(u_all)
        for _ in range(self.epochs):
            perm = torch.from_numpy(rng.permutation(n))
            for s in range(0, n, self.batch):
                b = perm[s:s + self.batch]
                u, i = u_all[b], i_all[b]
                j = torch.from_numpy(rng.integers(0, ts.n_items, len(b)))
                eu, _, squ = self._repr(self.uid[u], self.uw[u])
                ei, li, sqi = self._repr(self.iid[i], self.iw[i])
                ej, lj, sqj = self._repr(self.iid[j], self.iw[j])
                # члены, зависящие только от пользователя, сокращаются в разности
                si = li + (eu * ei).sum(1) + 0.5 * ((ei ** 2).sum(1) - sqi)
                sj = lj + (eu * ej).sum(1) + 0.5 * ((ej ** 2).sum(1) - sqj)
                loss = -torch.nn.functional.logsigmoid(si - sj).mean()
                loss = loss + self.reg * (squ + sqi + sqj).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            ei, li, sqi = self._repr(self.iid, self.iw)
            self.item_vec = ei.numpy()
            self.item_bias = (li + 0.5 * ((ei ** 2).sum(1) - sqi)).numpy()
            eu, _, _ = self._repr(self.uid, self.uw)
            self.user_vec = eu.numpy()
        return self

    def score(self, users):
        return self.user_vec[users] @ self.item_vec.T + self.item_bias[None, :]
