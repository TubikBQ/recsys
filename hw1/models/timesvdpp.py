import numpy as np
import scipy.sparse as sp
import torch


class TimeSVDpp:
    """timeSVD++ (Koren, 2009):
    r(u,i,t) = mu + b_u + a_u dev_u(t) + b_{u,t} + (b_i + b_{i,Bin(t)}) (c_u + c_{u,t})
               + q_i^T (p_u + a^p_u dev_u(t) + |N(u)|^{-1/2} sum_{j in N(u)} y_j),
    dev_u(t) = sign(t - t_u) |t - t_u|^beta, нормированный на своё стандартное отклонение в train.
    Обучение как у Koren: SGD по оценкам, квадрат ошибки плюс L2 на параметры каждой оценки.
    Три коэффициента L2: reg для факторов, reg_bias для смещений, reg_day для b_{u,t}, c_{u,t}.
    Если передан val, после каждой эпохи считается RMSE на val, обучение останавливается
    после patience эпох без улучшения, сохраняется лучшая эпоха."""

    def __init__(self, factors=32, reg=0.05, reg_bias=0.02, reg_day=0.1, lr=0.005,
                 epochs=30, batch=512, patience=3, n_bins=30, beta=0.4, seed=42):
        self.factors, self.reg, self.reg_bias, self.reg_day = factors, reg, reg_bias, reg_day
        self.lr, self.epochs, self.batch, self.patience = lr, epochs, batch, patience
        self.n_bins, self.beta, self.seed = n_bins, beta, seed

    def _dev(self, u, t):
        d = t - self.t_mean[u]
        return torch.sign(d) * torch.abs(d) ** self.beta / self.dev_scale

    def _bin(self, t):
        b = ((t - self.t_min) / self.bin_width).long()
        return torch.clamp(b, 0, self.n_bins - 1)

    def _predict(self, ysum, u, i, t):
        """Прогноз без дневных компонент: в будущие дни они неизвестны и равны нулю."""
        dev = self._dev(u, t)
        pu = self.p[u] + self.ap[u] * dev[:, None] + ysum[u]
        return (self.mu + self.bu[u] + self.au[u] * dev
                + (self.bi[i] + self.bibin[i, self._bin(t)]) * self.cu[u] + (self.q[i] * pu).sum(1))

    def fit(self, ts, val=None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        nu, ni, k = ts.n_users, ts.n_items, self.factors
        u = torch.from_numpy(ts.u).long()
        i = torch.from_numpy(ts.i).long()
        r = torch.from_numpy(ts.r).float()
        t = torch.from_numpy(ts.t).float()
        self.mu = float(r.mean())
        self.t_min = float(t.min())
        self.bin_width = max((float(t.max()) - self.t_min + 1) / self.n_bins, 1.0)
        cnt = torch.bincount(u, minlength=nu).float()
        self.t_mean = torch.zeros(nu).index_add_(0, u, t) / cnt
        self.dev_scale = 1.0
        self.dev_scale = max(float(self._dev(u, t).std()), 1e-6)
        ud = np.unique(np.stack([ts.u, ts.t.astype(np.int64)], 1), axis=0, return_inverse=True)[1].ravel()
        d_idx = torch.from_numpy(ud).long()
        n_ud = int(d_idx.max()) + 1
        norm = 1.0 / np.sqrt(np.bincount(ts.u, minlength=nu))
        N = sp.csr_matrix((norm[ts.u], (ts.u, ts.i)), shape=(nu, ni)).tocoo()
        idx = torch.from_numpy(np.vstack([N.row, N.col]).astype(np.int64))
        self.N = torch.sparse_coo_tensor(idx, torch.from_numpy(N.data.astype(np.float32)), (nu, ni),
                                         check_invariants=False).coalesce()
        Ncsr = N.tocsr()

        def P(*s, std=0.0, val=0.0):
            x = torch.full(s, val) if std == 0 else torch.randn(*s) * std
            return torch.nn.Parameter(x)

        self.bu, self.au, self.bi = P(nu), P(nu), P(ni)
        self.bibin, self.cu = P(ni, self.n_bins), P(nu, val=1.0)
        self.but, self.cut = P(n_ud), P(n_ud)
        self.p, self.ap = P(nu, k, std=0.1), P(nu, k)
        self.q, self.y = P(ni, k, std=0.1), P(ni, k, std=0.1)
        params = {"bu": self.bu, "au": self.au, "bi": self.bi, "bibin": self.bibin, "cu": self.cu,
                  "but": self.but, "cut": self.cut, "p": self.p, "ap": self.ap, "q": self.q, "y": self.y}
        opt = torch.optim.SGD(list(params.values()), lr=self.lr)
        dev, bins, n = self._dev(u, t), self._bin(t), len(r)
        if val is not None:
            vu, vi, vt, vr = (torch.as_tensor(x) for x in val)
            vu, vi, vt, vr = vu.long(), vi.long(), vt.float(), vr.float()
        self.best_rmse, self.best_epoch, best_state, bad = np.inf, self.epochs, None, 0
        self.history = []
        for ep in range(1, self.epochs + 1):
            perm = torch.from_numpy(rng.permutation(n))
            for s in range(0, n, self.batch):
                b = perm[s:s + self.batch]
                ub, ib, db, dvb, bb = u[b], i[b], d_idx[b], dev[b], bins[b]
                Nb = Ncsr[ts.u[b.numpy()]].tocoo()
                Nb = torch.sparse_coo_tensor(np.vstack([Nb.row, Nb.col]).astype(np.int64),
                                             Nb.data.astype(np.float32), Nb.shape, check_invariants=False)
                ysum = torch.sparse.mm(Nb, self.y)
                pu = self.p[ub] + self.ap[ub] * dvb[:, None] + ysum
                bit = self.bi[ib] + self.bibin[ib, bb]
                cut = self.cu[ub] + self.cut[db]
                pred = (self.mu + self.bu[ub] + self.au[ub] * dvb + self.but[db]
                        + bit * cut + (self.q[ib] * pu).sum(1))
                loss = ((r[b] - pred) ** 2).sum()
                loss = loss + self.reg * (self.p[ub].pow(2).sum() + self.q[ib].pow(2).sum()
                                          + self.y[ib].pow(2).sum() + self.ap[ub].pow(2).sum())
                loss = loss + self.reg_bias * (self.bu[ub].pow(2).sum() + self.bi[ib].pow(2).sum()
                                               + self.bibin[ib, bb].pow(2).sum() + (self.cu[ub] - 1).pow(2).sum()
                                               + self.au[ub].pow(2).sum())
                loss = loss + self.reg_day * (self.but[db].pow(2).sum() + self.cut[db].pow(2).sum())
                opt.zero_grad(); loss.backward(); opt.step()
            if val is not None:
                with torch.no_grad():
                    vp = self._predict(torch.sparse.mm(self.N, self.y), vu, vi, vt).clamp(1, 5)
                    rmse = float(torch.sqrt(((vp - vr) ** 2).mean()))
                self.history.append((ep, rmse))
                if rmse < self.best_rmse:
                    self.best_rmse, self.best_epoch, bad = rmse, ep, 0
                    best_state = {kk: v.detach().clone() for kk, v in params.items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            with torch.no_grad():
                for kk, v in params.items():
                    v.copy_(best_state[kk])
        with torch.no_grad():
            self.ysum = torch.sparse.mm(self.N, self.y)
        self.t_rec = ts.t_rec
        return self

    @torch.no_grad()
    def predict(self, users, items, t):
        u = torch.as_tensor(users).long(); i = torch.as_tensor(items).long()
        return self._predict(self.ysum, u, i, torch.as_tensor(t).float()).numpy()

    @torch.no_grad()
    def score(self, users):
        u = torch.as_tensor(users).long()
        t = torch.full((len(u),), float(self.t_rec))
        dev = self._dev(u, t)
        pu = self.p[u] + self.ap[u] * dev[:, None] + self.ysum[u]
        bi = self.bi + self.bibin[:, int(self._bin(t[:1])[0])]
        s = (self.mu + self.bu[u] + self.au[u] * dev)[:, None] + bi[None, :] * self.cu[u][:, None] + pu @ self.q.T
        return s.numpy()
