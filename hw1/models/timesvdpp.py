import numpy as np
import scipy.sparse as sp
import torch


class TimeSVDpp:
    """timeSVD++ (Koren, 2009):
    r(u,i,t) = mu + b_u + a_u dev_u(t) + b_{u,t} + (b_i + b_{i,Bin(t)}) (c_u + c_{u,t})
               + q_i^T (p_u + a^p_u dev_u(t) + |N(u)|^{-1/2} sum_{j in N(u)} y_j),
    dev_u(t) = sign(t - t_u) |t - t_u|^beta.
    Обучение: MSE + L2, полный батч, Adam."""

    def __init__(self, factors=32, reg=5.0, reg_bias=1.0, lr=0.01, epochs=200,
                 n_bins=30, beta=0.4, seed=42):
        self.factors, self.reg, self.reg_bias, self.lr = factors, reg, reg_bias, lr
        self.epochs, self.n_bins, self.beta, self.seed = epochs, n_bins, beta, seed

    def _dev(self, u, t):
        d = t - self.t_mean[u]
        return torch.sign(d) * torch.abs(d) ** self.beta

    def _bin(self, t):
        b = ((t - self.t_min) / self.bin_width).long()
        return torch.clamp(b, 0, self.n_bins - 1)

    def fit(self, ts):
        torch.manual_seed(self.seed)
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
        ud = np.unique(np.stack([ts.u, ts.t.astype(np.int64)], 1), axis=0, return_inverse=True)[1].ravel()
        d_idx = torch.from_numpy(ud).long()
        n_ud = int(d_idx.max()) + 1
        norm = 1.0 / np.sqrt(np.bincount(ts.u, minlength=nu))
        N = sp.csr_matrix((norm[ts.u], (ts.u, ts.i)), shape=(nu, ni)).tocoo()
        self.N = torch.sparse_coo_tensor(np.vstack([N.row, N.col]), N.data.astype(np.float32), (nu, ni)).coalesce()

        def P(*s, std=0.0, val=0.0):
            x = torch.full(s, val) if std == 0 else torch.randn(*s) * std
            return torch.nn.Parameter(x)

        self.bu, self.au, self.bi = P(nu), P(nu), P(ni)
        self.bibin, self.cu = P(ni, self.n_bins), P(nu, val=1.0)
        self.but, self.cut = P(n_ud), P(n_ud)
        self.p, self.ap = P(nu, k, std=0.01), P(nu, k)
        self.q, self.y = P(ni, k, std=0.01), P(ni, k, std=0.01)
        params = [self.bu, self.au, self.bi, self.bibin, self.cu, self.but, self.cut,
                  self.p, self.ap, self.q, self.y]
        opt = torch.optim.Adam(params, lr=self.lr)
        dev, bins, n = self._dev(u, t), self._bin(t), len(r)
        for _ in range(self.epochs):
            ysum = torch.sparse.mm(self.N, self.y)
            pu = self.p[u] + self.ap[u] * dev[:, None] + ysum[u]
            pred = (self.mu + self.bu[u] + self.au[u] * dev + self.but[d_idx]
                    + (self.bi[i] + self.bibin[i, bins]) * (self.cu[u] + self.cut[d_idx])
                    + (self.q[i] * pu).sum(1))
            loss = ((r - pred) ** 2).sum()
            loss = loss + self.reg * (self.p.pow(2).sum() + self.q.pow(2).sum()
                                      + self.y.pow(2).sum() + self.ap.pow(2).sum())
            loss = loss + self.reg_bias * (self.bu.pow(2).sum() + self.bi.pow(2).sum()
                                           + self.bibin.pow(2).sum() + self.but.pow(2).sum()
                                           + (self.cu - 1).pow(2).sum() + self.cut.pow(2).sum()
                                           + self.au.pow(2).sum())
            opt.zero_grad(); (loss / n).backward(); opt.step()
        with torch.no_grad():
            self.ysum = torch.sparse.mm(self.N, self.y)
        self.t_rec = ts.t_rec
        return self

    @torch.no_grad()
    def predict(self, users, items, t):
        """Прогноз оценки; дневные компоненты b_{u,t}, c_{u,t} в будущем равны нулю."""
        u = torch.as_tensor(users).long(); i = torch.as_tensor(items).long()
        t = torch.as_tensor(t).float()
        dev = self._dev(u, t)
        pu = self.p[u] + self.ap[u] * dev[:, None] + self.ysum[u]
        pred = (self.mu + self.bu[u] + self.au[u] * dev
                + (self.bi[i] + self.bibin[i, self._bin(t)]) * self.cu[u] + (self.q[i] * pu).sum(1))
        return pred.numpy()

    @torch.no_grad()
    def score(self, users):
        u = torch.as_tensor(users).long()
        t = torch.full((len(u),), self.t_rec)
        dev = self._dev(u, t)
        pu = self.p[u] + self.ap[u] * dev[:, None] + self.ysum[u]
        bi = self.bi + self.bibin[:, int(self._bin(t[:1])[0])]
        s = (self.mu + self.bu[u] + self.au[u] * dev)[:, None] + bi[None, :] * self.cu[u][:, None] + pu @ self.q.T
        return s.numpy()
