import numpy as np
import scipy.sparse as sp


def _solve_side(R_pref, R_conf, Y, reg, chunk=400):
    """Для каждой строки решает (Y^T C Y + reg I) x = Y^T C p,
    где C = 1 + (c - 1) на наблюдаемых, p = 0 на ненаблюдаемых."""
    n, k = R_pref.shape[0], Y.shape[1]
    YtY = Y.T @ Y + reg * np.eye(k)
    out = np.zeros((n, k))
    indptr, indices = R_conf.indptr, R_conf.indices
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        a, b = indptr[s], indptr[e]
        idx = indices[a:b]
        cm1 = R_conf.data[a:b] - 1.0
        cp = R_conf.data[a:b] * R_pref.data[a:b]
        Yi = Y[idx]
        lens = np.diff(indptr[s:e + 1])
        starts = np.minimum(indptr[s:e] - a, max(b - a - 1, 0))
        outer = cm1[:, None, None] * Yi[:, :, None] * Yi[:, None, :]
        A = YtY[None] + np.add.reduceat(outer, starts, axis=0) * (lens > 0)[:, None, None]
        rhs = np.add.reduceat(cp[:, None] * Yi, starts, axis=0) * (lens > 0)[:, None]
        out[s:e] = np.linalg.solve(A, rhs[:, :, None])[:, :, 0]
    return out


class IALS:
    """iALS (Hu, Koren, Volinsky):
    min sum_{u,i} c_ui (p_ui - x_u^T y_i)^2 + reg (|X|^2 + |Y|^2).
    Базовый вариант: p_ui = 1, c_ui = 1 + alpha для наблюдаемых пар.
    Можно передать свой целевой сигнал target(ts) -> p_ui для наблюдаемых пар."""

    def __init__(self, factors=64, reg=1.0, alpha=10.0, iters=10, target=None, seed=42):
        self.factors, self.reg, self.alpha, self.iters = factors, reg, alpha, iters
        self.target, self.seed = target, seed

    def fit(self, ts):
        p = np.ones(len(ts.u)) if self.target is None else self.target(ts)
        c = 1.0 + self.alpha * np.ones(len(ts.u))
        shape = (ts.n_users, ts.n_items)
        Pu = sp.csr_matrix((p, (ts.u, ts.i)), shape=shape)
        Cu = sp.csr_matrix((c, (ts.u, ts.i)), shape=shape)
        Pu.sort_indices(); Cu.sort_indices()
        Pi, Ci = Pu.T.tocsr(), Cu.T.tocsr()
        Pi.sort_indices(); Ci.sort_indices()
        rng = np.random.default_rng(self.seed)
        self.U = rng.normal(0, 0.01, (ts.n_users, self.factors))
        self.V = rng.normal(0, 0.01, (ts.n_items, self.factors))
        for _ in range(self.iters):
            self.U = _solve_side(Pu, Cu, self.V, self.reg)
            self.V = _solve_side(Pi, Ci, self.U, self.reg)
        return self

    def score(self, users):
        return self.U[users] @ self.V.T
