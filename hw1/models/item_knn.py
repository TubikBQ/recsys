import numpy as np
import scipy.sparse as sp


class ItemKNN:
    """Item-based CF: косинусное сходство со сжатием, k ближайших соседей.
    score(u, i) = sum_{j in I_u} sim_k(j, i)."""

    def __init__(self, k=50, shrink=10.0):
        self.k, self.shrink = k, shrink

    def fit(self, ts):
        X = ts.X.tocsc().astype(np.float32)
        co = (X.T @ X).tocsr()
        co.setdiag(0)
        co.eliminate_zeros()
        norms = np.sqrt(np.asarray(X.sum(axis=0)).ravel())
        co = co.tocoo()
        sim = co.data / (norms[co.row] * norms[co.col] + self.shrink + 1e-9)
        S = sp.csr_matrix((sim, (co.row, co.col)), shape=co.shape)
        rows, cols, vals = [], [], []
        for j in range(S.shape[0]):
            s, e = S.indptr[j], S.indptr[j + 1]
            if e == s:
                continue
            idx, v = S.indices[s:e], S.data[s:e]
            if len(v) > self.k:
                top = np.argpartition(-v, self.k)[:self.k]
                idx, v = idx[top], v[top]
            rows.append(np.full(len(idx), j))
            cols.append(idx)
            vals.append(v)
        self.S = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=S.shape,
        )
        self.X = ts.X
        return self

    def score(self, users):
        return (self.X[users] @ self.S).toarray()
