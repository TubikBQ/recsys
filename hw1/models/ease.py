import numpy as np


class EASE:
    """EASE: B = I - P diagMat(1 / diag(P)), P = (X^T X + lambda I)^{-1}, diag(B) = 0."""

    def __init__(self, reg=500.0):
        self.reg = reg

    def fit(self, ts):
        X = ts.X
        G = (X.T @ X).toarray().astype(np.float64)
        G[np.diag_indices_from(G)] += self.reg
        P = np.linalg.inv(G)
        B = -P / np.diag(P)[None, :]
        B[np.diag_indices_from(B)] = 0.0
        self.B = B.astype(np.float32)
        self.X = X
        return self

    def score(self, users):
        return np.asarray(self.X[users] @ self.B)
