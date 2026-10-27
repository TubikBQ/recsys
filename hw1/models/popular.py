import numpy as np


class Popular:
    """Бейзлайн: популярность объекта в обучающей части."""

    def fit(self, ts):
        self.pop = np.asarray(ts.X.sum(axis=0)).ravel().astype(np.float32)
        return self

    def score(self, users):
        return np.tile(self.pop, (len(users), 1))
