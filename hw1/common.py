import gzip
import html
import json
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp


def read_jsonl_gz(path, fields=None):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if fields is not None:
                d = {k: d.get(k) for k in fields}
            rows.append(d)
    return pd.DataFrame(rows)


def parse_price(s):
    if not isinstance(s, str):
        return np.nan
    m = re.match(r"^\s*\$\s*([\d,]+(?:\.\d+)?)", s)
    if m is None:
        return np.nan
    return float(m.group(1).replace(",", ""))


def clean_categories(c):
    if not isinstance(c, list):
        return []
    out = []
    for x in c:
        if isinstance(x, list):
            x = x[-1] if x else ""
        x = html.unescape(str(x)).strip()
        if x and len(x) < 80 and "<" not in x:
            out.append(x)
    return out[1:] if out and out[0] == "Musical Instruments" else out


class TrainSet:
    """Индексация пользователей и объектов по обучающей части."""

    def __init__(self, df, t_rec):
        self.users = np.sort(df["user_id"].unique())
        self.items = np.sort(df["item_id"].unique())
        self.user2idx = pd.Series(np.arange(len(self.users)), index=self.users)
        self.item2idx = pd.Series(np.arange(len(self.items)), index=self.items)
        self.n_users, self.n_items = len(self.users), len(self.items)
        self.u = self.user2idx.loc[df["user_id"]].to_numpy()
        self.i = self.item2idx.loc[df["item_id"]].to_numpy()
        self.r = df["rating"].to_numpy(dtype=np.float32)
        self.t = df["day"].to_numpy(dtype=np.float32)
        self.t_rec = float(t_rec)
        self.X = sp.csr_matrix(
            (np.ones(len(self.u), dtype=np.float32), (self.u, self.i)),
            shape=(self.n_users, self.n_items),
        )
        self.X.data[:] = 1.0
        self.user_features = None
        self.item_features = None


def build_ground_truth(ts, future, min_rating=4.0):
    """Релевантные объекты: покупка в будущем периоде с оценкой >= min_rating.
    Только известные пользователи и объекты."""
    f = future[future["rating"] >= min_rating]
    f = f[f["user_id"].isin(ts.user2idx.index) & f["item_id"].isin(ts.item2idx.index)]
    uu = ts.user2idx.loc[f["user_id"]].to_numpy()
    ii = ts.item2idx.loc[f["item_id"]].to_numpy()
    gt = {}
    for u, i in zip(uu, ii):
        gt.setdefault(u, set()).add(i)
    return gt


def evaluate(model, ts, gt, k=10, batch=1024):
    users = np.array(sorted(gt.keys()))
    ndcg, recall, hr = [], [], []
    recommended = set()
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    for s in range(0, len(users), batch):
        ub = users[s:s + batch]
        scores = np.asarray(model.score(ub), dtype=np.float32)
        known = ts.X[ub]
        scores[known.nonzero()] = -np.inf
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
        top = np.take_along_axis(top, order, axis=1)
        for u, rec in zip(ub, top):
            rel = gt[u]
            hits = np.array([x in rel for x in rec], dtype=float)
            idcg = disc[:min(len(rel), k)].sum()
            ndcg.append((hits * disc).sum() / idcg)
            recall.append(hits.sum() / len(rel))
            hr.append(float(hits.any()))
            recommended.update(rec.tolist())
    return {
        "NDCG@10": float(np.mean(ndcg)),
        "Recall@10": float(np.mean(recall)),
        "HR@10": float(np.mean(hr)),
        "Coverage@10": len(recommended) / ts.n_items,
        "n_users": len(users),
    }


def rating_arrays(ts, part):
    """Оценки из part для известных пользователей и объектов: индексы, день, оценка."""
    k = part[part["user_id"].isin(ts.user2idx.index) & part["item_id"].isin(ts.item2idx.index)]
    return (ts.user2idx.loc[k["user_id"]].to_numpy(), ts.item2idx.loc[k["item_id"]].to_numpy(),
            k["day"].to_numpy(dtype=np.float32), k["rating"].to_numpy(dtype=np.float32))
