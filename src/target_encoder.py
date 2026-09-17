"""
SmoothTargetEncoder, ported verbatim from Notebook 5.

Why this file has to exist: the class was originally defined inline in the
notebook, so target_encoder.joblib was pickled with a module reference of
`__main__.SmoothTargetEncoder` (the notebook kernel's __main__). Outside
that notebook process — in this API, in pytest, anywhere — Python's
unpickler can't find it there and raises AttributeError.

The fix is two-part: (1) the class needs to live somewhere importable
(here), and (2) something has to register it under `__main__` before
`joblib.load()` runs, since that's the module path baked into the pickle.
Part 2 lives in src/data_access.py.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SmoothTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols, m: int = 50):
        self.cols = cols
        self.m = m

    def fit(self, X, y):
        X_ = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_ = np.asarray(y, dtype=float)
        self.global_mean_ = float(y_.mean())
        self.maps_ = {}
        for col in self.cols:
            if col not in X_.columns:
                continue
            tmp = pd.DataFrame({"cat": X_[col].astype(str), "y": y_})
            stats = tmp.groupby("cat")["y"].agg(["mean", "count"])
            smoothed = (stats["count"] * stats["mean"] + self.m * self.global_mean_) / (
                stats["count"] + self.m
            )
            self.maps_[col] = smoothed.to_dict()
        return self

    def transform(self, X):
        X_ = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col in self.cols:
            if col not in X_.columns:
                continue
            X_[col] = (
                X_[col]
                .astype(str)
                .map(self.maps_.get(col, {}))
                .fillna(self.global_mean_)
                .astype("float64")
            )
        return X_
