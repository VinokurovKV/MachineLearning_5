import numpy as np
import typing


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

        self.std_[self.std_ == 0] = 1.0  # чтобы 0 не было

        return self

    def transform(self, X):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet. Call fit first.")

        X = np.array(X)
        return (X - self.mean_) / self.std_


class MinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.max_ = None
        self.range_ = None

    def fit(self, X):
        X = np.array(X)
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.range_ = self.max_ - self.min_

        self.range_[self.range_ == 0] = 1.0  # чтобы 0 не было

        return self

    def transform(self, X):
        if self.min_ is None or self.max_ is None:
            raise ValueError("Scaler has not been fitted yet. Call fit first.")

        X = np.array(X)
        return (X - self.min_) / self.range_
