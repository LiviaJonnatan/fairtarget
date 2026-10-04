"""Heterogeneous treatment-effect estimators.

Two learners are provided, both built from scikit-learn:

* ``TLearner``   - fit separate outcome models for treated and control,
                   CATE = mu1(x) - mu0(x). Simple, but can be biased
                   toward prognostic (baseline-risk) variation.
* ``DRLearner``  - doubly-robust pseudo-outcomes with cross-fitting, then a
                   final regression of the pseudo-outcome on x. Robust to
                   misspecification of either the outcome or propensity model.

Both expose ``fit(X, W, Y)`` and ``predict(X)``. Cross-fitting is used so
that the predictions for the policy stage are not overfit to the same
observations, which matters because we rank on them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import KFold


def _gbr(seed: int) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                     subsample=0.8, random_state=seed)


def _gbc(seed: int) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05,
                                      subsample=0.8, random_state=seed)


class TLearner:
    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X: pd.DataFrame, W: np.ndarray, Y: np.ndarray) -> "TLearner":
        self.m1 = _gbc(self.seed).fit(X[W == 1], Y[W == 1])
        self.m0 = _gbc(self.seed + 1).fit(X[W == 0], Y[W == 0])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.m1.predict_proba(X)[:, 1] - self.m0.predict_proba(X)[:, 1]


class DRLearner:
    """Doubly-robust learner (Kennedy, 2020) with K-fold cross-fitting.

    Because the data come from a randomised trial, the propensity e(x) is
    known; we still estimate it to keep the code honest for observational
    use, but clip it away from 0/1.
    """

    def __init__(self, n_folds: int = 5, seed: int = 0):
        self.n_folds = n_folds
        self.seed = seed

    def fit(self, X: pd.DataFrame, W: np.ndarray, Y: np.ndarray) -> "DRLearner":
        X = X.reset_index(drop=True)
        W = np.asarray(W)
        Y = np.asarray(Y)
        pseudo = np.zeros(len(Y))
        kf = KFold(self.n_folds, shuffle=True, random_state=self.seed)
        for k, (tr, te) in enumerate(kf.split(X)):
            m1 = _gbc(self.seed + k).fit(X.iloc[tr][W[tr] == 1], Y[tr][W[tr] == 1])
            m0 = _gbc(self.seed + 10 + k).fit(X.iloc[tr][W[tr] == 0], Y[tr][W[tr] == 0])
            e = _gbc(self.seed + 20 + k).fit(X.iloc[tr], W[tr])
            mu1 = m1.predict_proba(X.iloc[te])[:, 1]
            mu0 = m0.predict_proba(X.iloc[te])[:, 1]
            eh = np.clip(e.predict_proba(X.iloc[te])[:, 1], 0.05, 0.95)
            w, y = W[te], Y[te]
            pseudo[te] = (mu1 - mu0
                          + w * (y - mu1) / eh
                          - (1 - w) * (y - mu0) / (1 - eh))
        self.final = _gbr(self.seed + 99).fit(X, pseudo)
        self.pseudo_ = pseudo
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.final.predict(X)


def crossfit_cate(learner_cls, X: pd.DataFrame, W: np.ndarray, Y: np.ndarray,
                  n_folds: int = 5, seed: int = 0, **kw) -> np.ndarray:
    """Out-of-fold CATE predictions for every row, so that ranking for
    allocation is never done on in-sample fits."""
    X = X.reset_index(drop=True)
    W, Y = np.asarray(W), np.asarray(Y)
    out = np.zeros(len(Y))
    kf = KFold(n_folds, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(kf.split(X)):
        m = learner_cls(seed=seed + k, **kw).fit(X.iloc[tr], W[tr], Y[tr])
        out[te] = m.predict(X.iloc[te])
    return out
