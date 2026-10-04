"""Evaluation of allocation rules against the known ground truth.

All welfare numbers are *realised expected gains*: sum of the true CATE over
the treated set, i.e. the expected number of additional good outcomes the
programme buys with its budget. Reported both in absolute terms and relative
to the oracle (allocation on true tau) so the loss can be decomposed into

    oracle - greedy(estimated)      -> cost of estimation error
    greedy(estimated) - fair(est.)  -> price of fairness
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def welfare(a: np.ndarray, tau: np.ndarray) -> float:
    return float((a * tau).sum())


def group_coverage(a: np.ndarray, group: np.ndarray) -> dict[int, float]:
    """Share of each group that receives treatment."""
    return {int(g): float(a[group == g].mean()) for g in np.unique(group)}


def budget_share(a: np.ndarray, group: np.ndarray) -> dict[int, float]:
    """Share of the budget that goes to each group."""
    k = a.sum()
    return {int(g): float(a[group == g].sum() / k) for g in np.unique(group)}


def benefit_share(a: np.ndarray, tau: np.ndarray, group: np.ndarray) -> dict[int, float]:
    """Share of total realised benefit accruing to each group."""
    tot = welfare(a, tau)
    return {int(g): float((a * tau)[group == g].sum() / tot) for g in np.unique(group)}


def summarise(allocations: dict[str, np.ndarray], tau: np.ndarray,
              group: np.ndarray, group_names: dict[int, str] | None = None) -> pd.DataFrame:
    group_names = group_names or {int(g): str(g) for g in np.unique(group)}
    oracle_w = max(welfare(a, tau) for a in allocations.values())
    rows = []
    for name, a in allocations.items():
        w = welfare(a, tau)
        row = {"rule": name, "treated": int(a.sum()), "expected_gain": w,
               "share_of_oracle": w / oracle_w}
        for g, cov in group_coverage(a, group).items():
            row[f"coverage_{group_names[g]}"] = cov
        for g, s in benefit_share(a, tau, group).items():
            row[f"benefit_share_{group_names[g]}"] = s
        rows.append(row)
    return pd.DataFrame(rows).set_index("rule")


def bootstrap_gain(a: np.ndarray, tau: np.ndarray, n_boot: int = 500,
                   seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Nonparametric bootstrap over individuals for the expected gain of a
    fixed allocation. Returns (point, lower, upper)."""
    rng = np.random.default_rng(seed)
    n = len(tau)
    gains = a * tau
    point = gains.sum()
    boots = np.array([gains[rng.integers(0, n, n)].sum() for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)
