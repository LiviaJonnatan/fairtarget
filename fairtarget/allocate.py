"""Budget-constrained allocation rules.

Given a score s_i for each individual (estimated or true CATE), a budget k
(number of people who can be treated) and a group label g_i, choose a binary
allocation a in {0,1}^n to maximise sum_i s_i a_i subject to

    sum_i a_i <= k                                    (budget)
    sum_{i in g} a_i >= floor_g * k   for each g      (fairness floors)

The fairness floor says each group must receive at least a given share of
the budget. Because the group constraints partition the units and there is a
single budget row, the constraint matrix is totally unimodular, so the LP
relaxation is integral and a plain LP solver returns a valid 0/1 solution.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def uniform(n: int, k: int, seed: int = 0) -> np.ndarray:
    """Treat k people chosen at random (the RCT's own allocation rule)."""
    rng = np.random.default_rng(seed)
    a = np.zeros(n, dtype=int)
    a[rng.choice(n, size=k, replace=False)] = 1
    return a


def greedy(scores: np.ndarray, k: int) -> np.ndarray:
    """Treat the k people with the highest scores."""
    a = np.zeros(len(scores), dtype=int)
    a[np.argsort(-scores)[:k]] = 1
    return a


def fair_floor(scores: np.ndarray, group: np.ndarray, k: int,
               floors: dict[int, float]) -> np.ndarray:
    """Maximise total score subject to a budget and per-group minimum shares.

    Parameters
    ----------
    scores : predicted (or true) benefit per person
    group  : integer group label per person
    k      : budget (number of treatments)
    floors : {group_label: minimum share of k that must go to that group}

    Returns a 0/1 allocation. Infeasible floors (summing to > 1, or exceeding
    a group's size) raise ValueError.
    """
    n = len(scores)
    if sum(floors.values()) > 1 + 1e-9:
        raise ValueError("Floors sum to more than 100% of the budget")
    A_ub, b_ub = [np.ones(n)], [k]
    for g, f in floors.items():
        mask = (group == g).astype(float)
        need = f * k
        if need > mask.sum():
            raise ValueError(f"Group {g} has {int(mask.sum())} members but floor requires {need:.0f}")
        A_ub.append(-mask)  # -sum_{i in g} a_i <= -need
        b_ub.append(-need)
    res = linprog(c=-scores, A_ub=np.vstack(A_ub), b_ub=np.array(b_ub),
                  bounds=[(0, 1)] * n, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    a = np.rint(res.x).astype(int)
    # Guard against rare LP ties producing fractional solutions.
    if a.sum() > k:
        extra = np.where(a == 1)[0]
        drop = extra[np.argsort(scores[extra])[: a.sum() - k]]
        a[drop] = 0
    return a


def group_floor_from_population(group: np.ndarray) -> dict[int, float]:
    """Proportional floors: each group gets at least its population share."""
    vals, counts = np.unique(group, return_counts=True)
    return {int(v): c / len(group) for v, c in zip(vals, counts)}
