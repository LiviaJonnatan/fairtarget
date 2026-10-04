"""Synthetic population with known heterogeneous treatment effects.

The design mimics a demand-side health intervention (e.g. an SMS reminder
plus small incentive for child immunisation). Each individual has:

* covariates: rural/urban, household income (log), distance to clinic (km),
  maternal education (years), prior-visit indicator;
* a baseline probability of the good outcome Y0 (e.g. completing a vaccine
  schedule) that is lower for remote, poorer households;
* a true conditional average treatment effect (CATE) tau(x) that is
  heterogeneous: the intervention helps most where access is the binding
  constraint but not so remote that a reminder cannot change behaviour.

Because tau(x) is known, we can evaluate any targeting rule against the
oracle and attribute losses to estimation error vs. fairness constraints.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

COVARIATES = ["rural", "log_income", "distance_km", "mother_edu", "prior_visit"]
GROUP_COL = "rural"


@dataclass
class Population:
    X: pd.DataFrame
    y0: np.ndarray  # P(Y=1 | control)
    tau: np.ndarray  # true CATE on the probability scale
    group: np.ndarray  # protected group label (0 = urban, 1 = rural)

    @property
    def n(self) -> int:
        return len(self.tau)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def simulate_population(n: int = 5000, rural_share: float = 0.4, seed: int = 0) -> Population:
    rng = np.random.default_rng(seed)
    rural = rng.binomial(1, rural_share, n)

    # Rural households: poorer, further from clinics, fewer years of schooling.
    log_income = rng.normal(loc=np.where(rural == 1, 7.6, 8.4), scale=0.5, size=n)
    distance_km = rng.gamma(shape=np.where(rural == 1, 3.0, 1.5),
                            scale=np.where(rural == 1, 4.0, 1.5), size=n)
    mother_edu = np.clip(rng.normal(np.where(rural == 1, 6.0, 9.5), 3.0, n), 0, 16)
    prior_visit = rng.binomial(1, _sigmoid(0.3 * (log_income - 8) - 0.05 * distance_km + 0.5), n)

    X = pd.DataFrame({
        "rural": rural,
        "log_income": log_income,
        "distance_km": distance_km,
        "mother_edu": mother_edu,
        "prior_visit": prior_visit,
    })

    # Baseline outcome probability: higher for richer, closer, better-educated.
    y0 = _sigmoid(-0.4 + 0.5 * (log_income - 8) - 0.06 * distance_km
                  + 0.08 * (mother_edu - 8) + 0.8 * prior_visit)

    # True CATE: an inverted-U in distance. A reminder/incentive helps most
    # at moderate distance (3-6 km): close households attend anyway, remote
    # ones face a binding travel constraint a nudge cannot remove. Poorer
    # households respond more; prior attenders respond less. Because rural
    # households are concentrated at long distances, unconstrained targeting
    # on tau will under-serve them relative to their population share.
    access_curve = np.exp(-((distance_km - 4.5) ** 2) / (2 * 3.5 ** 2))
    raw = 0.03 + 0.22 * access_curve + 0.04 * (8.2 - log_income) - 0.04 * prior_visit
    raw += rng.normal(0, 0.02, n)  # idiosyncratic component
    tau = np.clip(raw, -0.02, 1 - y0 - 1e-3)

    return Population(X=X, y0=y0, tau=tau, group=rural)


def draw_trial(pop: Population, p_treat: float = 0.5, seed: int = 1) -> pd.DataFrame:
    """Draw a randomised trial from the population: assign treatment at
    random, realise binary outcomes from potential-outcome probabilities."""
    rng = np.random.default_rng(seed)
    w = rng.binomial(1, p_treat, pop.n)
    p = pop.y0 + w * pop.tau
    y = rng.binomial(1, np.clip(p, 0, 1))
    df = pop.X.copy()
    df["W"] = w
    df["Y"] = y
    return df
