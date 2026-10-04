import numpy as np
import pytest

from fairtarget import (COVARIATES, TLearner, crossfit_cate, draw_trial, fair_floor, greedy,
                        group_floor_from_population, simulate_population, uniform, welfare)


def test_population_probabilities_are_valid():
    pop = simulate_population(n=2000, seed=3)
    assert np.all(pop.y0 >= 0) and np.all(pop.y0 <= 1)
    assert np.all(pop.y0 + pop.tau <= 1 + 1e-9)
    assert pop.tau.mean() > 0  # intervention helps on average


def test_trial_is_balanced_and_binary():
    pop = simulate_population(n=4000, seed=1)
    df = draw_trial(pop, p_treat=0.5, seed=2)
    assert set(df["W"].unique()) == {0, 1}
    assert set(df["Y"].unique()) <= {0, 1}
    assert abs(df["W"].mean() - 0.5) < 0.03


def test_greedy_respects_budget_and_is_optimal_unconstrained():
    scores = np.array([0.5, 0.1, 0.9, 0.3])
    a = greedy(scores, 2)
    assert a.sum() == 2 and a[2] == 1 and a[0] == 1


def test_fair_floor_is_binary_meets_floor_and_budget():
    rng = np.random.default_rng(0)
    n, k = 1000, 300
    scores = rng.normal(size=n)
    group = rng.binomial(1, 0.4, n)
    scores[group == 1] -= 1.0  # make greedy under-serve group 1
    a = fair_floor(scores, group, k, {1: 0.5})
    assert set(np.unique(a)) <= {0, 1}
    assert a.sum() == k
    assert a[group == 1].sum() >= 0.5 * k - 1e-6


def test_fair_floor_equals_greedy_when_not_binding():
    rng = np.random.default_rng(1)
    scores = rng.normal(size=500)
    group = rng.binomial(1, 0.5, 500)
    k = 100
    a_g = greedy(scores, k)
    a_f = fair_floor(scores, group, k, {1: 0.0})
    assert welfare(a_f, scores) == pytest.approx(welfare(a_g, scores))


def test_fair_floor_rejects_infeasible():
    scores = np.ones(10)
    group = np.array([0] * 8 + [1] * 2)
    with pytest.raises(ValueError):
        fair_floor(scores, group, 8, {1: 0.9})
    with pytest.raises(ValueError):
        fair_floor(scores, group, 4, {0: 0.7, 1: 0.7})


def test_proportional_floors_sum_to_one():
    pop = simulate_population(n=1000, seed=0)
    f = group_floor_from_population(pop.group)
    assert sum(f.values()) == pytest.approx(1.0)


def test_crossfit_cate_is_positively_correlated_with_truth():
    pop = simulate_population(n=4000, seed=5)
    df = draw_trial(pop, seed=6)
    est = crossfit_cate(TLearner, df[COVARIATES], df["W"].values, df["Y"].values, n_folds=3)
    assert np.corrcoef(est, pop.tau)[0, 1] > 0.3


def test_uniform_budget():
    a = uniform(100, 25, seed=0)
    assert a.sum() == 25
