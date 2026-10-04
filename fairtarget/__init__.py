"""fairtarget: budget-constrained intervention targeting with fairness floors,
from heterogeneous-treatment-effect estimates."""
from .simulate import Population, simulate_population, draw_trial, COVARIATES, GROUP_COL
from .estimate import TLearner, DRLearner, crossfit_cate
from .allocate import uniform, greedy, fair_floor, group_floor_from_population
from .evaluate import welfare, group_coverage, budget_share, benefit_share, summarise, bootstrap_gain

__all__ = [
    "Population", "simulate_population", "draw_trial", "COVARIATES", "GROUP_COL",
    "TLearner", "DRLearner", "crossfit_cate",
    "uniform", "greedy", "fair_floor", "group_floor_from_population",
    "welfare", "group_coverage", "budget_share", "benefit_share", "summarise", "bootstrap_gain",
]
