"""Run the full pipeline: simulate -> trial -> estimate CATE -> allocate -> evaluate.

Usage:  python run_experiment.py [--n 5000] [--budget 0.3] [--seed 0]
Writes figures to ./figures and prints the summary table.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fairtarget import (COVARIATES, DRLearner, TLearner, bootstrap_gain, crossfit_cate,
                        draw_trial, fair_floor, greedy, simulate_population, summarise, uniform)

GROUP_NAMES = {0: "urban", 1: "rural"}
FIG = Path(__file__).parent / "figures"


def main(n: int, budget_frac: float, seed: int) -> None:
    FIG.mkdir(exist_ok=True)
    pop = simulate_population(n=n, seed=seed)
    trial = draw_trial(pop, seed=seed + 1)
    X, W, Y = trial[COVARIATES], trial["W"].values, trial["Y"].values
    k = int(budget_frac * n)

    # ---- 1. Estimate CATE (out-of-fold) -------------------------------------
    tau_t = crossfit_cate(TLearner, X, W, Y, seed=seed)
    tau_dr = crossfit_cate(DRLearner, X, W, Y, seed=seed)
    print("CATE estimation quality (out-of-fold):")
    for name, est in [("T-learner", tau_t), ("DR-learner", tau_dr)]:
        rmse = np.sqrt(np.mean((est - pop.tau) ** 2))
        corr = np.corrcoef(est, pop.tau)[0, 1]
        print(f"  {name:11s} RMSE={rmse:.4f}  corr(true,est)={corr:.3f}")

    # ---- 2. Allocate under a fixed budget ------------------------------------
    rural_pop_share = pop.group.mean()
    allocs = {
        "uniform (RCT rule)": uniform(n, k, seed),
        "greedy T-learner": greedy(tau_t, k),
        "greedy DR-learner": greedy(tau_dr, k),
        f"DR + rural floor {rural_pop_share:.0%}": fair_floor(tau_dr, pop.group, k, {1: rural_pop_share}),
        "DR + rural floor 60%": fair_floor(tau_dr, pop.group, k, {1: 0.60}),
        "oracle (true CATE)": greedy(pop.tau, k),
    }
    table = summarise(allocs, pop.tau, pop.group, GROUP_NAMES)
    cis = {name: bootstrap_gain(a, pop.tau, seed=seed) for name, a in allocs.items()}
    table["gain_ci_low"] = [cis[r][1] for r in table.index]
    table["gain_ci_high"] = [cis[r][2] for r in table.index]
    pd.set_option("display.width", 160)
    print(f"\nBudget = {k} treatments ({budget_frac:.0%} of {n}). Expected additional good outcomes:")
    print(table.round(3).to_string())
    table.to_csv(FIG / "summary.csv")

    # ---- 3. Figure A: welfare + coverage per rule ----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    names = list(allocs)
    gains = [cis[r][0] for r in names]
    err = np.array([[cis[r][0] - cis[r][1], cis[r][2] - cis[r][0]] for r in names]).T
    axes[0].barh(names, gains, xerr=err, color="#4C72B0", capsize=3)
    axes[0].set_xlabel("Expected additional good outcomes")
    axes[0].set_title("Welfare by allocation rule (95% bootstrap CI)")
    axes[0].invert_yaxis()
    cov = table[[f"coverage_{g}" for g in GROUP_NAMES.values()]]
    cov.plot.barh(ax=axes[1], color=["#DD8452", "#55A868"])
    axes[1].set_xlabel("Share of group treated")
    axes[1].set_title("Who gets treated")
    axes[1].invert_yaxis()
    axes[1].legend(["urban", "rural"], loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "welfare_and_coverage.png", dpi=150)

    # ---- 4. Figure B: price of fairness curve --------------------------------
    floors = np.linspace(0.0, 0.9, 19)
    curve = []
    for f in floors:
        try:
            a = fair_floor(tau_dr, pop.group, k, {1: f})
        except ValueError:
            break
        curve.append((f, (a * pop.tau).sum(), a[pop.group == 1].mean()))
    curve = pd.DataFrame(curve, columns=["rural_floor", "gain", "rural_coverage"])
    oracle_gain = (allocs["oracle (true CATE)"] * pop.tau).sum()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve.rural_floor, curve.gain / oracle_gain, marker="o", color="#4C72B0")
    ax.axvline(rural_pop_share, ls="--", color="grey", label=f"rural population share ({rural_pop_share:.0%})")
    greedy_share = allocs["greedy DR-learner"][pop.group == 1].sum() / k
    ax.axvline(greedy_share, ls=":", color="#C44E52", label=f"unconstrained rural share ({greedy_share:.0%})")
    ax.set_xlabel("Minimum share of budget reserved for rural households")
    ax.set_ylabel("Welfare as share of oracle")
    ax.set_title("Price of fairness: welfare vs. rural budget floor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "price_of_fairness.png", dpi=150)
    curve.to_csv(FIG / "price_of_fairness.csv", index=False)

    # ---- 5. Figure C: budget sweep ------------------------------------------
    fracs = np.linspace(0.05, 0.6, 12)
    rows = []
    for fr in fracs:
        kk = int(fr * n)
        rows.append({
            "budget": fr,
            "uniform": (uniform(n, kk, seed) * pop.tau).sum(),
            "greedy DR": (greedy(tau_dr, kk) * pop.tau).sum(),
            "DR + proportional floor": (fair_floor(tau_dr, pop.group, kk, {1: rural_pop_share}) * pop.tau).sum(),
            "oracle": (greedy(pop.tau, kk) * pop.tau).sum(),
        })
    sweep = pd.DataFrame(rows).set_index("budget")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sweep.plot(ax=ax, marker="o")
    ax.set_xlabel("Budget (share of population treated)")
    ax.set_ylabel("Expected additional good outcomes")
    ax.set_title("Targeting gain over the RCT's uniform rule, by budget")
    fig.tight_layout()
    fig.savefig(FIG / "budget_sweep.png", dpi=150)
    sweep.to_csv(FIG / "budget_sweep.csv")

    # ---- 6. Figure D: estimated vs true CATE ---------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(pop.tau, tau_dr, s=6, alpha=0.4, c=pop.group, cmap="coolwarm")
    lim = [pop.tau.min() - 0.02, pop.tau.max() + 0.02]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("True CATE")
    ax.set_ylabel("DR-learner estimate (out-of-fold)")
    ax.set_title("CATE recovery (blue=urban, red=rural)")
    fig.tight_layout()
    fig.savefig(FIG / "cate_recovery.png", dpi=150)
    print(f"\nFigures written to {FIG}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--budget", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args.n, args.budget, args.seed)
