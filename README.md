# fairtarget

Fairtarget is a small python package for deciding who should receive a scarce intervention when a programme can't afford to treat everyone, and for measuring what it costs to make that allocation equitable. The input is evidence from a randomized trial; the output is a treated set under a budget, plus a report on how much good it does and who it reaches.

This package answers two questions a programme manager actually has:

1. **Who should get it?** Rank on estimated conditional treatment effects (CATE) and treat the top-k, instead of the trial's uniform assignment.
2. **What does equity cost?** Impose minimum budget shares per group (e.g. rural households) as linear constraints and measure the welfare lost, or gained, relative to unconstrained targeting.

Because the demo population has a *known* CATE, every rule is scored against the oracle and the loss is decomposed into **estimation error** and **price of fairness**.

## The Problem it Addresses

Randomized trial tells you whether an intervention works on average. Programme manager faces a different question: with money to reach 30% of the population, which 30%? Treating people at random, as the trail did, leaves gains on the table if some people respond more than others. Treating purely by predicted response can quietly exclude whole groups, such as remote rural households, if the estimated effects there are smaller or noisier. The package makes that tension explicit and quantifies it.

## How it Works
1. Estimate heterogeneous effects. From trial data (covariates, treatment indicator, outcome), fit models that predict each individual's conditional average treatment effect. Two estimators are provided: a T-learner and a doubly-robust learner, both cross-fitted so no one is scored by a model that saw their own outcome
2. Allocate under constraints. Treat the k people with the highest predicted effect, optionally subject to minimum budget shares per group. The constrained version is solved as a linear program whose structure guarantees a clean 0/1 solution.
3. Evaluate against a known truth. The package includes a simulator that generates a population where every individual's effect is known. That gives an oracle allocation to benchmark against, so the welfare shortfall of any rule can be split into two parts: how much comes from imperfect estimation and how much from the fairness constraint.
4. Report in policy terms. Expected additional good outcomes, bootstrap confidence intervals, share of each group treated, share of total benefit accruing to each group, and a price-of-fairness curve showing welfare as the equity floor tightens. 

## Quick start

```bash
pip install -r requirements.txt
python run_experiment.py            # writes figures/ and prints the summary table
python -m pytest tests              # 9 tests, ~3 s
jupyter notebook demo.ipynb         # narrated walkthrough
```

## Headline result (n = 10 000, budget = 30 %, seed 0)

| rule | expected additional outcomes | share of oracle | rural coverage |
|---|---:|---:|---:|
| uniform (RCT rule) | 427 | 0.63 | 0.29 |
| greedy on DR-learner CATE | 579 | 0.86 | 0.19 |
| DR + proportional rural floor (40 %) | 583 | 0.86 | 0.30 |
| DR + rural floor 60 % | 554 | 0.82 | 0.45 |
| oracle (true CATE) | 676 | 1.00 | 0.27 |

Targeting lifts outcomes ~36 % over uniform assignment. The **proportional** rural floor is essentially free (+0.6 %), because the CATE estimators under-shoot rural effects and the constraint pulls the allocation back toward what a perfect estimator would do. Fairness and efficiency are not always in tension; the price only becomes material once floors exceed what the true effects justify.

The general lesson: equity constraints can act as a corrective against estimation bias rather than a pure efficiency loss, and a tool should be able to show a manager which regime they're in.

![](figures/welfare_and_coverage.png)
![](figures/price_of_fairness.png)

## Package layout

```
fairtarget/
  simulate.py   population with known CATE; draw_trial() yields an RCT
  estimate.py   TLearner, DRLearner (cross-fitted, sklearn only), crossfit_cate()
  allocate.py   uniform(), greedy(), fair_floor()  [LP with integral relaxation]
  evaluate.py   welfare, coverage, benefit share, bootstrap CIs, summarise()
run_experiment.py   end-to-end run, writes figures/
demo.ipynb          executed notebook
tests/              pytest suite
```

## Method notes

* **CATE estimation.** T-learner and doubly-robust learner with K-fold cross-fitting; allocation always uses out-of-fold predictions so ranking is never on in-sample fits.
* **Allocation.** `fair_floor` solves max Σ sᵢaᵢ s.t. Σaᵢ ≤ k and Σ_{i∈g} aᵢ ≥ floor_g·k. One budget row plus group rows that partition the units gives a totally unimodular matrix, so `scipy.optimize.linprog` returns a 0/1 solution without integer programming.
* **Evaluation.** Welfare = Σ true CATE over the treated set (expected extra good outcomes). Bootstrap CIs over individuals.

## What this is not (yet)

Synthetic data only; one protected attribute; single-round decision; two-stage predict-then-optimise rather than decision-focused learning.

## References
Athey & Wager (2021), "Policy Learning with Observational Data," *Econometrica* (https://doi.org/10.3982/ECTA15732)

Chernozhukov et al. (2018), "Double/debiased machine learning for treatment and structural parameters," *Econometrics Journal* (https://doi.org/10.1111/ectj.12097)

Kennedy (2020), "Towards optimal doubly robust estimation of heterogeneous causal effects," *Electronic Journal of Statistics* (https://doi.org/10.48550/arXiv.2004.14497)

Killian, Jain, Jia, Amar, Huang, & Tambe (2023), "Equitable Restless Multi-Armed Bandits: A General Framework Inspired By Digital Health" (https://doi.org/10.48550/arXiv.2308.09726)

Kitagawa & Tetenov (2018), "Who Should Be Treated? Empirical Welfare Maximization Methods for Treatment Choice," *Econometrica* (https://doi.org/10.3982/ECTA13288)

Künzel, Sekhon, Bickel, & Yu (2019), "Metalearners for estimating heterogeneous treatment effects using machine learning," *PNAS* (https://doi.org/10.1073/pnas.1804597116)

Manski (2004), "Statistical Treatment Rules for Heterogeneous Populations," *Econometrica* (https://doi.org/10.1111/j.1468-0262.2004.00530.x)

Verma, Zhao, Shah, Boehmer, Taneja, & Tambe (2024), "Group Fairness in Predict-Then-Optimize Settings for Restless Bandits," *UAI* (https://proceedings.mlr.press/v244/verma24a.html)
