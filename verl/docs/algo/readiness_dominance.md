# Readiness Dominance DAPO

Readiness Dominance DAPO is an experimental algorithm beside On-Policy Probe Credit Redistribution. It retains the existing DAPO Dynamic Sampling and terminal verifier reward, then compares answer-readiness profiles only among terminal-equivalent successful trajectories from the same prompt.

The implementation does not claim that the forced-answer Probe itself is novel. Its experimental object is the direct comparison at fixed absolute budgets and optional group-level advantage redistribution.

## Protocol

For configured absolute horizons such as:

```text
[256, 512, 1024, 2048]
```

a trajectory with active response length `L_i` is probed at horizon `h` only when `h < L_i`. The input remains:

```text
prompt token IDs
+ response token IDs[:h]
+ encode("\n\nAnswer:", add_special_tokens=false)
```

When `h >= L_i`, the cell is invalid and no request is generated. The implementation does not fill post-EOS cells with zero, one, or terminal correctness.

Terminal correctness comes only from finite binary reward extra info `acc`. Missing or invalid `acc` fails closed. `token_level_scores.sum(-1) > 0` is logged only as a disagreement diagnostic because score tensors may include overlong, format, or other shaping terms; disagreement does not replace `acc` or stop training.

Probe planning happens after final retained complete-group selection but before standard GRPO advantage calculation. It uses retained terminal success only. Positive-trajectory eligibility is added after standard GRPO has produced advantages.

## Direct readiness dominance

Let `S_ij` be the absolute horizons valid for both trajectories `i` and `j`. Trajectory `i` directly readiness-dominates `j` if and only if:

```text
i and j have the same prompt UID
AND both are terminal-success according to acc
AND both have positive standard-GRPO trajectory advantage
AND |S_ij| >= min_common_positions
AND V_i(h) >= V_j(h) for every h in S_ij
AND V_i(h) - V_j(h) >= strict_branch_margin / n for at least one h in S_ij
```

The default `min_common_positions` is two. With `n=4` and `strict_branch_margin=1`, strict improvement is at least `0.25`.

Profiles that cross are incomparable. Identical profiles are incomparable. Different prompts, terminal failures, nonpositive trajectories, and pairs with insufficient active common support are not compared.

Every pair can have different common support, so direct readiness dominance is not assumed to be transitive. The implementation never computes a transitive closure. The directly nondominated set contains eligible trajectories with no incoming edge in the direct pairwise matrix; diagnostic tensors retain the shorter `frontier_mask` name.

The algorithm does not optimize:

- AUC;
- a random interruption-budget distribution;
- token-level local potential differences;
- temporal returns or GAE-like traces;
- a difficulty or length model;
- a dual objective or new verifier reward.

## Modes

`algorithm.readiness_dominance.mode` supports:

- `off`: no dominance Probe, seed consumption, analysis, or advantage change. The inherited standard DAPO/GRPO path is used.
- `shadow`: generate fixed-horizon Probes and report direct dominance, common-support, and coverage metrics. Standard GRPO `advantages` and `returns` are asserted bitwise unchanged.
- `reweight`: save standard GRPO as `terminal_advantages`, then apply trajectory weights to eligible positive trajectories in groups containing a direct edge.

For the supported `token-mean` actor aggregation:

```text
directly dominated weight = 0
directly nondominated weight = group positive-mass scale
all other weights = 1
```

The scale preserves the group's original masked positive advantage mass. Standard GRPO active-token advantages must be constant within each trajectory and padding must be zero. NaN, infinity, zero frontier mass, invalid scale, or a mass-conservation violation fails closed. There is no scale clipping; mean, maximum, p50, p90, and p99 are logged.

The first version explicitly rejects actor aggregation modes other than `token-mean`. It does not claim that the same conservation rule applies to sequence-level aggregation.

Terminal `token_level_scores` and `token_level_rewards`, UIDs, trajectory IDs, retained ordering, and Dynamic Sampling decisions remain unchanged in every mode.

## Metrics and denominators

Rate metrics use fixed denominators:

- `eligible_group_rate`: groups with a comparable pair divided by all retained prompt groups.
- `group_with_dominance_rate`: groups with a direct edge divided by all retained prompt groups.
- `eligible_positive_rate`: eligible trajectories divided by all retained trajectories.
- `dominated_positive_rate`: directly dominated trajectories divided by eligible trajectories.
- `frontier_fraction`: directly nondominated trajectories divided by eligible trajectories.
- `profile_cross_rate`: crossing pairs divided by comparable unordered pairs.
- `pair_coverage_rate`: comparable pairs divided by same-group eligible unordered pairs.
- `probe_valid_cell_rate`: active terminal-success cells divided by all configured terminal-success trajectory/horizon cells.
- `terminal_success_score_disagreement_rate`: `acc`/score-sum disagreements divided by all retained trajectories.

Zero denominators report zero. Raw comparable/same-group pair counts and common-position mean/min/max are logged alongside positive mass, residual, and frontier-scale metrics.

## Experimental use

Readiness Dominance is experimental. Run `shadow` before considering `reweight`. Formal reweight training must wait for offline split-half agreement, repeated-Probe stability, and sensitivity analysis over `n`; `n=4` with a one-branch margin may be noisy.

The smoke launcher at `examples/readiness_dominance/train_dapo_qwen3_8b_h100x8_dominance_smoke.sh` is fixed to `shadow`, one optimizer step, and `n=2`. It invokes Python directly and does not submit or allocate a cluster job.
