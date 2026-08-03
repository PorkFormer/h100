# Budgeted Success-Support Floor

BSSF protects only naturally completed, verifier-correct Base responses observed within a registered reference budget. It is not a length objective and does not anchor the current policy to the complete Base distribution.

For a cached witness \((x,y)\), the implementation uses temperature-consistent response-token sequence log probabilities:

\[
z_\theta(x,y)=\log\pi_\theta(y\mid x)-\log\pi_0(y\mid x),
\qquad
g_\theta(x,y)=[\log\alpha-z_\theta(x,y)]_+.
\]

Protected prompts are sampled uniformly without replacement, followed by one uniformly sampled eligible witness per prompt. The actor objective on support-update steps is

\[
\mathcal L_{\rm DAPO}+\lambda_t\widehat G_t.
\]

After the actor update:

\[
\bar G_t=\beta\bar G_{t-1}+(1-\beta)\widehat G_t,
\qquad
\lambda_{t+1}=\operatorname{clip}(\lambda_t+\eta_\lambda(\bar G_t-\delta),0,\lambda_{\max}).
\]

The response-distribution formulation has a convex one-sided constraint and the Base is strictly feasible when \(\alpha<1\) and \(\delta>0\). Transformer parameter-space training remains non-convex; the implementation seeks stochastic first-order KKT stationarity and makes no global-optimality claim. Exact witness likelihood is a finite-sample surrogate, not the complete success-event probability.

## Modes

- `off`: no cache read, support fields, forward, state, or loss change.
- `shadow`: standard DAPO update followed periodically by a forward-only witness metric batch. It never changes \(\lambda\).
- `dual`: periodically augments the actor-only batch and updates one projected dual scalar after the actor step.

Support witnesses never enter reward, advantage, dynamic sampling, validation, rollout dumps, or generated-token statistics. BSSF creates no Base worker, verifier call, rollout, continuation, rescue branch, length penalty, global KL, second optimizer, or grouped-Probe dependency.

## Numerical and optimizer contract

Sequence log probabilities and shortfalls accumulate in FP32. Training never exponentiates sequence ratios. Non-finite values fail closed. PPO and support masks are disjoint; support-only microbatches use a differentiable zero PPO term. The support loss is normalized by the global witness count and DP size.

The active trainer passes `num_mini_batch=K`, where \(K=N_{RL}/M_{PPO}\), instead of adding a support optimizer step. Consequently optimizer and LR-scheduler step counts match DAPO. The engine's `loss_mask` is the PPO-only mask, so support tokens do not dilute token-mean PPO normalization.

## Metrics

Scientific metrics use the `support_floor/` prefix: log-ratio summaries, mean shortfall, active fraction, residual, feasibility, \(\lambda\), EMA, and complementarity proxy. Actor loss terms use `actor/support_floor_*`; preparation/forward/token overhead uses `perf/support_floor_*`. These are witness-support statistics and must not be described as internal capability, q-ratio, accuracy loss, or forgotten-prompt fraction.
