# Formal first-three-step milestone

Verified on 2026-11-12T02:47:53.426022+00:00. Ray job 22000000 remains RUNNING toward 300 steps. This milestone is not completion of the formal experiment. Current-session checks were performed at approximately five-minute intervals; no standalone monitoring script was used.

| Step | Actual cumulative updates | Gradient norm | Actor rows | Retained groups |
|---|---|---|---|---|
| 1 | 16.0 | 0.29846501257270575 | 2048 | 256 |
| 2 | 32.0 | 0.2747257351875305 | 2048 | 256 |
| 3 | 48.0 | 0.25291604036465287 | 2048 | 256 |

All three losses and gradient norms were finite. Each step used 6144 original candidates and 49152 continuations. All 147456 request IDs were unique, all 18432 original parents had branches 0–7, and policy versions matched steps 0/1/2. No full-mask fallback. Frozen configuration hash unchanged. Formal checkpoint first due at Step 50; Step 300 evaluation NOT RUN.
