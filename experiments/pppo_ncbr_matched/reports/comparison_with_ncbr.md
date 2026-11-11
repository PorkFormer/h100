# Comparison with NCBR: incomplete experiment

PPPO formal training and downstream evaluation are NOT RUN because Gate 2 failed. This is a baseline/protocol record, not a completed efficacy or efficiency comparison.

Historical NCBR Step 300, H8192, 32 samples per question (recomputed from preserved trajectory scores):

| Benchmark | NCBR Avg@32 / Pass@1 estimate | NCBR Pass@32 | NCBR mean tokens | PPPO |
|---|---:|---:|---:|---|
| AIME2024 | 0.165625 | 0.466667 | 2245.46 | NOT RUN |
| AIME2025 | 0.144792 | 0.400000 | 2143.31 | NOT RUN |
| AMC23 | 0.698438 | 0.950000 | 1415.60 | NOT RUN |

All 3200 NCBR trajectory seeds were checked against the historical SHA256 seed formula; only the Boundary Return arm at Step 300 is included. Source hashes and per-question reference metrics are in provenance/ncbr_step300_h8192.json.

Intentional algorithm differences: PPPO original 8K response + eight binary-scored continuations, prefix-only clipped PPO, original-length denominator, and no dual clip/overlong or boundary reward correction. Dynamic sampling uses cumulative reward after all continuations; this is explicitly PPPO + DAPO Dynamic Sampling.
Environment/protocol differences: 256 unique holdout identities and every duplicate removed; historical NCBR trained on them. Original repeated-row order otherwise preserved. Twelve ambiguous normalized source identities remain in training but cannot enter holdout. Plateau is a disclosed reproduction completion rule. Historical NCBR resumed from Step 20; PPPO must start from Base.
Logging is local/offline following automatic approval rejection of online W&B disclosure. The existing Ray head object store remains 1 GiB versus historical 200 GB; full-scale memory/scheduling capacity is NOT RUN.
No PPPO GPU-hours speedup is claimed. cost_report.json contains diagnostic job costs only; optimized-token fractions cannot be interpreted as GPU-hours savings. Historical resumed-run costs must not be confused with a fresh 300-step run.
Both algorithms target the same two allocated A100-40GB nodes, Base model revision, verifier, optimizer and batch settings; config_diff.json contains the field audit.
