# OBCF Capability Floor Cache

The cache contains prompt-level audit statistics only. It contains no Base response token, chain of thought, sequence log probability, witness trajectory, or teacher-forcing input.

The current cache format is schema version 2. Schema version 1 is rejected in both shadow and dual modes because it does not bind the offline floor event to the online exact-prefix protocol. Off mode requires neither a cache nor an attestation.

## Layout

```text
<cache_root>/
  manifest.json
  prompts.parquet
  hashes.json
  audit_report.json
```

`prompts.parquet` has the strict fields:

```text
prompt_key
prompt_id
original_dataset_index
prompt_hash
prompt_token_ids
prompt_token_count
base_rollout_count
base_prefix_success_count
q_reference
floor_count
capability_floor
```

`prompt_key` is the shared canonical SHA-256 identity over tokenizer fingerprint, chat-template fingerprint, and exact rendered prompt token IDs. The builder derives tokenizer/template fingerprints from the local Base model and rejects conflicting supplied values; prompt artifacts must attest to both identities. Prompt, rollout, and score artifacts must agree on prompt identity. Every Base prompt must have exactly `base_rollouts_per_prompt` unique rollout indices, and every score must contain a boolean `prefix_reward_<reference_budget>` with an empty prefix-error field.

Only `prefix_reward_<B>` contributes to `base_prefix_success_count`. Full reward, finish reason, token-cap status, EOS, and natural completion do not affect protection or floor arithmetic. Rows below `support_threshold` are not stored.

The schema-v2 manifest binds the reference budget and count parameters, tokenizer/template identities, prompt/rollout/score artifact file-set fingerprints, reward-manager/verifier source fingerprint, `prefix_protocol_fingerprint`, source commit, and a SHA-256 over real local Base weight files. Each score row must attest to the same verifier fingerprint and source commit. The verifier fingerprint covers registered verifier sources or exact importlib/custom module bytes, manager selection, custom and manager keyword arguments, and sandbox settings. Partitioned Parquet directories, quoted globs, and shell-expanded globs are loaded in sorted part order with identical schemas and hashed with their part names and bytes. Online cache loading independently recomputes both the configured local reward-pipeline fingerprint and the exact-prefix protocol fingerprint and requires exact matches. `hashes.json` binds the manifest, Parquet data, and audit report. Loading recomputes file hashes, validates the strict Parquet schema and count arithmetic, and requires `audit_report.json` to have `passed=true` with internally consistent prompt, success-count, and floor-count histograms.

`build_floor_cache.py` requires `--event-equivalence-attestation`. The schema-1 attestation must have `passed=true`, the same budget and tokenizer/template/verifier fingerprints, exact prompt/rollout/historical-score file-set hashes, zero directional mismatches, zero historical and recomputed errors, and `exact_match_count == row_count`. Its protocol fingerprint is copied into the cache manifest only after independently recomputing and matching it. Missing or failed attestations and all source, budget, verifier, or protocol mismatches fail closed.

Cache mismatches, missing identities, duplicate identities, verifier errors, non-boolean rewards, malformed counts, unsupported schemas, and hash corruption fail closed. Relocating an unchanged cache is allowed; changing its scientific contents changes its cache fingerprint.

Immutable legacy audit artifacts that predate the native provenance columns may be converted with `--legacy-artifact-attestation`. The attestation must have `schema_version=1`, `passed=true`, the single `config_fingerprint` carried by every prompt/rollout/score row, the full source commit, locally verified tokenizer/template and verifier fingerprints, and exact prompt/rollout/score artifact fingerprints. The builder validates every field and enriches copies in memory; it never rewrites the source artifacts. Legacy rows without this hash-bound attestation fail closed.
