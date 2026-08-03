# OBCF Capability Floor Cache

The cache contains prompt-level audit statistics only. It contains no Base response token, chain of thought, sequence log probability, witness trajectory, or teacher-forcing input.

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

`prompt_key` is the shared canonical SHA-256 identity over tokenizer fingerprint, chat-template fingerprint, and exact rendered prompt token IDs. Prompt, rollout, and score artifacts must agree on prompt identity. Every Base prompt must have exactly `base_rollouts_per_prompt` unique rollout indices, and every score must contain a boolean `prefix_reward_<reference_budget>` with an empty prefix-error field.

Only `prefix_reward_<B>` contributes to `base_prefix_success_count`. Full reward, finish reason, token-cap status, EOS, and natural completion do not affect protection or floor arithmetic. Rows below `support_threshold` are not stored.

The manifest binds the reference budget and count parameters, tokenizer/template identities, prompt/rollout/score artifact fingerprints, verifier implementation fingerprint, source commit, and a SHA-256 over real local Base weight files. `hashes.json` binds the manifest, Parquet data, and audit report. Loading recomputes file hashes, validates the strict Parquet schema and count arithmetic, and requires `audit_report.json` to have `passed=true` with internally consistent prompt and success-count histograms.

Cache mismatches, missing identities, duplicate identities, verifier errors, non-boolean rewards, malformed counts, unsupported schemas, and hash corruption fail closed. Relocating an unchanged cache is allowed; changing its scientific contents changes its cache fingerprint.
