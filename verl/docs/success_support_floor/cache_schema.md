# BSSF cache contract

A cache is an immutable directory:

```text
manifest.json
prompts.parquet
witnesses.parquet
hashes.json
validation_report.json
```

`hashes.json` contains SHA-256 hashes of the three contract files and their combined cache fingerprint. Training checks hashes, schema version, algorithm name, budget, support threshold, tokenizer vocabulary/special-token fingerprint, chat-template fingerprint, log-probability temperature, EOS convention, protected-prompt counts, witness counts, rewards, natural finish reasons, lengths, and finite reference probabilities.

The prompt key is

```text
SHA256(tokenizer_fingerprint || chat_template_fingerprint || canonical_prompt_token_ids)
```

Dataset row indices are audit metadata, not runtime identity. Ground truth is retained in the source audit manifest and is not actor input.

## Eligibility

A witness requires positive full and budget-prefix verifier rewards, response length at most the reference budget, a natural EOS/stop finish, no token-cap hit, and no generation or scoring error. A prompt is protected only when it has at least `support_threshold` eligible witnesses. Duplicate response sequences are retained because their multiplicity is evidence about Base conditional support.

## Reference probability calibration

The builder recomputes every saved witness with Base teacher forcing using the cached response tokens, response-token sum convention, configured temperature, and FP32 log-softmax/summation. Rollout `cumulative_logprob` is diagnostic only. Before Shadow mode, run the validator; defaults require mean absolute per-token error at most `1e-5`, maximum at most `1e-4`, and no non-finite/token mismatch.

Build and validate:

```bash
python tools/success_support_floor/build_cache.py \
  --rollout-glob '/path/rollouts/*.parquet' \
  --score-glob '/path/scores/*.parquet' \
  --prompt-manifest /path/prompts.parquet \
  --reference-model-path /path/base \
  --tokenizer-path /path/base \
  --reference-budget 2048 \
  --support-threshold 2 \
  --logprob-temperature 1.0 \
  --output-dir /path/bssf-cache

python tools/success_support_floor/validate_cache_logprobs.py \
  --cache-path /path/bssf-cache \
  --reference-model-path /path/base
```
