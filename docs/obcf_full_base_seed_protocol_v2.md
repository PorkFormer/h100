# OBCF Full-Base Generation Seed Protocol v2

Protocol identifier: `obcf-full-base-generation-seed-protocol-v2`

## Scope and amendment rationale

This protocol audits frozen Base rollout artifacts used to construct an OBCF
reference cache. It does not alter generation, sampling, reward computation,
the OBCF objective, actor loss, or online rollout count.

The previous audit combined two requirements:

1. reproduce the frozen `offline-answer-timing-v1` 31-bit deterministic seed
   rule exactly; and
2. require every scalar seed in a full artifact to be globally unique.

Those requirements are structurally incompatible at sufficient scale. For
139,064 samples in a seed space of size `2^31`, the birthday-collision
expectation is approximately 4.50 pairs. Protocol v2 retains exact rule
reproduction and all sample identities while changing only the interpretation
of scalar collisions between different prompts. This amendment is independent
of verifier or reward outcomes.

## Hard requirements

### A. Globally unique sample identity

Every `(prompt_id, rollout_index)` pair must be globally unique. A `sample_uid`
is derived as a domain-separated SHA-256 digest over:

- source dataset fingerprint;
- `prompt_id`;
- `rollout_index`;
- `prompt_hash`; and
- generation config fingerprint.

Every derived or declared `sample_uid` must be globally unique and match the
derived value. Duplicate identities, duplicate UIDs, or UID mismatches are hard
failures.

### B. Exact frozen seed rule

Every row must satisfy:

```text
sampling_seed =
  little_endian_uint32(
    SHA256("offline-answer-timing-v1\0{master_seed}\0{prompt_id}\0{rollout_index}")[0:4]
  ) & 0x7fffffff
```

Any mismatch is a hard failure. Existing seeds must not be changed.

### C. Within-prompt scalar seed uniqueness

The eight rollouts belonging to one prompt must use eight distinct scalar
seeds. Any within-prompt seed collision is a hard failure.

### D. Cross-prompt scalar seed collisions

Equal scalar seeds belonging to different prompts must be detected, listed,
counted, and cryptographically bound into the generation-protocol
attestation. They are not a hard failure under protocol v2.

Cross-prompt collision rows must not be deleted, resampled, rewritten,
reseeded, or reindexed. Their complete sample identities remain in the
artifact.

### E. Generation identity and provenance

The audit remains fail closed for:

- prompt identity, hash, and exact prompt token IDs;
- response hash and exact response token IDs;
- response token count;
- generation config fingerprint;
- tokenizer fingerprint;
- chat-template fingerprint;
- model fingerprint;
- shard identity and non-overlap;
- source dataset fingerprint;
- generation errors; and
- expected prompt/rollout coverage.

Enrichment must preserve every source `(prompt_id, rollout_index)`, response
token list, sampling seed, and response hash exactly.

## Attestation binding

The sorted cross-prompt collision identities are hashed into a seed-collision
attestation. That hash is included in the generation protocol fingerprint.
Changing collision membership, identities, or interpretation therefore
changes the protocol fingerprint.

A legacy protocol artifact, including the original v1 failure artifact, cannot
be validated as a v2 passing attestation.

## Historical evidence

Protocol v2 does not supersede or rewrite the historical decision. A complete
audit record must retain, by immutable path and hash:

- the original Gate C `NO_GO` decision;
- the original seed collision audit;
- the original schema-enrichment failure log;
- the original report and archive;
- this amendment; and
- the new v2 audit and attestations.

The original Gate C result remains `FAIL`. Any subsequent Gate C v2 decision is
a separate decision under this explicit amendment.

## Forbidden interpretations

This amendment does not authorize outcome-dependent filtering, successful-row
selection, verifier-error suppression, response modification, seed changes,
new rollouts, adaptive-budget OBCF, witness replay, teacher forcing, or changes
to the OBCF mathematical objective.
