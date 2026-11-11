"""Fresh reward-only batches. Repeated parents are intentional."""
import copy
import numpy as np
import torch
from verl import DataProto


def scoring_batch(original, generations, pad_token_id):
    if len({g.request.request_id for g in generations}) != len(generations):
        raise ValueError("duplicate scoring branch")
    prompts = [g.request.prompt for g in generations]
    responses = [g.request.prefix + g.tail for g in generations]
    pw, rw = max(map(len, prompts)), max(1, max(map(len, responses)))
    size = len(generations)
    p = torch.full((size, pw), pad_token_id, dtype=torch.long)
    r = torch.full((size, rw), pad_token_id, dtype=torch.long)
    pm, rm = torch.zeros_like(p), torch.zeros_like(r)
    for i, (prompt, response) in enumerate(zip(prompts, responses, strict=True)):
        p[i, -len(prompt):] = torch.tensor(prompt)
        pm[i, -len(prompt):] = 1
        if response:
            r[i, :len(response)] = torch.tensor(response)
            rm[i, :len(response)] = 1
    attention = torch.cat((pm, rm), -1)
    # Whitelist metadata: no rm_scores/acc/reward_scores or stale nested extra_info.
    metadata = {}
    for key in ("data_source", "reward_model"):
        metadata[key] = np.asarray([copy.deepcopy(original.non_tensor_batch[key][g.request.parent])
                                    for g in generations], dtype=object)
    metadata["uid"] = np.asarray([g.request.request_id for g in generations], dtype=object)
    return DataProto.from_dict(tensors={"prompts": p, "responses": r, "input_ids": torch.cat((p, r), -1),
                                       "attention_mask": attention, "position_ids": (attention.cumsum(-1)-1).clamp(min=0),
                                       "response_mask": rm}, non_tensors=metadata,
                               meta_info={"boundary_reward_only": True})


def correctness(extra, expected):
    if "acc" not in extra:
        raise ValueError("verifier did not return correctness acc")
    values = np.asarray(extra["acc"], dtype=np.float64)
    if values.shape != (expected,) or not np.isin(values, (0, 1)).all():
        raise ValueError("verifier acc must be one binary value per row")
    return values
