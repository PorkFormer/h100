"""Diagnostic only: identical BF16 GEMM row shapes, no parameter replacement.

Full reference still computes every suffix token. Both views use the same
execution adapter. This is not evidence of equivalence to the native kernels,
nor a production efficiency claim. Padding is local to individual projections
and never changes attention, position IDs, objective or token masks.
"""
import types

import torch
import torch.nn.functional as F


def fixed_tile_forward(self, inputs):
    shape = inputs.shape
    flat = inputs.reshape(-1, shape[-1])
    tile = self._pppo_diagnostic_linear_tile
    outputs = []
    for start in range(0, flat.shape[0], tile):
        part = flat[start:start + tile]
        count = part.shape[0]
        part = F.pad(part, (0, 0, 0, tile - count))
        outputs.append(F.linear(part, self.weight, self.bias)[:count])
    return torch.cat(outputs, dim=0).reshape(*shape[:-1], self.out_features)


def install_fixed_tile_linear(model, tile):
    if tile <= 0:
        raise ValueError('tile must be positive')
    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module._pppo_diagnostic_linear_tile = tile
            module.forward = types.MethodType(fixed_tile_forward, module)
            count += 1
    if not count:
        raise ValueError('No linear modules patched')
    return count
