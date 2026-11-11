"""Diagnostic fixed-shape BF16 GEMMs with a single autograd edge per linear.

Full reference still evaluates all suffix tokens. No attention/objective changes.
Both views use this adapter; this is not native-kernel equivalence evidence.
"""
import types

import torch
import torch.nn.functional as F


class FixedTileLinear(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.bfloat16)
    def forward(ctx, inputs, weight, bias, tile):
        ctx.save_for_backward(inputs, weight)
        ctx.tile = tile
        ctx.has_bias = bias is not None
        flat = inputs.reshape(-1, inputs.shape[-1])
        outputs = []
        for start in range(0, flat.shape[0], tile):
            part = flat[start:start + tile]
            count = part.shape[0]
            outputs.append(F.linear(F.pad(part, (0, 0, 0, tile-count)), weight, bias)[:count])
        return torch.cat(outputs).reshape(*inputs.shape[:-1], weight.shape[0])

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        inputs, weight = ctx.saved_tensors
        flat = inputs.reshape(-1, inputs.shape[-1])
        grad = grad_output.reshape(-1, weight.shape[0])
        accum_dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
        dw = torch.zeros_like(weight, dtype=accum_dtype)
        db = torch.zeros(weight.shape[0], device=weight.device, dtype=accum_dtype) if ctx.has_bias else None
        dx = []
        for start in range(0, flat.shape[0], ctx.tile):
            x = flat[start:start + ctx.tile]
            g = grad[start:start + ctx.tile]
            count = x.shape[0]
            x = F.pad(x, (0, 0, 0, ctx.tile-count))
            g = F.pad(g, (0, 0, 0, ctx.tile-count))
            dx.append(torch.mm(g, weight)[:count])
            dw.add_(torch.mm(g.T, x).to(accum_dtype))
            if db is not None:
                db.add_(g.to(accum_dtype).sum(0))
        return torch.cat(dx).reshape_as(inputs), dw.to(weight.dtype), None if db is None else db.to(weight.dtype), None


def fixed_tile_forward(self, inputs):
    return FixedTileLinear.apply(inputs, self.weight, self.bias, self._pppo_diagnostic_linear_tile)


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
