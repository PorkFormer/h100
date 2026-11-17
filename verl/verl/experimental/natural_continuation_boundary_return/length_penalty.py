"""DAPO length shaping used only by the opt-in full-response NCBR mode."""
import math
from numbers import Real
import numpy as np


def parameters(kwargs, long_length):
    if kwargs is None:
        raise ValueError("full_response requires DAPO reward kwargs")
    buffer = kwargs.get("overlong_buffer_cfg")
    limit = kwargs.get("max_resp_len")
    if buffer is None or buffer.get("enable") is not True:
        raise ValueError("full_response requires enabled overlong penalty")
    width = buffer.get("len")
    factor = buffer.get("penalty_factor")
    if (type(limit) is not int or type(width) is not int or limit != long_length
            or not 0 < width <= limit):
        raise ValueError("full_response requires max_resp_len=L and 0 < buffer <= L")
    if isinstance(factor, bool) or not isinstance(factor, Real) or not math.isfinite(factor) or factor < 0:
        raise ValueError("length penalty factor must be finite and nonnegative")
    return limit, width, float(factor)


def penalty(lengths, kwargs, long_length):
    limit, width, factor = parameters(kwargs, long_length)
    values = np.asarray(lengths)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer) or np.any(values <= 0) or np.any(values > limit):
        raise ValueError("invalid full-response token lengths")
    return np.minimum(-(values.astype(np.float64) - (limit - width)) / width * factor, 0.0)
