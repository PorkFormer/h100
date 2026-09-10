"""DAPO group diagnostics, deliberately outside the reward hook."""
import numpy as np

BOUNDARY_NUMERIC_TOLERANCE = 1.0e-6

def group_unlocked_mask(uids, short_values, boundary_values):
    grouped_short: dict[str, list[float]] = {}
    grouped_boundary: dict[str, list[float]] = {}
    for uid, short_value, boundary_value in zip(
        np.asarray(uids, dtype=object).tolist(),
        np.asarray(short_values).tolist(),
        np.asarray(boundary_values).tolist(),
        strict=True,
    ):
        grouped_short.setdefault(str(uid), []).append(float(short_value))
        grouped_boundary.setdefault(str(uid), []).append(float(boundary_value))
    unlocked_uids = {
        uid
        for uid, values in grouped_short.items()
        if np.std(np.asarray(values, dtype=np.float64)) <= BOUNDARY_NUMERIC_TOLERANCE
        and np.std(np.asarray(grouped_boundary[uid], dtype=np.float64)) > BOUNDARY_NUMERIC_TOLERANCE
    }
    group_unlocked = np.asarray([str(uid) in unlocked_uids for uid in uids], dtype=bool)
    return group_unlocked
