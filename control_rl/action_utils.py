"""Shared helpers for action-space handling in RL environments/wrappers."""

from __future__ import annotations

from typing import Any

import numpy as np


def to_scalar_action(action: Any) -> float:
    """Convert scalar-like actions to a Python float."""
    action_array = np.asarray(action)
    if action_array.ndim == 0:
        return float(action_array.item())
    if action_array.size == 1:
        return float(action_array.reshape(-1)[0])
    raise ValueError(f"Action must be scalar-like, got shape={action_array.shape}")


def build_action_quantities(q_min: float, q_max: float, action_step: float) -> np.ndarray:
    """Build an action grid (including 0.0) from bounds and step."""
    if q_min < 0 or q_max < 0 or q_max < q_min:
        raise ValueError("Require 0 <= q_min <= q_max")
    if action_step <= 0:
        raise ValueError("action_step must be > 0")

    quantities = [0.0]
    q = float(q_min) if q_min > 0 else float(action_step)
    while q <= float(q_max) + 1e-12:
        quantities.append(float(q))
        q += float(action_step)
    return np.array(sorted(set(quantities)), dtype=np.float32)
