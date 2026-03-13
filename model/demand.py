"""Demand distribution utilities for experiments and policy scenario generation."""

from __future__ import annotations

from random import Random
from typing import Any, Mapping, Tuple


class DemandDistribution:
    """Build a demand sampler from a density specification dictionary.

    Supported specs:
    - {"kind": "uniform", "low": <float>, "high": <float>}
    - {"kind": "normal_clipped", "mean": <float>, "std": <float>, "low": <float>, "high": <float>}
    """

    def __init__(self, spec: Mapping[str, Any]) -> None:
        self._spec = dict(spec)
        self._kind = str(self._spec.get("kind", "")).strip()
        if self._kind not in {"uniform", "normal_clipped"}:
            raise ValueError(f"Unsupported demand kind: {self._kind}")
        self._validate()

    def _validate(self) -> None:
        if self._kind == "uniform":
            low = float(self._spec["low"])
            high = float(self._spec["high"])
            if high < low:
                raise ValueError("uniform distribution requires high >= low")
            return

        mean = float(self._spec["mean"])
        std = float(self._spec["std"])
        if std <= 0:
            raise ValueError("normal_clipped distribution requires std > 0")
        low = float(self._spec.get("low", 0.0))
        high = float(self._spec.get("high", mean + 3.0 * std))
        if high < low:
            raise ValueError("normal_clipped distribution requires high >= low")

    def bounds(self) -> Tuple[float, float]:
        """Return (low, high) bounds used by this distribution."""
        if self._kind == "uniform":
            return float(self._spec["low"]), float(self._spec["high"])

        mean = float(self._spec["mean"])
        std = float(self._spec["std"])
        low = float(self._spec.get("low", 0.0))
        high = float(self._spec.get("high", mean + 3.0 * std))
        return low, high

    def sample(self, _k: int, _s: int, rng: Random) -> float:
        """Sample demand for period/scenario indices (k, s) using the provided RNG."""
        if self._kind == "uniform":
            low, high = self.bounds()
            return rng.uniform(low, high)

        mean = float(self._spec["mean"])
        std = float(self._spec["std"])
        low, high = self.bounds()
        x = rng.gauss(mean, std)
        if x < low:
            return low
        if x > high:
            return high
        return x

