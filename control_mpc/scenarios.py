"""Scenario generation module for inventory MPC."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Dict, List, Optional, Protocol, Sequence


class DemandGenerator(Protocol):
    """Interface for custom demand generation inside ScenarioGenerator.

    Args:
        k: Time index within the MPC horizon (0..H-1).
        s: Scenario index (0..Ns-1).
        rng: Random generator to sample stochastic demand in a reproducible way.
    Returns:
        Non-negative demand for (k, s).
    """

    def __call__(self, k: int, s: int, rng: Random) -> float:
        """Return demand value for period k and scenario s."""
        ...


@dataclass
class ScenarioGenerator:
    """Generate demand and lead-time trajectories for MPC."""

    H: int
    Ns: int
    Lmax: int
    demand_low: float = 0.0
    demand_high: float = 10.0
    lead_time_weights: Optional[Sequence[float]] = None
    seed: Optional[int] = None
    demand_generator: Optional[DemandGenerator] = None
    _rng: Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.H <= 0:
            raise ValueError("H must be > 0")
        if self.Ns <= 0:
            raise ValueError("Ns must be > 0")
        if self.Lmax <= 0:
            raise ValueError("Lmax must be > 0")
        if self.demand_high < self.demand_low:
            raise ValueError("demand_high must be >= demand_low")
        if self.lead_time_weights is None:
            self.lead_time_weights = [1.0] * self.Lmax
        if len(self.lead_time_weights) != self.Lmax:
            raise ValueError("lead_time_weights must have length Lmax")
        if any(w < 0 for w in self.lead_time_weights):
            raise ValueError("lead_time_weights must be non-negative")
        if sum(self.lead_time_weights) <= 0:
            raise ValueError("lead_time_weights must contain a positive value")
        if self.demand_generator is None:
            self.demand_generator = self._uniform_demand
        self._rng = Random(self.seed)

    def _uniform_demand(self, _k: int, _s: int, rng: Random) -> float:
        """Default demand generator.

        `_k` and `_s` are the same indices defined in `DemandGenerator`:
        period index and scenario index. They are prefixed with `_` here
        because this default implementation does not use them.
        """
        return rng.uniform(self.demand_low, self.demand_high)

    def generate(self) -> Dict[str, List]:
        """Return scenario tensors D[H,Ns], L[H,Ns], delta[H,Ns,Lmax]."""
        lead_times = list(range(1, self.Lmax + 1))

        D: List[List[float]] = [[0.0 for _ in range(self.Ns)] for _ in range(self.H)]
        L: List[List[int]] = [[1 for _ in range(self.Ns)] for _ in range(self.H)]
        delta: List[List[List[int]]] = [
            [[0 for _ in range(self.Lmax)] for _ in range(self.Ns)] for _ in range(self.H)
        ]

        for k in range(self.H):
            for s in range(self.Ns):
                demand = float(self.demand_generator(k, s, self._rng))
                if demand < 0:
                    raise ValueError("demand_generator must return non-negative demand")
                D[k][s] = demand
                lead = self._rng.choices(lead_times, weights=self.lead_time_weights, k=1)[0]
                L[k][s] = lead
                delta[k][s][lead - 1] = 1

        return {"D": D, "L": L, "delta": delta}
