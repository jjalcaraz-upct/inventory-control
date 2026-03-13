"""State definitions for inventory MPC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class SKUState:
    """Single-SKU inventory state with on-hand and lead-time pipeline."""

    on_hand: float
    pipeline: List[float]

    def __post_init__(self) -> None:
        if self.on_hand < 0:
            raise ValueError("on_hand must be non-negative")
        if not self.pipeline:
            raise ValueError("pipeline must contain at least one lead-time position")
        if any(x < 0 for x in self.pipeline):
            raise ValueError("pipeline values must be non-negative")

    def receive(self) -> float:
        """Receive inventory scheduled for the current period."""
        received = self.pipeline[0]
        self.on_hand += received
        self.pipeline[0] = 0.0
        return received

    def shift(self) -> None:
        """Shift pipeline one period closer to arrival."""
        self.pipeline = self.pipeline[1:] + [0.0]

    def inject_order(self, q: float, L: int) -> None:
        """Inject new order quantity into the pipeline at lead-time position L (1-based)."""
        if q < 0:
            raise ValueError("q must be non-negative")
        if L < 1 or L > len(self.pipeline):
            raise ValueError("L must be within [1, len(pipeline)]")
        self.pipeline[L - 1] += q

    def step(self, demand: float, q: float, L: int) -> Dict[str, float]:
        """Advance one period and return key transition outputs."""
        if demand < 0:
            raise ValueError("demand must be non-negative")
        if q < 0:
            raise ValueError("q must be non-negative")

        received = self.receive()
        sales = min(self.on_hand, demand)
        lost_sales = demand - sales
        self.on_hand -= sales
        self.shift()
        self.inject_order(q=q, L=L)

        return {
            "sales": float(sales),
            "lost_sales": float(lost_sales),
            "received": float(received),
            "I_next": float(self.on_hand),
        }
