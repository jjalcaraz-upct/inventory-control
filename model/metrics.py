"""Metrics and KPI helpers for inventory MPC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class KPITracker:
    """Aggregate simulation KPIs across SKUs and days."""

    K_fix: float
    v: float
    h: float
    p: float
    total_demand: float = 0.0
    total_sales: float = 0.0
    total_inventory: float = 0.0
    total_cost: float = 0.0
    stockout_count: int = 0
    periods: int = 0

    def compute_costs(self, q: float, inventory: float, lost_sales: float) -> Dict[str, float]:
        """Compute one-step cost components for the current KPI coefficients."""
        ordering_cost = (self.K_fix if float(q) > 0.0 else 0.0) + self.v * float(q)
        holding_cost = self.h * float(inventory)
        lost_sales_cost = self.p * float(lost_sales)
        step_cost = ordering_cost + holding_cost + lost_sales_cost
        return {
            "ordering_cost": float(ordering_cost),
            "holding_cost": float(holding_cost),
            "lost_sales_cost": float(lost_sales_cost),
            "step_cost": float(step_cost),
        }

    def record_step(
        self, q: float, demand: float, sales: float, lost_sales: float, inventory: float
    ) -> Dict[str, float]:
        """Update running KPI totals and return the one-step cost components."""
        costs = self.compute_costs(q=q, inventory=inventory, lost_sales=lost_sales)
        self.total_demand += float(demand)
        self.total_sales += float(sales)
        self.total_inventory += float(inventory)
        self.total_cost += float(costs["step_cost"])
        if float(lost_sales) > 0.0:
            self.stockout_count += 1
        self.periods += 1
        return costs

    def summary(self) -> Dict[str, Any]:
        """Return the KPI summary"""
        fill_rate = self.total_sales / self.total_demand if self.total_demand > 0.0 else 1.0
        avg_inventory = self.total_inventory / self.periods if self.periods > 0 else 0.0
        return {
            "total_cost": float(self.total_cost),
            "fill_rate": float(fill_rate),
            "avg_inventory": float(avg_inventory),
            "stockout_count": int(self.stockout_count),
        }
