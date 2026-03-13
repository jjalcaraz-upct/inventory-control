"""Multi-SKU controller entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from control_mpc.mpc_policy import MPCPolicy
from model.state import SKUState


@dataclass
class Controller:
    """Run multiple SKUs independently, one MPC policy per SKU."""

    states: List[SKUState]
    policies: List[MPCPolicy]

    def __post_init__(self) -> None:
        if len(self.states) == 0:
            raise ValueError("Controller requires at least one SKU")
        if len(self.states) != len(self.policies):
            raise ValueError("states and policies must have same length")

    @classmethod
    def from_shared_params(
        cls,
        skus: int,
        params: Mapping[str, Any],
        solver_name: str = "highs",
        base_scenario_seed: int = 0,
        initial_on_hand: float = 0.0,
        initial_pipeline: Sequence[float] | None = None,
    ) -> "Controller":
        """Build independent state/policy pairs for each SKU."""
        if skus <= 0:
            raise ValueError("skus must be > 0")
        Lmax = int(params["Lmax"])
        if initial_pipeline is None:
            initial_pipeline = [0.0] * Lmax
        if len(initial_pipeline) != Lmax:
            raise ValueError("initial_pipeline length must equal Lmax")

        states = [
            SKUState(on_hand=float(initial_on_hand), pipeline=[float(x) for x in initial_pipeline])
            for _ in range(skus)
        ]
        policies = [
            MPCPolicy(params=params, solver_name=solver_name, scenario_seed=base_scenario_seed + i)
            for i in range(skus)
        ]
        return cls(states=states, policies=policies)

    def step(self, demands: Sequence[float], lead_times: Sequence[int]) -> List[Dict[str, float]]:
        """Advance one day for each SKU independently."""
        if len(demands) != len(self.states):
            raise ValueError("demands length must match number of SKUs")
        if len(lead_times) != len(self.states):
            raise ValueError("lead_times length must match number of SKUs")

        outputs: List[Dict[str, float]] = []
        for i, (state, policy) in enumerate(zip(self.states, self.policies)):
            q = policy.compute_action(state)
            tr = state.step(demand=float(demands[i]), q=q, L=int(lead_times[i]))
            out = {"q": float(q), **tr, "I": float(state.on_hand)}
            outputs.append(out)
        return outputs
