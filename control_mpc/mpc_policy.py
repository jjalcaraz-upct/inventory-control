"""MPC policy for selecting replenishment actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from pyomo.environ import SolverFactory, value
from pyomo.opt import TerminationCondition

from control_mpc.milp_single_sku import build_model
from control_mpc.scenarios import DemandGenerator, ScenarioGenerator
from model.state import SKUState


@dataclass
class MPCPolicy:
    """Single-SKU MPC policy that optimizes and returns the first action q0."""

    params: Mapping[str, Any]
    solver_name: str = "highs"
    scenario_seed: int = 0
    demand_generator: Optional[DemandGenerator] = None
    _scenario_generator: ScenarioGenerator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        H = int(self.params["H"])
        Ns = int(self.params["Ns"])
        Lmax = int(self.params["Lmax"])
        demand_low = float(self.params.get("scenario_demand_low", 0.0))
        demand_high = float(self.params.get("scenario_demand_high", 10.0))
        lead_time_weights = self.params.get("scenario_lead_time_weights")
        if lead_time_weights is None:
            lead_time_weights = [1.0] * Lmax
        else:
            lead_time_weights = [float(w) for w in lead_time_weights]
            if len(lead_time_weights) != Lmax:
                raise ValueError("scenario_lead_time_weights length must equal Lmax")

        self._scenario_generator = ScenarioGenerator(
            H=H,
            Ns=Ns,
            Lmax=Lmax,
            demand_low=demand_low,
            demand_high=demand_high,
            lead_time_weights=lead_time_weights,
            seed=self.scenario_seed,
            demand_generator=self.demand_generator,
        )

    def _solve(self, model: Any) -> None:
        solver = SolverFactory(self.solver_name)
        if not solver.available(False):
            raise RuntimeError(f"Solver '{self.solver_name}' is not available")
        result = solver.solve(model, tee=False)
        term = result.solver.termination_condition
        if term not in (TerminationCondition.optimal, TerminationCondition.feasible):
            raise RuntimeError(f"MILP solve failed with termination condition: {term}")

    def compute_action(self, state: SKUState) -> float:
        """Generate scenarios, solve MPC MILP, and return first action q0."""
        scenarios = self._scenario_generator.generate()
        model = build_model(params=self.params, state=state, scenarios=scenarios)
        self._solve(model)
        q0 = float(value(model.q[0]))
        return max(0.0, q0)
