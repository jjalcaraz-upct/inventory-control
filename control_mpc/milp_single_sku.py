"""Pyomo MILP model builder for a single SKU."""

from __future__ import annotations

from typing import Any, Mapping

from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    Param,
    RangeSet,
    Var,
    minimize,
)

from model.state import SKUState


def _first_present(mapping: Mapping[str, Any], keys: tuple[str, ...], name: str) -> Any:
    # Accept small naming variants (e.g., lambda/lam) without duplicating parse logic.
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise KeyError(f"Missing required parameter: {name}")


def _state_to_initials(state: Any, Lmax: int) -> tuple[float, list[float]]:
    # Support both SKUState objects and plain dictionaries for easier integration.
    if isinstance(state, SKUState):
        I0 = float(state.on_hand)
        P0 = [float(x) for x in state.pipeline]
    elif isinstance(state, Mapping):
        I0 = float(_first_present(state, ("I0", "on_hand"), "I0/on_hand"))
        pipeline = _first_present(state, ("P0", "pipeline"), "P0/pipeline")
        P0 = [float(x) for x in pipeline]
    else:
        raise TypeError("state must be SKUState or mapping")

    if len(P0) != Lmax:
        raise ValueError("Initial pipeline length must equal Lmax")
    if I0 < 0 or any(x < 0 for x in P0):
        raise ValueError("Initial state must be non-negative")
    return I0, P0


def build_model(params: Mapping[str, Any], state: Any, scenarios: Mapping[str, Any]) -> ConcreteModel:
    """Build the single-SKU MPC MILP model from parameters, state, and scenarios."""
    H = int(_first_present(params, ("H",), "H"))
    Lmax = int(_first_present(params, ("Lmax", "L_max"), "Lmax"))
    K_fix = float(_first_present(params, ("K_fix",), "K_fix"))
    v = float(_first_present(params, ("v",), "v"))
    h = float(_first_present(params, ("h",), "h"))
    p = float(_first_present(params, ("p",), "p"))
    lam = float(_first_present(params, ("lambda", "lambda_", "lam"), "lambda"))
    q_min = float(_first_present(params, ("q_min",), "q_min"))
    q_max = float(_first_present(params, ("q_max",), "q_max"))
    I_target = float(_first_present(params, ("I_target",), "I_target"))

    # D and delta are the scenario tensors generated before model construction.
    D = scenarios["D"]
    delta = scenarios["delta"]

    if len(D) != H or len(delta) != H:
        raise ValueError("Scenario tensors must have horizon H in first dimension")
    Ns = len(D[0]) if H > 0 else 0
    if Ns <= 0:
        raise ValueError("At least one scenario is required")

    for k in range(H):
        if len(D[k]) != Ns:
            raise ValueError("All D rows must have Ns columns")
        if len(delta[k]) != Ns:
            raise ValueError("All delta rows must have Ns columns")
        for s in range(Ns):
            if len(delta[k][s]) != Lmax:
                raise ValueError("delta must have Lmax entries in third dimension")

    I0, P0 = _state_to_initials(state=state, Lmax=Lmax)

    m = ConcreteModel(name="single_sku_inventory_mpc")
    m.H = H
    m.Ns = Ns
    m.Lmax = Lmax

    # Kplus includes terminal inventory state at k = H.
    m.K = RangeSet(0, H - 1)
    m.Kplus = RangeSet(0, H)
    m.L = RangeSet(1, Lmax)
    m.S = RangeSet(0, Ns - 1)

    m.D = Param(m.K, m.S, initialize=lambda _m, k, s: float(D[k][s]))
    m.delta = Param(m.K, m.S, m.L, initialize=lambda _m, k, s, ell: int(delta[k][s][ell - 1]))
    m.I0 = Param(initialize=I0)
    m.P0 = Param(m.L, initialize=lambda _m, ell: float(P0[ell - 1]))

    m.q = Var(m.K, domain=NonNegativeReals)
    m.z = Var(m.K, domain=Binary)
    m.I = Var(m.Kplus, m.S, domain=NonNegativeReals)
    m.P = Var(m.Kplus, m.S, m.L, domain=NonNegativeReals)
    m.R = Var(m.K, m.S, domain=NonNegativeReals)
    m.lost = Var(m.K, m.S, domain=NonNegativeReals)
    m.u = Var(m.S, domain=NonNegativeReals)

    m.order_lb = Constraint(m.K, rule=lambda _m, k: q_min * _m.z[k] <= _m.q[k])
    m.order_ub = Constraint(m.K, rule=lambda _m, k: _m.q[k] <= q_max * _m.z[k])

    m.init_I = Constraint(m.S, rule=lambda _m, s: _m.I[0, s] == _m.I0)
    m.init_P = Constraint(m.S, m.L, rule=lambda _m, s, ell: _m.P[0, s, ell] == _m.P0[ell])

    m.arrivals = Constraint(m.K, m.S, rule=lambda _m, k, s: _m.R[k, s] == _m.P[k, s, 1])
    # Lost-sales linearization: lost >= demand - available.
    m.lost_sales = Constraint(
        m.K,
        m.S,
        rule=lambda _m, k, s: _m.lost[k, s] >= _m.D[k, s] - (_m.I[k, s] + _m.R[k, s]),
    )
    m.inventory_transition = Constraint(
        m.K,
        m.S,
        rule=lambda _m, k, s: _m.I[k + 1, s]
        == _m.I[k, s] + _m.R[k, s] - _m.D[k, s] + _m.lost[k, s],
    )

    def _pipeline_rule(_m: ConcreteModel, k: int, s: int, ell: int) -> Any:
        # Shift pipeline forward one slot and inject new order at sampled lead time.
        shifted = _m.P[k, s, ell + 1] if ell < _m.Lmax else 0.0
        return _m.P[k + 1, s, ell] == shifted + _m.q[k] * _m.delta[k, s, ell]

    m.pipeline_dynamics = Constraint(m.K, m.S, m.L, rule=_pipeline_rule)

    m.term_dev_pos = Constraint(m.S, rule=lambda _m, s: _m.u[s] >= _m.I[_m.H, s] - I_target)
    m.term_dev_neg = Constraint(m.S, rule=lambda _m, s: _m.u[s] >= I_target - _m.I[_m.H, s])

    # First-stage order costs are scenario independent.
    first_stage_cost = sum(K_fix * m.z[k] + v * m.q[k] for k in m.K)
    # Scenario-dependent costs are averaged to approximate expectation.
    expected_second_stage_cost = (1.0 / Ns) * sum(
        sum(h * m.I[k, s] + p * m.lost[k, s] for k in m.K) + lam * m.u[s] for s in m.S
    )
    m.obj = Objective(expr=first_stage_cost + expected_second_stage_cost, sense=minimize)

    return m
