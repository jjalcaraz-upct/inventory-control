"""Shared policy evaluation helpers for one-SKU simulations."""

from __future__ import annotations

from random import Random
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from model.demand import DemandDistribution
from model.metrics import KPITracker
from model.state import SKUState


def _as_scalar(x: Any) -> float:
    return float(np.asarray(x, dtype=float).reshape(-1)[0])


def _cost_params(env_config: Mapping[str, Any]) -> dict[str, float]:
    return {
        "K_fix": float(env_config["K_fix"]),
        "v": float(env_config["v"]),
        "h": float(env_config["h"]),
        "p": float(env_config["p"]),
    }


def generate_real_trajectories(
    *,
    R: int,
    N_days: int,
    demand_model: DemandDistribution | Mapping[str, Any],
    lead_weights: Sequence[float],
    L_max: int,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate realized demand and lead times for one SKU."""
    demand = demand_model if isinstance(demand_model, DemandDistribution) else DemandDistribution(demand_model)
    lead_values = list(range(1, int(L_max) + 1))

    real_D = np.zeros((int(R), int(N_days)), dtype=float)
    real_L = np.zeros((int(R), int(N_days)), dtype=int)
    for replica in range(int(R)):
        rng = Random(int(base_seed) + 10000 * replica)
        for t in range(int(N_days)):
            real_D[replica, t] = float(demand.sample(t, 0, rng))
            real_L[replica, t] = int(rng.choices(lead_values, weights=lead_weights, k=1)[0])
    return real_D, real_L


def generate_initial_states(
    *,
    R: int,
    L_max: int,
    base_seed: int,
    on_hand_range: Sequence[float],
    pipeline_range: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Generate reproducible randomized initial states for one SKU."""
    on_low, on_high = float(on_hand_range[0]), float(on_hand_range[1])
    p_low, p_high = float(pipeline_range[0]), float(pipeline_range[1])

    init_on_hand = np.zeros((int(R),), dtype=float)
    init_pipeline = np.zeros((int(R), int(L_max)), dtype=float)
    for replica in range(int(R)):
        rng = Random(int(base_seed) + 30000 * replica + 7)
        init_on_hand[replica] = rng.uniform(on_low, on_high)
        for ell in range(int(L_max)):
            init_pipeline[replica, ell] = rng.uniform(p_low, p_high)
    return init_on_hand, init_pipeline


def _default_initial_ranges(env_config: Mapping[str, Any], L_max: int) -> tuple[tuple[float, float], tuple[float, float]]:
    if "initial_on_hand" in env_config or "initial_pipeline" in env_config:
        on_hand = float(env_config.get("initial_on_hand", 0.0))
        pipeline = [float(x) for x in env_config.get("initial_pipeline", [0.0] * int(L_max))]
        if len(pipeline) != int(L_max):
            pipeline = [0.0] * int(L_max)
        return (on_hand, on_hand), (min(pipeline), max(pipeline))

    q_min = float(env_config.get("q_min", 0.0))
    q_max = float(env_config.get("q_max", q_min))
    return (q_min, q_max), (q_min, q_max)


def build_scenario(
    *,
    R: int,
    N_days: int,
    env_config: Mapping[str, Any],
    base_seed: int,
) -> dict[str, Any]:
    """Build one evaluation scenario from env_config."""
    env_cfg = dict(env_config)
    L_max = int(env_cfg.get("Lmax", len(env_cfg.get("initial_pipeline", [])) or 1))
    real_D, real_L = generate_real_trajectories(
        R=int(R),
        N_days=int(N_days),
        demand_model=env_cfg.get("demand_spec", {"kind": "uniform", "low": 0.0, "high": 10.0}),
        lead_weights=env_cfg.get("lead_time_weights", [1.0] * L_max),
        L_max=L_max,
        base_seed=int(base_seed),
    )
    on_hand_range, pipeline_range = _default_initial_ranges(env_cfg, L_max)
    init_on_hand, init_pipeline = generate_initial_states(
        R=int(R),
        L_max=L_max,
        base_seed=int(base_seed),
        on_hand_range=on_hand_range,
        pipeline_range=pipeline_range,
    )
    return {
        "real_D": real_D,
        "real_L": real_L,
        "init_on_hand": init_on_hand,
        "init_pipeline": init_pipeline,
    }


def _simulate_one(
    *,
    model_name: str,
    policy: Any,
    replica: int,
    real_D: np.ndarray,
    real_L: np.ndarray,
    init_on_hand: np.ndarray,
    init_pipeline: np.ndarray,
    cost_params: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    compute_action = getattr(policy, "compute_action", None)
    if compute_action is None:
        raise TypeError("policy must expose compute_action(state)")

    state = SKUState(
        on_hand=float(init_on_hand[replica]),
        pipeline=[float(x) for x in init_pipeline[replica]],
    )
    tracker = KPITracker(
        K_fix=float(cost_params["K_fix"]),
        v=float(cost_params["v"]),
        h=float(cost_params["h"]),
        p=float(cost_params["p"]),
    )

    raw_rows: list[dict[str, Any]] = []
    ordering_cost = 0.0
    holding_cost = 0.0
    lost_sales_cost = 0.0
    total_lost_sales = 0.0
    n_orders = 0
    n_days = int(real_D.shape[1])

    for day in range(n_days):
        demand = float(real_D[replica, day])
        lead_time = int(real_L[replica, day])
        q = _as_scalar(compute_action(state))

        tr = state.step(demand=demand, q=q, L=lead_time)
        sales = float(tr["sales"])
        lost_sales = float(tr["lost_sales"])
        inventory_end = float(tr["I_next"])
        costs = tracker.record_step(
            q=q,
            demand=demand,
            sales=sales,
            lost_sales=lost_sales,
            inventory=inventory_end,
        )

        ordering_cost += float(costs["ordering_cost"])
        holding_cost += float(costs["holding_cost"])
        lost_sales_cost += float(costs["lost_sales_cost"])
        total_lost_sales += lost_sales
        n_orders += int(q > 0.0)

        raw_rows.append(
            {
                "replica": int(replica),
                "day": int(day),
                "model_name": model_name,
                "demand": demand,
                "lead_time": lead_time,
                "policy_order_qty": float(q),
                "sales": sales,
                "lost_sales": lost_sales,
                "inventory_end": inventory_end,
                "ordering_cost": float(costs["ordering_cost"]),
                "holding_cost": float(costs["holding_cost"]),
                "lost_sales_cost": float(costs["lost_sales_cost"]),
                "step_cost": float(costs["step_cost"]),
            }
        )

    summary = tracker.summary()
    out = {
        "replica": int(replica),
        "model_name": model_name,
        "total_cost": float(summary["total_cost"]),
        "ordering_cost": float(ordering_cost),
        "holding_cost": float(holding_cost),
        "lost_sales_cost": float(lost_sales_cost),
        "fill_rate": float(summary["fill_rate"]),
        "total_lost_sales": float(total_lost_sales),
        "avg_inventory": float(summary["avg_inventory"]),
        "n_orders": int(n_orders),
        "stockout_count": int(summary["stockout_count"]),
        "n_days": int(n_days),
    }
    return out, raw_rows


def run_one(
    *,
    model_name: str,
    policy: Any,
    replica: int,
    real_D: Any,
    real_L: Any,
    init_on_hand: Any,
    init_pipeline: Any,
    cost_params: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one replica from precomputed trajectories."""
    out, _ = _simulate_one(
        model_name=model_name,
        policy=policy,
        replica=int(replica),
        real_D=np.asarray(real_D, dtype=float),
        real_L=np.asarray(real_L, dtype=int),
        init_on_hand=np.asarray(init_on_hand, dtype=float).reshape(-1),
        init_pipeline=np.asarray(init_pipeline, dtype=float),
        cost_params=cost_params,
    )
    return out


def _aggregate_summary(kpi_df: pd.DataFrame) -> dict[str, Any]:
    agg = {"n_replicas": int(len(kpi_df))}
    if kpi_df.empty:
        return agg
    for col in kpi_df.select_dtypes(include=["number"]).columns:
        agg[f"{col}_mean"] = float(kpi_df[col].mean())
    return agg


def run_replicas(
    *,
    model_name: str,
    policy: Any,
    real_D: Any,
    real_L: Any,
    init_on_hand: Any,
    init_pipeline: Any,
    cost_params: Mapping[str, Any],
) -> dict[str, Any]:
    """Run replicas in a simple for-loop: for replica in range(R)."""
    D = np.asarray(real_D, dtype=float)
    L = np.asarray(real_L, dtype=int)
    on_hand = np.asarray(init_on_hand, dtype=float).reshape(-1)
    pipeline = np.asarray(init_pipeline, dtype=float)

    raw_rows: list[dict[str, Any]] = []
    kpi_rows: list[dict[str, Any]] = []
    for replica in range(int(D.shape[0])):
        kpi, steps = _simulate_one(
            model_name=model_name,
            policy=policy,
            replica=replica,
            real_D=D,
            real_L=L,
            init_on_hand=on_hand,
            init_pipeline=pipeline,
            cost_params=cost_params,
        )
        kpi_rows.append(kpi)
        raw_rows.extend(steps)

    raw_df = pd.DataFrame(raw_rows)
    kpi_df = pd.DataFrame(kpi_rows)
    return {
        "raw_steps": raw_df,
        "kpi_summary": kpi_df,
        "aggregate_summary": _aggregate_summary(kpi_df),
    }


def evaluate_policy(
    *,
    model_name: str,
    policy: Any,
    scenario: Mapping[str, Any],
    env_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one policy on a prebuilt scenario."""
    return run_replicas(
        model_name=model_name,
        policy=policy,
        real_D=scenario["real_D"],
        real_L=scenario["real_L"],
        init_on_hand=scenario["init_on_hand"],
        init_pipeline=scenario["init_pipeline"],
        cost_params=_cost_params(env_config),
    )
