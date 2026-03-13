#!/usr/bin/env python
"""Toy runner scaffold for the inventory MPC prototype."""

import argparse
import sys
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control_mpc.controller import Controller
from control_mpc.milp_single_sku import build_model
from model.metrics import KPITracker
from control_mpc.scenarios import ScenarioGenerator
from model.state import SKUState


def run_smoke_state() -> None:
    """Small deterministic smoke test for Step 1 state transitions."""
    state = SKUState(on_hand=5.0, pipeline=[2.0, 3.0, 0.0])
    out1 = state.step(demand=6.0, q=4.0, L=2)
    assert out1 == {"sales": 6.0, "lost_sales": 0.0, "received": 2.0, "I_next": 1.0}
    assert state.on_hand == 1.0
    assert state.pipeline == [3.0, 4.0, 0.0]

    out2 = state.step(demand=10.0, q=1.0, L=1)
    assert out2 == {"sales": 4.0, "lost_sales": 6.0, "received": 3.0, "I_next": 0.0}
    assert state.on_hand == 0.0
    assert state.pipeline == [5.0, 0.0, 0.0]

    print("smoke-state: OK")


def run_smoke_scenarios() -> None:
    """Smoke test for Step 2 scenario generation."""
    H, Ns, Lmax = 4, 3, 3
    gen = ScenarioGenerator(H=H, Ns=Ns, Lmax=Lmax, demand_low=5.0, demand_high=12.0, seed=123)
    out = gen.generate()

    D = out["D"]
    L = out["L"]
    delta = out["delta"]

    assert len(D) == H
    assert len(L) == H
    assert len(delta) == H

    for k in range(H):
        assert len(D[k]) == Ns
        assert len(L[k]) == Ns
        assert len(delta[k]) == Ns
        for s in range(Ns):
            assert 5.0 <= D[k][s] <= 12.0
            assert 1 <= L[k][s] <= Lmax
            assert len(delta[k][s]) == Lmax
            assert sum(delta[k][s]) == 1
            assert delta[k][s][L[k][s] - 1] == 1

    print("smoke-scenarios: OK")


def run_smoke_milp() -> None:
    """Smoke test for Step 3 MILP model construction."""
    H, Ns, Lmax = 4, 3, 3
    params = {
        "H": H,
        "Lmax": Lmax,
        "K_fix": 10.0,
        "v": 1.0,
        "h": 0.2,
        "p": 5.0,
        "lambda": 0.5,
        "q_min": 0.0,
        "q_max": 50.0,
        "I_target": 8.0,
    }
    state = SKUState(on_hand=7.0, pipeline=[2.0, 1.0, 0.0])
    scenarios = ScenarioGenerator(
        H=H, Ns=Ns, Lmax=Lmax, demand_low=3.0, demand_high=12.0, seed=7
    ).generate()

    # Build only (no solve yet): Step 3 validates model construction and dimensions.
    model = build_model(params=params, state=state, scenarios=scenarios)

    # Quick structural checks to catch indexing mistakes in sets/variables.
    assert len(list(model.K)) == H
    assert len(list(model.Kplus)) == H + 1
    assert len(list(model.S)) == Ns
    assert len(list(model.L)) == Lmax
    assert len(model.q) == H
    assert len(model.z) == H
    assert len(model.I) == (H + 1) * Ns
    assert len(model.P) == (H + 1) * Ns * Lmax
    assert len(model.R) == H * Ns
    assert len(model.lost) == H * Ns
    assert len(model.u) == Ns
    assert model.D[0, 0] >= 0.0
    assert sum(model.delta[0, 0, ell] for ell in model.L) == 1

    print("smoke-milp: OK")


def run_days(days: int, skus: int) -> None:
    """Run a small MPC simulation for Step 5 (independent multi-SKU)."""
    if days <= 0:
        raise ValueError("--days must be > 0")
    if skus <= 0:
        raise ValueError("--skus must be > 0")

    params = {
        "H": 5,
        "Ns": 12,
        "Lmax": 3,
        "K_fix": 8.0,
        "v": 1.0,
        "h": 0.15,
        "p": 6.0,
        "lambda": 0.5,
        "q_min": 0.0,
        "q_max": 40.0,
        "I_target": 12.0,
        "scenario_demand_low": 4.0,
        "scenario_demand_high": 13.0,
        "scenario_lead_time_weights": [0.5, 0.35, 0.15],
    }

    controller = Controller.from_shared_params(
        skus=skus,
        params=params,
        solver_name="highs",
        base_scenario_seed=101,
        initial_on_hand=12.0,
        initial_pipeline=[2.0, 0.0, 0.0],
    )
    env_rng = Random(2026 + skus)
    lead_time_weights = params["scenario_lead_time_weights"]
    lead_time_values = [1, 2, 3]
    kpis = KPITracker(K_fix=params["K_fix"], v=params["v"], h=params["h"], p=params["p"])

    print(f"running: days={days}, skus={skus}")
    for t in range(days):
        demands = [
            env_rng.uniform(params["scenario_demand_low"], params["scenario_demand_high"])
            for _ in range(skus)
        ]
        lead_times = [
            env_rng.choices(lead_time_values, weights=lead_time_weights, k=1)[0]
            for _ in range(skus)
        ]
        steps = controller.step(demands=demands, lead_times=lead_times)
        for sku_id, step in enumerate(steps):
            kpis.record_step(
                q=step["q"],
                demand=demands[sku_id],
                sales=step["sales"],
                lost_sales=step["lost_sales"],
                inventory=step["I"],
            )
            print(
                "day={t} sku={sku} q={q:.2f} demand={d:.2f} L={L} sales={s:.2f} "
                "lost={ls:.2f} I={I:.2f}".format(
                    t=t,
                    sku=sku_id,
                    q=step["q"],
                    d=demands[sku_id],
                    L=lead_times[sku_id],
                    s=step["sales"],
                    ls=step["lost_sales"],
                    I=step["I"],
                )
            )

    summary = kpis.summary()
    print(
        "kpi total_cost={total_cost:.2f} fill_rate={fill_rate:.4f} "
        "avg_inventory={avg_inventory:.2f} stockout_count={stockout_count}".format(
            total_cost=summary["total_cost"],
            fill_rate=summary["fill_rate"],
            avg_inventory=summary["avg_inventory"],
            stockout_count=int(summary["stockout_count"]),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory MPC toy runner")
    parser.add_argument("--smoke-state", action="store_true", help="run SKUState smoke test")
    parser.add_argument(
        "--smoke-scenarios", action="store_true", help="run scenario generator smoke test"
    )
    parser.add_argument("--smoke-milp", action="store_true", help="run MILP builder smoke test")
    parser.add_argument("--days", type=int, default=None, help="simulation horizon in days")
    parser.add_argument("--skus", type=int, default=1, help="number of SKUs to simulate")
    args = parser.parse_args()

    if args.smoke_state:
        run_smoke_state()
    if args.smoke_scenarios:
        run_smoke_scenarios()
    if args.smoke_milp:
        run_smoke_milp()
    if args.days is not None:
        run_days(days=args.days, skus=args.skus)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
