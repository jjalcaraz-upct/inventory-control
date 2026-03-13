#!/usr/bin/env python
"""Compare the first two registered RL agents for one SKU on a common scenario."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control_rl.agents import get_sku_config, list_registered_agents, load_policy
from model.evaluation import build_scenario, evaluate_policy

SKU_ID = "sku_001"
REGISTRY_ROOT = "artifacts/sku_registry"
OUTPUT_DIR = Path("artifacts/trained_models") / SKU_ID / "comparison"

R = 20
N_DAYS = 30
SEED = 123
KPI_LIST = ["total_cost", "total_lost_sales", "avg_inventory"]

def _plot_violin_kpis(compare_df: pd.DataFrame, labels: list[str], out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(KPI_LIST), figsize=(14, 4), constrained_layout=True)
    for ax, kpi in zip(axes, KPI_LIST):
        data = [compare_df.loc[compare_df["agent_label"] == label, kpi].to_numpy(dtype=float) for label in labels]
        ax.violinplot(data, showmeans=True, showextrema=True)
        ax.set_xticks(range(1, len(labels) + 1), labels)
        ax.set_title(kpi)
        ax.grid(axis="y", alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.show()


def main() -> int:
    env_config = get_sku_config(REGISTRY_ROOT, SKU_ID)

    registered_agents = list_registered_agents(REGISTRY_ROOT, SKU_ID)
    if len(registered_agents) < 2:
        raise RuntimeError(f"SKU '{SKU_ID}' must have at least two registered agents to compare")

    selected_agents = registered_agents[:2]

    agents: list[dict] = []
    for agent in selected_agents:
        agent = load_policy(
            registry=REGISTRY_ROOT,
            sku_id=SKU_ID,
            agent_id=str(agent["agent_id"]),
            include_metadata=True,
        )
        agents.append(agent)

    scenario = build_scenario(
        R=R,
        N_days=N_DAYS,
        env_config=env_config,
        base_seed=SEED,
    )

    labels: list[str] = []
    result_frames: list[pd.DataFrame] = []
    for agent in agents:
        model_name = str(agent["model_name"])
        policy = agent["policy"]

        result = evaluate_policy(
            model_name=model_name,
            policy=policy,
            scenario=scenario,
            env_config=env_config,
        )

        labels.append(model_name)
        result_frames.append(result["kpi_summary"].assign(agent_label=model_name))

    compare_df = pd.concat(result_frames, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "comparison_kpis_first_two_agents.csv"
    plot_path = OUTPUT_DIR / "violin_kpis_first_two_agents.png"
    compare_df.to_csv(csv_path, index=False)
    _plot_violin_kpis(compare_df, labels, plot_path)

    print(f"Compared agents: {labels[0]} vs {labels[1]}")
    print(f"Saved KPI table: {csv_path.resolve()}")
    print(f"Saved violin plot: {plot_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
