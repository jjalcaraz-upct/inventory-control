#!/usr/bin/env python
"""Minimal workflow: register one SKU, train DQN + SAC, plot learning curves."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control_rl.agents import evaluate_agent, get_learning_curves, register_agent, train_agent


SKU_ID = "sku_001"
REGISTRY_ROOT = "artifacts/sku_registry"
TRAINED_MODELS_ROOT = Path("artifacts/trained_models")
SKU_MODELS_DIR = TRAINED_MODELS_ROOT / SKU_ID
PLOTS_DIR = SKU_MODELS_DIR / "comparison"
TOTAL_TIMESTEPS = 5_000
SEED = 123
RUNS = 5
EVAL_FREQ = 250
EVAL_EPISODES = 5
LEARNING_CURVE_SMOOTH_WINDOW = 3
CONF_Z = 1.697
BASE_TRAINING_CONFIG = {
    "n_replicas": RUNS,
    "base_seed": SEED,
    "total_timesteps": TOTAL_TIMESTEPS,
    "eval_freq": EVAL_FREQ,
    "n_eval_episodes": EVAL_EPISODES,
    "output_dir": str(SKU_MODELS_DIR),
}

ENV_CONFIG = {
    "Lmax": 3,
    "demand_spec": {
        "kind": "normal_clipped",
        "mean": 8.5,
        "std": 2.0,
        "low": 0.0,
        "high": 20.0,
    },
    "lead_time_weights": [0.5, 0.35, 0.15],
    "q_min": 0.0,
    "q_max": 40.0,
    "K_fix": 8.0,
    "v": 1.0,
    "h": 0.15,
    "p": 6.0,
    "max_steps": 30,
}

dqn_algo_config = {
    "action_mode": "discrete",
    "action_step": 1.0,
    "learning_rate": 5e-4,
    "buffer_size": 8_000,
    "batch_size": 128,
    "gamma": 0.97,
    "learning_starts": 200,
    "train_freq": 4,
    "target_update_interval": 250,
}

# sac_algo_config_default = {
#     "action_mode": "continuous",
# }

# Alternative SAC config
sac_algo_config = {
    "action_mode": "continuous",
    'learning_rate': 3e-4,
    'buffer_size': 5_000,
    'batch_size': 128,
    'learning_starts': 500,
    'tau': 0.005,
    'gamma': 0.99,
}


def aggregate_curve_runs(curve_list: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate per-run learning curves on a common step axis."""
    common_steps = np.asarray(curve_list[0][0], dtype=float)
    stacked = []
    for steps, rewards in curve_list:
        steps = np.asarray(steps, dtype=float)
        rewards = np.asarray(rewards, dtype=float)
        if not np.array_equal(steps, common_steps):
            rewards = np.interp(common_steps, steps, rewards)
        stacked.append(rewards)

    mat = np.vstack(stacked)
    r_mean = np.nanmean(mat, axis=0)
    r_std = np.nanstd(mat, axis=0, ddof=1 if mat.shape[0] > 1 else 0)
    return common_steps, r_mean, r_std


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    """Apply moving-average smoothing without edge artifacts."""
    values = np.asarray(values, dtype=float)
    w = int(window)
    if w <= 1 or values.size < w:
        return values
    if w % 2 == 0:
        w += 1

    half = w // 2
    kernel = np.ones(w, dtype=float) / float(w)
    padded = np.pad(values, (half, half), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def run_agent_workflow(*, algorithm_name: str, algorithm_config: dict[str, float | str | int]) -> dict[str, object]:
    training_result = train_agent(
        env_config=ENV_CONFIG,
        algorithm_name=algorithm_name,
        algorithm_config=algorithm_config,
        training_config=BASE_TRAINING_CONFIG,
    )
    evaluation_result = evaluate_agent(
        training_result=training_result,
        env_config=ENV_CONFIG,
        algorithm_name=algorithm_name,
        algorithm_config=algorithm_config,
        base_seed=SEED,
    )
    agent_metadata = register_agent(
        registry=REGISTRY_ROOT,
        sku_id=SKU_ID,
        env_config=ENV_CONFIG,
        algorithm_name=algorithm_name,
        algorithm_config=algorithm_config,
        training_result=training_result,
        training_config=BASE_TRAINING_CONFIG,
        overwrite=True,
    )
    return {
        "training_result": training_result,
        "evaluation_result": evaluation_result,
        "agent_metadata": agent_metadata,
    }


def main() -> int:
    dqn_result = run_agent_workflow(
        algorithm_name="DQN",
        algorithm_config=dqn_algo_config,
    )
    sac_result = run_agent_workflow(
        algorithm_name="SAC",
        algorithm_config=sac_algo_config,
    )

    dqn_curve_list = get_learning_curves(dqn_result["training_result"])
    sac_curve_list = get_learning_curves(sac_result["training_result"])

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for name, curve_list in [("DQN", dqn_curve_list), ("SAC", sac_curve_list)]:
        steps, r_mean, r_std = aggregate_curve_runs(curve_list)
        r_mean_plot = smooth_series(r_mean, LEARNING_CURVE_SMOOTH_WINDOW)
        r_std_plot = smooth_series(r_std, LEARNING_CURVE_SMOOTH_WINDOW)
        margin = CONF_Z * r_std_plot / np.sqrt(RUNS)
        ax.plot(steps, r_mean_plot, label=name)
        ax.fill_between(steps, r_mean_plot - margin, r_mean_plot + margin, alpha=0.2)

    ax.set_title("Learning Curves: mean eval reward vs training steps")
    ax.set_xlabel("training steps")
    ax.set_ylabel("mean evaluation reward")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    out_plot = PLOTS_DIR / "learning_curves_dqn_vs_sac_replicas.png"
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=140)
    plt.show()
    print(
        f"DQN agent: {dqn_result['agent_metadata']['agent_id']} "
        f"| evaluation replicas: {len(dqn_result['evaluation_result']['kpi_summary'])}"
    )
    print(
        f"SAC agent: {sac_result['agent_metadata']['agent_id']} "
        f"| evaluation replicas: {len(sac_result['evaluation_result']['kpi_summary'])}"
    )
    print(f"Saved plot: {out_plot.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
