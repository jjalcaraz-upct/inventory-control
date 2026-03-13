#!/usr/bin/env python
"""Train DQN + SAC with non-blocking worker processes and status polling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control_rl.agent_specs import build_agent_id
from control_rl.agents import evaluate_agent, get_learning_curves, register_agent, train_agent
from model.io import read_json, write_json


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
POLL_INTERVAL_SECONDS = 1.0
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

DQN_ALGO_CONFIG = {
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

SAC_ALGO_CONFIG = {
    "action_mode": "continuous",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN + SAC with status polling")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--algorithm-name", choices=["DQN", "SAC"], help=argparse.SUPPRESS)
    parser.add_argument("--result-path", help=argparse.SUPPRESS)
    return parser.parse_args()


def _algorithm_config(algorithm_name: str) -> dict[str, float | str | int]:
    if algorithm_name == "DQN":
        return DQN_ALGO_CONFIG
    if algorithm_name == "SAC":
        return SAC_ALGO_CONFIG
    raise ValueError(f"Unsupported algorithm_name='{algorithm_name}'")


def _status_path(algorithm_name: str, algorithm_config: dict[str, float | str | int]) -> Path:
    agent_id = build_agent_id(algorithm_name, algorithm_config)
    return SKU_MODELS_DIR / agent_id / "status.json"


def aggregate_curve_runs(curve_list: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _workflow_payload(algorithm_name: str) -> dict[str, object]:
    algorithm_config = _algorithm_config(algorithm_name)
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
        "agent_metadata": agent_metadata,
        "evaluation_summary": {
            "n_replicas": int(len(evaluation_result["kpi_summary"])),
            "aggregate_summary": dict(evaluation_result["aggregate_summary"]),
        },
    }


def _run_worker(algorithm_name: str, result_path: Path) -> int:
    payload = _workflow_payload(algorithm_name)
    write_json(result_path, payload)
    return 0


def _launch_worker(algorithm_name: str, result_path: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--algorithm-name",
            algorithm_name,
            "--result-path",
            str(result_path),
        ],
        cwd=str(ROOT),
        stdin=subprocess.DEVNULL,
    )


def _wait_for_worker(
    *,
    algorithm_name: str,
    algorithm_config: dict[str, float | str | int],
    process: subprocess.Popen,
) -> None:
    status_path = _status_path(algorithm_name, algorithm_config)
    last_snapshot: tuple[object, ...] | None = None

    while True:
        status = None
        if status_path.exists():
            status = read_json(status_path)
            snapshot = (
                status.get("state"),
                status.get("completed_replicas"),
                status.get("n_replicas"),
                status.get("best_replica_score"),
            )
            if snapshot != last_snapshot:
                completed = int(status.get("completed_replicas", 0))
                total = int(status.get("n_replicas", 0))
                best_score = status.get("best_replica_score")
                print(
                    f"{algorithm_name}: replica {completed} of {total} "
                    f"| best score: {best_score}"
                )
                last_snapshot = snapshot

        return_code = process.poll()
        if return_code is not None:
            if return_code != 0:
                error = None if status is None else status.get("error")
                raise RuntimeError(f"{algorithm_name} worker failed with code {return_code}: {error}")
            return

        time.sleep(POLL_INTERVAL_SECONDS)


def _run_nonblocking_workflow(algorithm_name: str) -> dict[str, object]:
    algorithm_config = _algorithm_config(algorithm_name)
    temp_dir = Path(tempfile.mkdtemp(prefix="inventory_rl_"))
    result_path = temp_dir / f"{algorithm_name.lower()}_result.json"
    process = _launch_worker(algorithm_name, result_path)
    try:
        _wait_for_worker(
            algorithm_name=algorithm_name,
            algorithm_config=algorithm_config,
            process=process,
        )
        return read_json(result_path)
    finally:
        if result_path.exists():
            result_path.unlink()
        temp_dir.rmdir()


def _plot_learning_curves(dqn_result: dict[str, object], sac_result: dict[str, object]) -> None:
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
    print(f"Saved plot: {out_plot.resolve()}")


def _run_manager() -> int:
    dqn_result = _run_nonblocking_workflow("DQN")
    sac_result = _run_nonblocking_workflow("SAC")
    _plot_learning_curves(dqn_result, sac_result)

    print(
        f"DQN agent: {dqn_result['agent_metadata']['agent_id']} "
        f"| evaluation replicas: {dqn_result['evaluation_summary']['n_replicas']}"
    )
    print(
        f"SAC agent: {sac_result['agent_metadata']['agent_id']} "
        f"| evaluation replicas: {sac_result['evaluation_summary']['n_replicas']}"
    )
    return 0


def main() -> int:
    args = _parse_args()
    if args.worker:
        if not args.algorithm_name or not args.result_path:
            raise SystemExit("worker mode requires --algorithm-name and --result-path")
        return _run_worker(args.algorithm_name, Path(args.result_path))
    return _run_manager()


if __name__ == "__main__":
    raise SystemExit(main())
