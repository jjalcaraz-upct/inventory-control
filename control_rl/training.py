"""Minimal training entrypoint for inventory RL."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from control_rl.factory import create_agent, make_env
from model.io import now_utc_iso, write_json


def _extract_eval_curve_rows(callback: EvalCallback) -> list[dict[str, Any]]:
    steps = np.asarray(callback.evaluations_timesteps, dtype=int)
    rewards = np.asarray(callback.evaluations_results, dtype=float)

    if steps.size == 0:
        return [{"num_timesteps": 0, "eval_reward_mean": float("nan"), "eval_reward_std": float("nan")}]

    if rewards.ndim == 1:
        means = rewards.astype(float)
        stds = np.zeros_like(means, dtype=float)
    else:
        means = rewards.mean(axis=1)
        stds = rewards.std(axis=1, ddof=0)

    return [
        {
            "num_timesteps": int(step),
            "eval_reward_mean": float(mean),
            "eval_reward_std": float(std),
        }
        for step, mean, std in zip(steps, means, stds)
    ]


def _write_learning_curve(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["num_timesteps", "eval_reward_mean", "eval_reward_std"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "num_timesteps": int(row["num_timesteps"]),
                    "eval_reward_mean": float(row["eval_reward_mean"]),
                    "eval_reward_std": float(row["eval_reward_std"]),
                }
            )


def _resolve_run_id(algorithm_name: str, cfg: Mapping[str, Any]) -> str:
    explicit_run_id = str(cfg.get("run_id", "")).strip()
    if explicit_run_id:
        return explicit_run_id
    return f"{algorithm_name.upper()}_{now_utc_iso().replace(':', '').replace('-', '')}"


def run_training(
    *,
    env_config: Mapping[str, Any],
    algorithm_name: str,
    algorithm_config: Mapping[str, Any],
    training_config: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    cfg = dict(training_config or {})
    output_dir = Path(str(cfg.get("output_dir", "artifacts/trained_models")))
    total_timesteps = int(cfg.get("total_timesteps", 10_000))
    eval_freq = int(cfg.get("eval_freq", max(1, total_timesteps // 20)))
    n_eval_episodes = int(cfg.get("n_eval_episodes", 5))
    n_replicas = int(cfg.get("n_replicas", 1))
    base_seed = int(cfg.get("base_seed", cfg.get("seed", 0)))
    if n_replicas <= 0:
        raise ValueError("n_replicas must be >= 1")

    run_id = _resolve_run_id(algorithm_name, cfg)
    run_dir = output_dir / run_id
    curves_dir = run_dir / "learning_curves"
    tmp_dir = run_dir / "_tmp_replicas"
    status_path = run_dir / "status.json"
    model_path = run_dir / "model_best.zip"

    run_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        status_path,
        {
            "state": "running",
            "updated_at": now_utc_iso(),
            "n_replicas": n_replicas,
            "completed_replicas": 0,
        },
    )

    algo = str(algorithm_name).strip().upper()
    base_agent_cfg = dict(algorithm_config)
    action_mode = base_agent_cfg.get("action_mode")
    if action_mode is None and algo == "DQN":
        action_mode = "discrete"

    best_replica_index: Optional[int] = None
    best_replica_score = float("-inf")
    learning_curve_paths: list[str] = []

    try:
        for replica in range(n_replicas):
            replica_seed = base_seed + replica
            replica_dir = tmp_dir / f"replica_{replica:03d}"
            replica_dir.mkdir(parents=True, exist_ok=True)

            env = make_env(
                env_kwargs=dict(env_config),
                action_mode=action_mode,
                action_quantities=base_agent_cfg.get("action_quantities"),
                action_step=float(base_agent_cfg.get("action_step", 1.0)),
            )
            eval_env = Monitor(
                make_env(
                    env_kwargs=dict(env_config),
                    action_mode=action_mode,
                    action_quantities=base_agent_cfg.get("action_quantities"),
                    action_step=float(base_agent_cfg.get("action_step", 1.0)),
                )
            )
            try:
                agent_cfg = dict(base_agent_cfg)
                agent_cfg["seed"] = replica_seed
                model = create_agent(algorithm_name=algorithm_name, algorithm_config=agent_cfg, env=env)
                eval_callback = EvalCallback(
                    eval_env,
                    best_model_save_path=str(replica_dir),
                    log_path=str(replica_dir),
                    eval_freq=eval_freq,
                    n_eval_episodes=n_eval_episodes,
                    deterministic=True,
                    render=False,
                    verbose=0,
                )
                model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=False)

                rows = _extract_eval_curve_rows(eval_callback)
                curve_path = curves_dir / f"replica_{replica:03d}.csv"
                _write_learning_curve(curve_path, rows)
                learning_curve_paths.append(str(curve_path))

                means = np.array([float(r["eval_reward_mean"]) for r in rows], dtype=float)
                replica_score = float(np.nanmax(means)) if np.isfinite(means).any() else float("-inf")

                candidate = replica_dir / "best_model.zip"
                if not candidate.exists():
                    model.save(str(replica_dir / "model_final"))
                    candidate = replica_dir / "model_final.zip"

                if replica_score > best_replica_score:
                    best_replica_score = replica_score
                    best_replica_index = replica
                    shutil.copy2(candidate, model_path)
            finally:
                env.close()
                eval_env.close()
                shutil.rmtree(replica_dir, ignore_errors=True)

            write_json(
                status_path,
                {
                    "state": "running",
                    "updated_at": now_utc_iso(),
                    "n_replicas": n_replicas,
                    "completed_replicas": replica + 1,
                    "best_replica_index": best_replica_index,
                    "best_replica_score": best_replica_score,
                },
            )
    except Exception as exc:
        write_json(
            status_path,
                {
                    "state": "failed",
                    "updated_at": now_utc_iso(),
                    "error": f"{exc.__class__.__name__}: {exc}",
                },
            )
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    write_json(
        status_path,
        {
            "state": "completed",
            "updated_at": now_utc_iso(),
            "n_replicas": n_replicas,
            "completed_replicas": n_replicas,
            "best_replica_index": best_replica_index,
            "best_replica_score": best_replica_score,
        },
    )

    run = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status_path": str(status_path),
        "best_model_path": str(model_path),
        "learning_curves_dir": str(curves_dir),
    }
    return {
        "run": run,
        "best_replica_index": best_replica_index,
        "best_replica_score": best_replica_score,
        "learning_curve_paths": learning_curve_paths,
        "n_replicas": n_replicas,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal RL training from JSON config")
    parser.add_argument("--config", required=True, help="JSON config path")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    payload = json.loads(Path(args.config).read_text())

    env_config = dict(payload["env_config"])
    algorithm_name = str(payload["algorithm_name"])
    algorithm_config = dict(payload.get("algorithm_config", {}))
    training_config = dict(payload.get("training_config", {}))

    result = run_training(
        env_config=env_config,
        algorithm_name=algorithm_name,
        algorithm_config=algorithm_config,
        training_config=training_config,
    )

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
