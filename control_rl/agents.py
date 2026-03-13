"""High-level services to persist and load RL agents by SKU and configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from control_rl.agent_specs import (
    build_agent_id,
    build_agent_metadata,
    normalize_algorithm_name,
    resolve_action_quantities,
)
from control_rl.factory import load_model
from control_rl.registry import SKURegistry
from control_rl.training import run_training
from control_rl.wrappers import RLPolicyAdapter
from model.io import now_utc_iso
from model.evaluation import build_scenario, evaluate_policy


def _as_registry(registry: SKURegistry | str | Path) -> SKURegistry:
    return registry if isinstance(registry, SKURegistry) else SKURegistry(registry)


def _curve_from_path(curve_path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(curve_path)
    return (
        df["num_timesteps"].to_numpy(dtype=float),
        df["eval_reward_mean"].to_numpy(dtype=float),
    )


def get_learning_curves(result: Mapping[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return learning curves as `(steps, mean_reward)` arrays.

    Accepts either a raw `training_result` or a mapping that contains it
    under `training_result`.
    """
    training_result = result
    nested = result.get("training_result")
    if isinstance(nested, Mapping):
        training_result = nested

    curve_paths = training_result.get("learning_curve_paths")
    if not isinstance(curve_paths, list):
        raise ValueError("result must include learning_curve_paths or training_result.learning_curve_paths")

    return [_curve_from_path(str(path)) for path in curve_paths]


def get_sku_config(registry: SKURegistry | str | Path, sku_id: str) -> dict[str, Any]:
    """Load the persisted configuration for one SKU."""
    reg = _as_registry(registry)
    return reg.get_sku_config(sku_id)


def list_registered_agents(registry: SKURegistry | str | Path, sku_id: str) -> list[dict[str, Any]]:
    """List persisted agents registered for one SKU."""
    reg = _as_registry(registry)
    return reg.list_agents(sku_id)


def train_agent(
    *,
    env_config: Mapping[str, Any],
    algorithm_name: str,
    algorithm_config: Mapping[str, Any] | None,
    training_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one RL agent and return the raw training artifacts.

    The service sets `run_id` to the canonical `agent_id` when it is not
    provided explicitly, so artifact paths stay aligned with registry ids.
    """
    resolved_training_config = dict(training_config or {})
    resolved_training_config.setdefault("run_id", build_agent_id(algorithm_name, algorithm_config))
    return run_training(
        env_config=env_config,
        algorithm_name=algorithm_name,
        algorithm_config=dict(algorithm_config or {}),
        training_config=resolved_training_config,
    )


def evaluate_agent(
    *,
    training_result: Mapping[str, Any],
    env_config: Mapping[str, Any],
    algorithm_name: str,
    algorithm_config: Mapping[str, Any] | None,
    n_days: int = 30,
    n_replicas: int = 20,
    base_seed: int = 0,
) -> dict[str, Any]:
    """Evaluate a trained agent before registration.

    Loads the best model from `training_result`, wraps it as a policy, builds
    a shared scenario, and returns the KPI outputs from `evaluate_policy(...)`.
    """
    run = dict(training_result["run"])
    model_path = str(run["best_model_path"])
    model = load_model(algorithm_name, model_path)
    action_quantities = resolve_action_quantities(
        env_config=env_config,
        algorithm_name=algorithm_name,
        algorithm_config=algorithm_config,
    )
    policy = RLPolicyAdapter(model, action_quantities=action_quantities)
    scenario = build_scenario(
        R=int(n_replicas),
        N_days=int(n_days),
        env_config=env_config,
        base_seed=int(base_seed),
    )
    model_name = f"{normalize_algorithm_name(algorithm_name)}:{run['run_id']}"
    return evaluate_policy(
        model_name=model_name,
        policy=policy,
        scenario=scenario,
        env_config=env_config,
    )


def register_agent(
    *,
    registry: SKURegistry | str | Path,
    sku_id: str,
    env_config: Mapping[str, Any],
    algorithm_name: str,
    algorithm_config: Mapping[str, Any] | None,
    training_result: Mapping[str, Any],
    training_config: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Persist one trained agent in the SKU registry.

    The returned `agent_metadata` is the canonical payload later used to list,
    load, compare, or delete the agent.
    """
    reg = _as_registry(registry)
    agent_id = build_agent_id(algorithm_name, algorithm_config)
    paths = reg.agent_paths(sku_id, agent_id)
    agent_metadata = build_agent_metadata(
        sku_id=sku_id,
        env_config=env_config,
        algorithm_name=algorithm_name,
        algorithm_config=algorithm_config,
        training_config=dict(training_config or {}),
        model_path=str(Path(training_result["run"]["best_model_path"]).expanduser().resolve()),
        status_path=str(Path(training_result["run"]["status_path"]).expanduser().resolve()),
        learning_curves_dir=str(Path(training_result["run"]["learning_curves_dir"]).expanduser().resolve()),
        created_at=now_utc_iso(),
        evaluation_dir=paths["evaluation_dir"],
        evaluation_index=paths["evaluation_index"],
    )
    return reg.save_agent(agent_metadata, overwrite=overwrite)


def delete_agent(
    *,
    registry: SKURegistry | str | Path,
    sku_id: str,
    agent_id: str,
    delete_artifacts: bool = False,
) -> dict[str, Any]:
    """Delete one registered agent and optionally its training artifacts."""
    reg = _as_registry(registry)
    return reg.delete_agent(sku_id, agent_id, delete_artifacts=delete_artifacts)


def delete_sku(
    *,
    registry: SKURegistry | str | Path,
    sku_id: str,
    delete_artifacts: bool = False,
) -> dict[str, Any]:
    """Delete one SKU, its registered agents, and optionally their artifacts."""
    reg = _as_registry(registry)
    return reg.delete_sku(sku_id, delete_artifacts=delete_artifacts)


def load_policy(
    registry: str | Path,
    sku_id: str,
    agent_id: str,
    *,
    include_metadata: bool = False,
) -> Any:
    """Load one registered agent as an `RLPolicyAdapter`.

    When `include_metadata=True`, returns a mapping with the wrapped policy,
    the display `model_name`, and the persisted `agent_metadata`.
    """
    reg = SKURegistry(registry)
    agent = reg.get_agent(sku_id, agent_id)
    model = load_model(agent["algorithm_name"], agent["model_path"])
    policy = RLPolicyAdapter(model, action_quantities=agent.get("action_quantities"))

    if not include_metadata:
        return policy

    model_name = f"{str(agent['algorithm_name']).upper()}:{agent['agent_id']}"
    return {
        "policy": policy,
        "model_name": model_name,
        "agent_metadata": agent,
    }
