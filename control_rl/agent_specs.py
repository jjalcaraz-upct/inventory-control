"""Pure helpers to normalize, identify and describe persisted RL agents."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np

from control_rl.action_utils import build_action_quantities


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def normalize_algorithm_name(algorithm_name: str) -> str:
    name = str(algorithm_name).strip().upper()
    if not name:
        raise ValueError("algorithm_name must be non-empty")
    return name


def normalize_algorithm_config(algorithm_config: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = dict(algorithm_config or {})
    return {str(key): _to_jsonable(value) for key, value in cfg.items()}


def resolve_action_mode(algorithm_name: str, algorithm_config: Mapping[str, Any] | None) -> str:
    algo = normalize_algorithm_name(algorithm_name)
    cfg = normalize_algorithm_config(algorithm_config)
    action_mode = cfg.get("action_mode")
    if action_mode is None:
        return "discrete" if algo == "DQN" else "continuous"

    resolved = str(action_mode).strip().lower()
    if resolved not in ("continuous", "discrete"):
        raise ValueError("action_mode must be 'continuous' or 'discrete'")
    return resolved


def resolve_action_quantities(
    *,
    env_config: Mapping[str, Any],
    algorithm_name: str,
    algorithm_config: Mapping[str, Any] | None,
) -> list[float] | None:
    cfg = normalize_algorithm_config(algorithm_config)
    action_mode = resolve_action_mode(algorithm_name, cfg)
    if action_mode == "continuous":
        return None

    explicit = cfg.get("action_quantities")
    if explicit is not None:
        values = [float(x) for x in _to_jsonable(explicit)]
        quantities = sorted(set(values))
        if not quantities:
            raise ValueError("action_quantities must contain at least one value")
        return quantities

    if "q_min" not in env_config or "q_max" not in env_config:
        raise KeyError("env_config must include q_min and q_max for discrete actions")
    action_step = float(cfg.get("action_step", 1.0))
    if action_step <= 0:
        raise ValueError("action_step must be > 0")

    quantities = build_action_quantities(
        q_min=float(env_config["q_min"]),
        q_max=float(env_config["q_max"]),
        action_step=action_step,
    ).tolist()
    if not quantities:
        raise ValueError("resolved action_quantities must contain at least one value")
    return [float(x) for x in quantities]


def build_algorithm_config_key(
    algorithm_name: str,
    algorithm_config: Mapping[str, Any] | None,
) -> str:
    payload = {
        "algorithm_config": normalize_algorithm_config(algorithm_config),
        "algorithm_name": normalize_algorithm_name(algorithm_name),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]


def build_agent_id(
    algorithm_name: str,
    algorithm_config: Mapping[str, Any] | None,
) -> str:
    algo = normalize_algorithm_name(algorithm_name)
    return f"{algo.lower()}_{build_algorithm_config_key(algo, algorithm_config)}"


def build_agent_metadata(
    *,
    sku_id: str,
    env_config: Mapping[str, Any],
    algorithm_name: str,
    algorithm_config: Mapping[str, Any] | None,
    training_config: Mapping[str, Any],
    model_path: str,
    status_path: str,
    learning_curves_dir: str,
    created_at: str,
    evaluation_dir: str,
    evaluation_index: str,
) -> dict[str, Any]:
    algo = normalize_algorithm_name(algorithm_name)
    cfg = normalize_algorithm_config(algorithm_config)
    action_mode = resolve_action_mode(algo, cfg)
    action_quantities = resolve_action_quantities(
        env_config=env_config,
        algorithm_name=algo,
        algorithm_config=cfg,
    )
    config_key = build_algorithm_config_key(algo, cfg)

    payload: dict[str, Any] = {
        "agent_id": build_agent_id(algo, cfg),
        "sku_id": str(sku_id),
        "algorithm_name": algo,
        "algorithm_config": cfg,
        "algorithm_config_key": config_key,
        "training_config": _to_jsonable(dict(training_config)),
        "env_config": _to_jsonable(dict(env_config)),
        "model_path": str(model_path),
        "status_path": str(status_path),
        "learning_curves_dir": str(learning_curves_dir),
        "action_mode": action_mode,
        "created_at": str(created_at),
        "evaluation_dir": str(evaluation_dir),
        "evaluation_index": str(evaluation_index),
    }
    if action_quantities is not None:
        payload["action_quantities"] = action_quantities
    return payload
