"""Minimal factories for inventory RL envs and SB3 agents."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import gymnasium as gym

from control_rl.env_single_sku import InventorySingleSKUEnv
from control_rl.wrappers import ActionModeWrapper, KPITrackerWrapper, RandomInitialStateWrapper


def make_env(
    env_kwargs: Optional[Mapping[str, Any]] = None,
    *,
    action_mode: Optional[str] = None,
    action_quantities: Optional[Sequence[float]] = None,
    action_step: float = 1.0,
    random_initial_state: bool = False,
    initial_on_hand_range: Optional[Sequence[float]] = None,
    initial_pipeline_range: Optional[Sequence[float]] = None,
    initial_state_seed: int = 0,
    with_kpi_tracker: bool = False,
) -> gym.Env:
    """Create a single-SKU environment with optional wrappers."""
    env: gym.Env = InventorySingleSKUEnv(**dict(env_kwargs or {}))

    if random_initial_state:
        if initial_on_hand_range is None or initial_pipeline_range is None:
            raise ValueError(
                "initial_on_hand_range and initial_pipeline_range are required when "
                "random_initial_state=True"
            )
        env = RandomInitialStateWrapper(
            env,
            on_hand_range=initial_on_hand_range,
            pipeline_range=initial_pipeline_range,
            seed=initial_state_seed,
        )

    if action_mode is not None:
        if action_mode not in ("continuous", "discrete"):
            raise ValueError("action_mode must be 'continuous' or 'discrete'")
        env = ActionModeWrapper(
            env,
            action_mode=action_mode,
            action_quantities=action_quantities,
            action_step=float(action_step),
        )

    if with_kpi_tracker:
        env = KPITrackerWrapper(env)
    return env


def _algo_class(algorithm_name: str) -> Any:
    from stable_baselines3 import A2C, DQN, PPO, SAC, TD3

    name = str(algorithm_name).strip().upper()
    classes = {
        "DQN": DQN,
        "A2C": A2C,
        "PPO": PPO,
        "SAC": SAC,
        "TD3": TD3,
    }
    if name not in classes:
        raise ValueError("Unsupported algorithm. Use one of: DQN, A2C, PPO, SAC, TD3")
    return classes[name]


def create_agent(
    algorithm_name: str,
    algorithm_config: Optional[Mapping[str, Any]],
    env: gym.Env,
) -> Any:
    """Create an SB3 model from algorithm name, config and environment."""
    cfg = dict(algorithm_config or {})

    # These keys are env concerns; ignore here so they are not passed to SB3 constructors.
    cfg.pop("action_mode", None)
    cfg.pop("action_quantities", None)
    cfg.pop("action_step", None)

    policy = cfg.pop("policy", "MlpPolicy")
    seed = cfg.pop("seed", None)
    verbose = int(cfg.pop("verbose", 0))
    model_kwargs = dict(cfg.pop("model_kwargs", {}))
    model_kwargs.update(cfg)

    model_cls = _algo_class(algorithm_name)
    name = str(algorithm_name).strip().upper()
    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    is_continuous = isinstance(env.action_space, gym.spaces.Box)

    if name == "DQN" and not is_discrete:
        raise ValueError("DQN requires discrete actions. Set action_mode='discrete'.")
    if name in ("SAC", "TD3") and not is_continuous:
        raise ValueError(f"{name} requires continuous actions.")
    if name in ("A2C", "PPO") and not (is_discrete or is_continuous):
        raise ValueError(f"{name} requires a discrete or continuous action space.")

    return model_cls(
        policy=policy,
        env=env,
        seed=seed,
        verbose=verbose,
        **model_kwargs,
    )


def load_model(algorithm_name: str, model_path: str) -> Any:
    """Load persisted SB3 model by algorithm name."""
    model_cls = _algo_class(algorithm_name)
    return model_cls.load(str(model_path))
