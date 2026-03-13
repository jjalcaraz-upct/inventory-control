"""Gymnasium wrappers for RL action adaptation and KPI tracking."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from control_rl.action_utils import build_action_quantities, to_scalar_action
from model.metrics import KPITracker


class RLPolicyAdapter:
    """Adapter with compute_action(state) for RL models."""

    def __init__(self, model: Any, action_quantities: Optional[Sequence[float]] = None):
        self.model = model
        self.action_quantities = list(action_quantities) if action_quantities is not None else None

    @staticmethod
    def _to_obs(state_or_obs: Any) -> np.ndarray:
        if hasattr(state_or_obs, "on_hand") and hasattr(state_or_obs, "pipeline"):
            return np.array([state_or_obs.on_hand] + list(state_or_obs.pipeline), dtype=np.float32)
        return np.asarray(state_or_obs, dtype=np.float32)

    @staticmethod
    def _to_scalar(x: Any) -> float:
        return float(np.asarray(x, dtype=float).reshape(-1)[0])

    def compute_action(self, state_or_obs: Any) -> float:
        action, _ = self.model.predict(self._to_obs(state_or_obs), deterministic=True)
        if self.action_quantities is None:
            return self._to_scalar(action)
        idx = int(np.asarray(action).reshape(-1)[0])
        return float(self.action_quantities[idx])


class ActionModeWrapper(gym.ActionWrapper):
    """Expose discrete or continuous actions over a continuous-action base env."""

    def __init__(
        self,
        env: gym.Env,
        *,
        action_mode: Literal["continuous", "discrete"] = "continuous",
        action_quantities: Optional[Sequence[float]] = None,
        action_step: float = 1.0,
    ) -> None:
        super().__init__(env)
        if action_mode not in ("continuous", "discrete"):
            raise ValueError("action_mode must be 'continuous' or 'discrete'")

        self.action_mode = action_mode
        self.action_quantities: Optional[np.ndarray] = None
        if self.action_mode == "discrete":
            if action_quantities is None:
                resolved_q_min = self._resolve_bound("q_min")
                resolved_q_max = self._resolve_bound("q_max")
                self.action_quantities = build_action_quantities(
                    q_min=resolved_q_min,
                    q_max=resolved_q_max,
                    action_step=action_step,
                )
            else:
                self.action_quantities = self._validate_action_quantities(action_quantities)
            self.action_space = spaces.Discrete(len(self.action_quantities))
        else:
            self.action_space = env.action_space

    def _resolve_bound(self, name: str) -> float:
        base_env = self.unwrapped
        if not hasattr(base_env, name):
            raise ValueError(f"Base env does not expose '{name}' required for discrete actions")
        return float(getattr(base_env, name))

    @staticmethod
    def _validate_action_quantities(action_quantities: Sequence[float]) -> np.ndarray:
        quantities = np.asarray(action_quantities, dtype=np.float32).reshape(-1)
        if quantities.size == 0:
            raise ValueError("action_quantities must contain at least one value")
        if np.any(~np.isfinite(quantities)):
            raise ValueError("action_quantities must be finite")
        return np.array(sorted(set(float(x) for x in quantities.tolist())), dtype=np.float32)

    def action(self, action: Any) -> float:
        if self.action_mode == "continuous":
            return float(to_scalar_action(action))

        if self.action_quantities is None:
            raise RuntimeError("action_quantities must be defined in discrete mode")
        action_scalar = to_scalar_action(action)
        action_idx = int(round(action_scalar))
        if not np.isclose(action_scalar, action_idx, atol=1e-8):
            raise ValueError(f"Discrete action must be an integer index, got {action!r}")
        if not self.action_space.contains(action_idx):
            raise ValueError(f"Invalid action index: {action_idx}")
        return float(self.action_quantities[action_idx])


class RandomInitialStateWrapper(gym.Wrapper):
    """Randomize initial inventory state on each reset unless explicitly provided."""

    def __init__(
        self,
        env: gym.Env,
        *,
        on_hand_range: Sequence[float],
        pipeline_range: Sequence[float],
        seed: int = 0,
    ) -> None:
        super().__init__(env)
        self.on_hand_range = self._validate_range("on_hand_range", on_hand_range)
        self.pipeline_range = self._validate_range("pipeline_range", pipeline_range)
        self._rng = np.random.default_rng(int(seed))

    @staticmethod
    def _validate_range(name: str, values: Sequence[float]) -> tuple[float, float]:
        if len(values) != 2:
            raise ValueError(f"{name} must contain exactly two values (low, high)")
        low, high = float(values[0]), float(values[1])
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError(f"{name} values must be finite")
        if high < low:
            raise ValueError(f"{name} must satisfy low <= high")
        return (low, high)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed) + 17)

        opts = dict(options or {})
        if "initial_on_hand" not in opts:
            opts["initial_on_hand"] = float(self._rng.uniform(*self.on_hand_range))
        if "initial_pipeline" not in opts:
            lmax = int(getattr(self.unwrapped, "Lmax"))
            opts["initial_pipeline"] = [
                float(self._rng.uniform(*self.pipeline_range))
                for _ in range(lmax)
            ]
        return self.env.reset(seed=seed, options=opts)


class KPITrackerWrapper(gym.Wrapper):
    """Track per-episode KPIs from step info and expose them at episode end."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._tracker: Optional[KPITracker] = None
        self._ordering_cost = 0.0
        self._holding_cost = 0.0
        self._lost_sales_cost = 0.0
        self._total_lost_sales = 0.0
        self._n_orders = 0
        self._episode_length = 0
        self._episode_return = 0.0
        self._last_episode_kpis: Optional[Dict[str, Any]] = None

    def _make_tracker(self) -> KPITracker:
        base_env = self.unwrapped
        required = ("K_fix", "v", "h", "p")
        missing = [name for name in required if not hasattr(base_env, name)]
        if missing:
            raise ValueError(f"Wrapped env is missing required cost attributes: {missing}")
        return KPITracker(
            K_fix=float(getattr(base_env, "K_fix")),
            v=float(getattr(base_env, "v")),
            h=float(getattr(base_env, "h")),
            p=float(getattr(base_env, "p")),
        )

    def _reset_episode(self) -> None:
        self._tracker = self._make_tracker()
        self._ordering_cost = 0.0
        self._holding_cost = 0.0
        self._lost_sales_cost = 0.0
        self._total_lost_sales = 0.0
        self._n_orders = 0
        self._episode_length = 0
        self._episode_return = 0.0
        self._last_episode_kpis = None

    @staticmethod
    def _require(info: Dict[str, Any], key: str) -> float:
        if key not in info:
            raise KeyError(f"step info is missing required key: {key}")
        return float(info[key])

    def reset(self, **kwargs: Any):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        self._reset_episode()
        return obs, info

    def step(self, action: Any):  # type: ignore[override]
        if self._tracker is None:
            self._reset_episode()

        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)

        q = self._require(info, "order_qty")
        demand = self._require(info, "demand")
        sales = self._require(info, "sales")
        lost_sales = self._require(info, "lost_sales")
        inventory_end = self._require(info, "inventory_end")
        ordering_cost = self._require(info, "ordering_cost")
        holding_cost = self._require(info, "holding_cost")
        lost_sales_cost = self._require(info, "lost_sales_cost")

        self._tracker.record_step(
            q=q,
            demand=demand,
            sales=sales,
            lost_sales=lost_sales,
            inventory=inventory_end,
        )

        self._ordering_cost += ordering_cost
        self._holding_cost += holding_cost
        self._lost_sales_cost += lost_sales_cost
        self._total_lost_sales += lost_sales
        self._n_orders += int(q > 0.0)
        self._episode_length += 1
        self._episode_return += float(reward)

        if terminated or truncated:
            base = self._tracker.summary()
            total_cost = self._ordering_cost + self._holding_cost + self._lost_sales_cost
            episode_kpis = {
                "total_cost": float(total_cost),
                "ordering_cost": float(self._ordering_cost),
                "holding_cost": float(self._holding_cost),
                "lost_sales_cost": float(self._lost_sales_cost),
                "fill_rate": float(base["fill_rate"]),
                "avg_inventory": float(base["avg_inventory"]),
                "total_lost_sales": float(self._total_lost_sales),
                "n_orders": int(self._n_orders),
                "stockout_count": int(base["stockout_count"]),
                "episode_length": int(self._episode_length),
                "episode_return": float(self._episode_return),
            }
            info["episode_kpis"] = episode_kpis
            self._last_episode_kpis = episode_kpis

        return obs, reward, terminated, truncated, info

    @property
    def last_episode_kpis(self) -> Optional[Dict[str, Any]]:
        """Return KPIs computed at the end of the latest episode, if available."""
        return self._last_episode_kpis
