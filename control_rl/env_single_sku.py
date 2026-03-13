"""Gymnasium environment for single-SKU inventory control."""

from __future__ import annotations

from random import Random
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from control_rl.action_utils import to_scalar_action
from model.demand import DemandDistribution
from model.metrics import KPITracker
from model.state import SKUState


class InventorySingleSKUEnv(gym.Env):
    """Infinite-horizon single-SKU inventory environment (continuing task)."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        *,
        Lmax: int = 3,
        demand_spec: Optional[Mapping[str, Any]] = None,
        lead_time_weights: Optional[Sequence[float]] = None,
        q_min: float = 0.0,
        q_max: float = 40.0,
        K_fix: float = 8.0,
        v: float = 1.0,
        h: float = 0.15,
        p: float = 6.0,
        initial_on_hand: float = 12.0,
        initial_pipeline: Optional[Sequence[float]] = None,
        max_steps: Optional[int] = None,
        normalize_obs: bool = False,
        inventory_scale: float = 100.0,
        pipeline_scale: float = 100.0,
    ) -> None:
        super().__init__()

        if Lmax <= 0:
            raise ValueError("Lmax must be > 0")
        if q_min < 0 or q_max < 0 or q_max < q_min:
            raise ValueError("Require 0 <= q_min <= q_max")
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be > 0 when provided")

        self.Lmax = int(Lmax)
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.K_fix = float(K_fix)
        self.v = float(v)
        self.h = float(h)
        self.p = float(p)
        self.max_steps = max_steps

        self.normalize_obs = bool(normalize_obs)
        self.inventory_scale = float(inventory_scale)
        self.pipeline_scale = float(pipeline_scale)
        if self.inventory_scale <= 0 or self.pipeline_scale <= 0:
            raise ValueError("Observation scales must be > 0")

        if initial_pipeline is None:
            initial_pipeline = [0.0] * self.Lmax
        if len(initial_pipeline) != self.Lmax:
            raise ValueError("initial_pipeline length must equal Lmax")
        self._initial_on_hand = float(initial_on_hand)
        self._initial_pipeline = [float(x) for x in initial_pipeline]

        if lead_time_weights is None:
            lead_time_weights = [1.0] * self.Lmax
        if len(lead_time_weights) != self.Lmax:
            raise ValueError("lead_time_weights length must equal Lmax")
        self.lead_time_weights = [float(w) for w in lead_time_weights]
        if any(w < 0 for w in self.lead_time_weights):
            raise ValueError("lead_time_weights must be non-negative")
        if sum(self.lead_time_weights) <= 0:
            raise ValueError("lead_time_weights must contain a positive value")

        if demand_spec is None:
            demand_spec = {"kind": "uniform", "low": 0.0, "high": 10.0}
        self.demand = DemandDistribution(demand_spec)

        self.action_space = spaces.Box(
            low=np.array([self.q_min], dtype=np.float32),
            high=np.array([self.q_max], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(1 + self.Lmax,),
            dtype=np.float32,
        )

        self.state = SKUState(on_hand=self._initial_on_hand, pipeline=list(self._initial_pipeline))
        self._kpis = KPITracker(K_fix=self.K_fix, v=self.v, h=self.h, p=self.p)
        self._rng = Random(0)
        self._t = 0

    def _get_obs(self) -> np.ndarray:
        obs = np.array([self.state.on_hand] + list(self.state.pipeline), dtype=np.float32)
        if self.normalize_obs:
            obs[0] = obs[0] / self.inventory_scale
            obs[1:] = obs[1:] / self.pipeline_scale
        return obs

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        on_hand = float(options.get("initial_on_hand", self._initial_on_hand))
        pipeline = options.get("initial_pipeline", self._initial_pipeline)
        pipeline = [float(x) for x in pipeline]
        if len(pipeline) != self.Lmax:
            raise ValueError("initial_pipeline length must equal Lmax")

        if seed is None:
            rng_seed = int(self.np_random.integers(0, 2**31 - 1))
        else:
            rng_seed = int(seed)
        self._rng = Random(rng_seed)

        self.state = SKUState(on_hand=on_hand, pipeline=pipeline)
        self._kpis = KPITracker(K_fix=self.K_fix, v=self.v, h=self.h, p=self.p)
        self._t = 0

        obs = self._get_obs()
        info = {"t": self._t}
        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        q_raw = to_scalar_action(action)
        if not np.isfinite(q_raw):
            raise ValueError(f"Continuous action must be finite, got {action!r}")
        q = float(np.clip(q_raw, self.q_min, self.q_max))
        demand = float(self.demand.sample(self._t, 0, self._rng))
        lead_time = None
        state_lead_time = 1
        if q > 0.0:
            lead_time = int(
                self._rng.choices(range(1, self.Lmax + 1), weights=self.lead_time_weights, k=1)[0]
            )
            state_lead_time = lead_time

        inventory_start = float(self.state.on_hand)
        transition = self.state.step(demand=demand, q=q, L=state_lead_time)
        inventory_end = float(transition["I_next"])

        step_costs = self._kpis.record_step(
            q=q,
            demand=demand,
            sales=float(transition["sales"]),
            lost_sales=float(transition["lost_sales"]),
            inventory=inventory_end,
        )
        ordering_cost = float(step_costs["ordering_cost"])
        holding_cost = float(step_costs["holding_cost"])
        lost_sales_cost = float(step_costs["lost_sales_cost"])
        step_cost = float(step_costs["step_cost"])
        reward = -float(step_cost)

        self._t += 1
        terminated = False
        truncated = self.max_steps is not None and self._t >= self.max_steps

        obs = self._get_obs()
        info = {
            "t": self._t,
            "inventory_start": inventory_start,
            "inventory_end": inventory_end,
            "demand": demand,
            "received": float(transition["received"]),
            "sales": float(transition["sales"]),
            "lost_sales": float(transition["lost_sales"]),
            "order_qty": q,
            "lead_time_sampled": lead_time,
            "ordering_cost": float(ordering_cost),
            "holding_cost": float(holding_cost),
            "lost_sales_cost": float(lost_sales_cost),
            "step_cost": float(step_cost),
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        msg = (
            f"t={self._t} I={self.state.on_hand:.2f} "
            f"P={list(round(x, 2) for x in self.state.pipeline)}"
        )
        print(msg)
        return msg

    def close(self) -> None:
        return None
