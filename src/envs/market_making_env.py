"""
market_making_env.py
Gymnasium-compatible market making environment.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.envs.lob_simulator import LOBSimulator, LOBState
from src.policies.avellaneda_stoikov import AvellanedaStoikov
from src.integration.volatility_model import VolatilityModel


class MarketMakingEnv(gym.Env):
    """
    Single-stock market making environment.

    Parameters
    ----------
    episode_steps : int
    sim_kwargs : dict
    as_kwargs : dict
    use_vol_features : bool
    vol_model : VolatilityModel | None
    inventory_limit : float
    inv_penalty_alpha : float
        Quadratic inventory penalty coefficient.
    inv_penalty_beta : float
        Linear inventory penalty coefficient.
    seed : int | None
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    # action index -> (delta_spread_idx, delta_skew_idx)
    _ACTION_MAP = {
        i * 3 + j: (i - 1, j - 1)
        for i in range(3) for j in range(3)
    }  # 9 actions total

    def __init__(
        self,
        episode_steps: int = 300,
        sim_kwargs: Optional[dict] = None,
        as_kwargs: Optional[dict] = None,
        use_vol_features: bool = True,
        vol_model: Optional[VolatilityModel] = None,
        inventory_limit: float = 10.0,
        inv_penalty_alpha: float = 0.01,
        inv_penalty_beta: float = 0.005,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.episode_steps = episode_steps
        self.use_vol_features = use_vol_features
        self.inventory_limit = inventory_limit
        self.inv_penalty_alpha = inv_penalty_alpha
        self.inv_penalty_beta = inv_penalty_beta

        self.sim = LOBSimulator(**(sim_kwargs or {}))
        self.as_policy = AvellanedaStoikov(**(as_kwargs or {}))
        self.vol_model: VolatilityModel = vol_model or VolatilityModel(stub=True)

        # action space: 9 discrete actions
        self.action_space = spaces.Discrete(9)

        # observation space
        base_dim = 8  # without vol features
        extra_dim = 2 if use_vol_features else 0
        obs_dim = base_dim + extra_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # internal episode state
        self._step: int = 0
        self._inventory: float = 0.0
        self._cash: float = 0.0
        self._prev_mid: float = 0.0
        self._bid_order_id: Optional[int] = None
        self._ask_order_id: Optional[int] = None
        self._spread_mult: float = 1.0
        self._skew_mult: float = 0.0
        self._lob_history: list[dict] = []
        self._seed = seed

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed or self._seed)
        lob = self.sim.reset(seed=seed or self._seed)
        self._step = 0
        self._inventory = 0.0
        self._cash = 0.0
        self._prev_mid = lob.mid_price
        self._bid_order_id = None
        self._ask_order_id = None
        self._spread_mult = 1.0
        self._skew_mult = 0.0
        self._lob_history = []
        obs = self._build_obs(lob)
        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        d_spread, d_skew = self._ACTION_MAP[int(action)]
        self._spread_mult = float(np.clip(self._spread_mult + d_spread * 0.1, 0.5, 3.0))
        self._skew_mult = float(np.clip(self._skew_mult + d_skew * 0.05, -0.5, 0.5))

        if self._bid_order_id is not None:
            self.sim.cancel_order(self._bid_order_id)
        if self._ask_order_id is not None:
            self.sim.cancel_order(self._ask_order_id)

        lob = self.sim._get_state()
        vol_est = self._get_vol_estimate()
        bid_p, ask_p = self.as_policy.compute_quotes(
            mid=lob.mid_price,
            inventory=self._inventory,
            t=self._step,
            T=self.episode_steps,
            sigma=vol_est,
            spread_mult=self._spread_mult,
            skew_mult=self._skew_mult,
        )

        self._bid_order_id, self._ask_order_id = self.sim.place_mm_quotes(
            bid_price=bid_p, ask_price=ask_p
        )

        lob = self.sim.step_sim()
        self._step += 1

        bid_filled = self.sim.get_fills(self._bid_order_id)
        ask_filled = self.sim.get_fills(self._ask_order_id)

        self._inventory += bid_filled - ask_filled
        self._cash += ask_filled * ask_p - bid_filled * bid_p

        unrealised = self._inventory * lob.mid_price
        pnl = self._cash + unrealised

        reward = self._compute_reward(
            bid_filled=bid_filled,
            ask_filled=ask_filled,
            bid_p=bid_p,
            ask_p=ask_p,
            pnl=pnl,
        )

        self._lob_history.append({
            "mid": lob.mid_price,
            "spread": lob.spread,
            "bid_depth": lob.bid_depth,
            "ask_depth": lob.ask_depth,
        })

        terminated = abs(self._inventory) > self.inventory_limit
        truncated = self._step >= self.episode_steps
        obs = self._build_obs(lob)
        info = {
            "pnl": pnl,
            "inventory": self._inventory,
            "bid_filled": bid_filled,
            "ask_filled": ask_filled,
            "spread": lob.spread,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        lob = self.sim._get_state()
        print(
            f"[t={self._step:4d}] mid={lob.mid_price:.3f} "
            f"inv={self._inventory:+.2f} "
            f"cash={self._cash:.3f}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(self, lob: LOBState) -> np.ndarray:
        vol_f, sys_score = self._get_vol_features()
        obs = [
            lob.mid_price / self.sim.initial_price - 1.0,
            lob.spread / self.sim.initial_price,
            float(np.log1p(lob.bid_depth)),
            float(np.log1p(lob.ask_depth)),
            self._inventory / self.inventory_limit,
            self._cash / (self.sim.initial_price * 100.0),
            (self._inventory * lob.mid_price) / (self.sim.initial_price * 100.0),
            (self.episode_steps - self._step) / self.episode_steps,
        ]
        if self.use_vol_features:
            obs += [vol_f * 20.0, sys_score]
        return np.array(obs, dtype=np.float32)

    def _compute_reward(
        self,
        bid_filled: float,
        ask_filled: float,
        bid_p: float,
        ask_p: float,
        pnl: float,
    ) -> float:
        spread_revenue = (ask_p - bid_p) * (bid_filled + ask_filled) / 2.0
        inventory_penalty = self.inv_penalty_alpha * (self._inventory ** 2)
        adverse_selection = self.inv_penalty_beta * abs(self._inventory)
        raw = float(spread_revenue - inventory_penalty - adverse_selection)
        return float(np.clip(raw, -50.0, 50.0))

    def _get_vol_estimate(self) -> float:
        if not self._lob_history:
            return self.sim.sigma
        window = self._lob_history[-100:]
        result = self.vol_model.predict(window)
        return float(result.get("vol_forecast", self.sim.sigma))

    def _get_vol_features(self) -> tuple[float, float]:
        if not self.use_vol_features or not self._lob_history:
            return 0.0, 0.0
        window = self._lob_history[-100:]
        result = self.vol_model.predict(window)
        return float(result.get("vol_forecast", 0.0)), float(result.get("systemic_score", 0.0))
