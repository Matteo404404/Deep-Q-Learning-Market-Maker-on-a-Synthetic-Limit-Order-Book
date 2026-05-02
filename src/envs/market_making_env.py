"""
market_making_env.py
====================
Gymnasium-compatible market making environment.

Observation (10-dim with vol features, 8-dim without):
  [norm_mid, spread, log_bid_depth, log_ask_depth,
   norm_inventory, norm_cash, norm_unrealised,
   time_remaining, vol_forecast*, systemic_score*]
  (* only when use_vol_features=True)

Action space: Discrete(9)
  3x3 grid of (delta_spread_mult, delta_skew_mult) adjustments.
  action = 3*i + j,  i/j in {0,1,2}  ->  delta in {-1, 0, +1}

Reward (per step, no terminal cliff):
  R_t = spread_revenue - alpha * inventory^2 - beta * |inventory|

  Calibration:
    spread_revenue  ~  0.005 - 0.02  per fill (realistic for 2-tick spread)
    inventory^2     :  alpha=0.001  ->  inv=5  costs 0.025  (~comparable)
                                        inv=10 costs 0.100  (strongly negative)
    |inventory|     :  beta=0.0005  (small linear drag)

  No hard terminal penalty.  Episode terminates when |inventory| >
  inventory_limit, which by itself cuts off future positive rewards —
  that IS the penalty for the agent.

  This design follows the reward shaping in:
    Avellaneda & Stoikov (2008) — inventory-penalised PnL
    Ragel (2024) — continuous shaping without terminal cliffs
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

from src.envs.lob_simulator import LOBSimulator, LOBState
from src.policies.avellaneda_stoikov import AvellanedaStoikov
from src.integration.volatility_model import VolatilityModel


# action index -> (delta_spread_mult, delta_skew_mult) each in {-1, 0, +1}
_ACTION_MAP: dict[int, tuple[int, int]] = {
    3 * i + j: (i - 1, j - 1) for i in range(3) for j in range(3)
}


class MarketMakingEnv(gym.Env):
    """
    Single-stock RL market making environment.

    Parameters
    ----------
    episode_steps : int
        Maximum number of simulator steps per episode (truncation horizon).
    sim_kwargs : dict | None
        Keyword arguments forwarded to LOBSimulator.
    as_kwargs : dict | None
        Keyword arguments forwarded to AvellanedaStoikov.
    use_vol_features : bool
        Whether to include volatility forecast in the observation vector.
    vol_model : VolatilityModel | None
        Pre-built volatility model.  If None, a stub returning zeros is used.
    inventory_limit : float
        Hard inventory cap.  Episode terminates if |inventory| exceeds this.
    inv_penalty_alpha : float
        Quadratic inventory penalty coefficient.
    inv_penalty_beta : float
        Linear inventory penalty coefficient (adverse selection proxy).
    seed : int | None
        Default random seed for resets.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        episode_steps:      int                       = 300,
        sim_kwargs:         Optional[dict]            = None,
        as_kwargs:          Optional[dict]            = None,
        use_vol_features:   bool                      = True,
        vol_model:          Optional[VolatilityModel] = None,
        inventory_limit:    float                     = 10.0,
        inv_penalty_alpha:  float                     = 0.001,
        inv_penalty_beta:   float                     = 0.0005,
        seed:               Optional[int]             = None,
    ) -> None:
        super().__init__()

        self.episode_steps     = episode_steps
        self.use_vol_features  = use_vol_features
        self.inventory_limit   = inventory_limit
        self.inv_penalty_alpha = inv_penalty_alpha
        self.inv_penalty_beta  = inv_penalty_beta
        self._seed             = seed

        self.sim       = LOBSimulator(**(sim_kwargs or {}))
        self.as_policy = AvellanedaStoikov(**(as_kwargs or {}))
        self.vol_model = vol_model or VolatilityModel(stub=True)

        # action / observation spaces
        self.action_space = spaces.Discrete(9)
        obs_dim = 10 if use_vol_features else 8
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # episode-level state (initialised in reset)
        self._step:        int           = 0
        self._inventory:   float         = 0.0
        self._cash:        float         = 0.0
        self._bid_id:      Optional[int] = None
        self._ask_id:      Optional[int] = None
        self._spread_mult: float         = 1.0
        self._skew_mult:   float         = 0.0
        self._lob_history: list[dict]    = []

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed:    Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed if seed is not None else self._seed)

        effective_seed = seed if seed is not None else self._seed
        lob = self.sim.reset(seed=effective_seed)

        self._step        = 0
        self._inventory   = 0.0
        self._cash        = 0.0
        self._bid_id      = None
        self._ask_id      = None
        self._spread_mult = 1.0
        self._skew_mult   = 0.0
        self._lob_history = []

        return self._build_obs(lob), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:

        # 1. Decode action -> spread / skew multiplier adjustments
        d_spr, d_skw      = _ACTION_MAP[int(action)]
        self._spread_mult = float(np.clip(self._spread_mult + d_spr * 0.1, 0.5, 3.0))
        self._skew_mult   = float(np.clip(self._skew_mult  + d_skw * 0.05, -0.5, 0.5))

        # 2. Cancel previous MM quotes
        if self._bid_id is not None:
            self.sim.cancel_order(self._bid_id)
        if self._ask_id is not None:
            self.sim.cancel_order(self._ask_id)

        # 3. Compute AS-guided quotes, modulated by RL multipliers
        lob_before = self.sim._get_state()
        sigma      = self._get_vol_estimate()
        bid_p, ask_p = self.as_policy.compute_quotes(
            mid          = lob_before.mid_price,
            inventory    = self._inventory,
            t            = self._step,
            T            = self.episode_steps,
            sigma        = sigma,
            spread_mult  = self._spread_mult,
            skew_mult    = self._skew_mult,
        )

        # 4. Place new MM quotes
        self._bid_id, self._ask_id = self.sim.place_mm_quotes(bid_p, ask_p)

        # 5. Advance simulator by one step
        lob = self.sim.step_sim()
        self._step += 1

        # 6. Account for fills and update portfolio
        bid_filled = self.sim.get_fills(self._bid_id)
        ask_filled = self.sim.get_fills(self._ask_id)

        self._inventory += bid_filled - ask_filled
        self._cash      += ask_filled * ask_p - bid_filled * bid_p

        # 7. Compute reward — no terminal cliff, no scaling
        reward = self._compute_reward(bid_filled, ask_filled, bid_p, ask_p)

        # 8. Check termination and truncation
        terminated = abs(self._inventory) > self.inventory_limit
        truncated  = self._step >= self.episode_steps

        # 9. Update LOB history for vol model (capped to last 60 steps)
        self._lob_history.append({
            "mid":       lob.mid_price,
            "spread":    lob.spread,
            "bid_depth": lob.bid_depth,
            "ask_depth": lob.ask_depth,
        })
        if len(self._lob_history) > 60:
            self._lob_history.pop(0)

        pnl = self._cash + self._inventory * lob.mid_price

        info = {
            "pnl":        pnl,
            "inventory":  self._inventory,
            "bid_filled": bid_filled,
            "ask_filled": ask_filled,
            "spread":     lob.spread,
            "spread_mult": self._spread_mult,
        }

        return self._build_obs(lob), reward, terminated, truncated, info

    def render(self) -> None:
        lob = self.sim._get_state()
        print(
            f"[t={self._step:4d}] mid={lob.mid_price:.3f}  "
            f"inv={self._inventory:+.2f}  cash={self._cash:.4f}  "
            f"spr_mult={self._spread_mult:.2f}"
        )

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        bid_filled: float,
        ask_filled: float,
        bid_p:      float,
        ask_p:      float,
    ) -> float:
        """
        Per-step reward with no terminal cliff.

        R_t = spread_revenue - alpha * q^2 - beta * |q|

        where q = self._inventory after this step's fills.

        Calibration targets:
          - inv = 0, 1 fill at 2-tick spread (0.02): R ≈ +0.010
          - inv = 5, no fills:                        R ≈ -0.025
          - inv = 10, no fills:                       R ≈ -0.100
          → agent learns: staying near zero inventory is profitable.
        """
        spread_rev  = (ask_p - bid_p) * (bid_filled + ask_filled) / 2.0
        inv_sq_cost = self.inv_penalty_alpha * (self._inventory ** 2)
        inv_lin_cost = self.inv_penalty_beta  * abs(self._inventory)
        return float(spread_rev - inv_sq_cost - inv_lin_cost)

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_obs(self, lob: LOBState) -> np.ndarray:
        vol_f, sys_s = self._get_vol_features()
        obs = [
            lob.mid_price / self.sim.initial_price - 1.0,           # normalised mid
            lob.spread    / self.sim.initial_price,                  # normalised spread
            np.log1p(lob.bid_depth),
            np.log1p(lob.ask_depth),
            self._inventory   / self.inventory_limit,                # in [-1, +1]
            self._cash        / (self.sim.initial_price * 10.0),
            (self._inventory * lob.mid_price) / (self.sim.initial_price * 10.0),
            (self.episode_steps - self._step) / self.episode_steps,  # time remaining
        ]
        if self.use_vol_features:
            obs += [float(vol_f), float(sys_s)]
        return np.array(obs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Volatility helpers
    # ------------------------------------------------------------------

    def _get_vol_estimate(self) -> float:
        """Scalar vol for AS spread computation."""
        if not self._lob_history:
            return self.sim.sigma
        result = self.vol_model.predict(self._lob_history)
        return float(result.get("vol_forecast", self.sim.sigma))

    def _get_vol_features(self) -> tuple[float, float]:
        """Return (vol_forecast, systemic_score) for the observation vector."""
        if not self.use_vol_features or not self._lob_history:
            return 0.0, 0.0
        result = self.vol_model.predict(self._lob_history)
        return (
            float(result.get("vol_forecast",   0.0)),
            float(result.get("systemic_score", 0.0)),
        )