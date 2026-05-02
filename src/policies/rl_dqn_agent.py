"""
rl_dqn_agent.py
===============
DQN market-making agent via Stable-Baselines3.

Hyperparameter defaults follow:
  - arXiv 2305.15821 (lr~1e-4, buffer~100k, batch~128)
  - Ragel 2024 thesis (gamma~0.99, exploration_fraction~0.5-0.7)
  - Double DQN enabled by default in SB3's DQN implementation
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from src.envs.market_making_env import MarketMakingEnv


class DQNMarketMaker:

    def __init__(
        self,
        env:                    MarketMakingEnv,
        learning_rate:          float           = 1e-4,
        buffer_size:            int             = 100_000,
        batch_size:             int             = 128,
        gamma:                  float           = 0.99,
        exploration_fraction:   float           = 0.7,
        exploration_final_eps:  float           = 0.02,
        target_update_interval: int             = 200,
        learning_starts:        int             = 2000,
        tensorboard_log:        Optional[str]   = None,
        seed:                   Optional[int]   = 42,
    ) -> None:
        self.env = Monitor(env)

        self.model = DQN(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            target_update_interval=target_update_interval,
            learning_starts=learning_starts,
            optimize_memory_usage=False,
            policy_kwargs={"net_arch": [256, 256]},
            tensorboard_log=tensorboard_log,
            verbose=1,
            seed=seed,
        )

    def train(
        self,
        total_timesteps: int                        = 200_000,
        eval_env:        Optional[MarketMakingEnv]  = None,
        eval_freq:       int                        = 5_000,
        save_dir:        str                        = "data/processed/simulations",
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)

        callbacks = [
            CheckpointCallback(
                save_freq=eval_freq,
                save_path=save_dir,
                name_prefix="dqn_mm",
            )
        ]
        if eval_env is not None:
            callbacks.append(
                EvalCallback(
                    Monitor(eval_env),
                    best_model_save_path=save_dir,
                    log_path=save_dir,
                    eval_freq=eval_freq,
                    n_eval_episodes=10,
                    deterministic=True,
                )
            )

        self.model.learn(total_timesteps=total_timesteps, callback=callbacks)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)

    def save(self, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, path: str, env: MarketMakingEnv) -> "DQNMarketMaker":
        obj       = cls.__new__(cls)
        obj.env   = Monitor(env)
        obj.model = DQN.load(path, env=obj.env)
        return obj