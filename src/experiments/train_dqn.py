"""
train_dqn.py
============
Entry-point to train the DQN market maker.

Usage:
  python -m src.experiments.train_dqn --config configs/dqn_default.yaml
"""
from __future__ import annotations

import argparse
import random

import numpy as np

from src.envs.market_making_env import MarketMakingEnv
from src.integration.volatility_model import VolatilityModel
from src.policies.rl_dqn_agent import DQNMarketMaker
from src.utils.config import load_config


def make_env(cfg: dict, seed: int, use_vol: bool) -> MarketMakingEnv:
    """Factory so train and eval envs are built identically."""
    vol_model = VolatilityModel(
        stub        = not use_vol,
        default_vol = cfg["env"]["sim_kwargs"]["sigma"],
    )
    return MarketMakingEnv(
        episode_steps     = cfg["env"]["episode_steps"],
        sim_kwargs        = {**cfg["env"]["sim_kwargs"], "seed": seed},
        as_kwargs         = cfg["env"]["as_kwargs"],
        use_vol_features  = use_vol,
        vol_model         = vol_model,
        inventory_limit   = cfg["env"]["inventory_limit"],
        inv_penalty_alpha = cfg["env"].get("inv_penalty_alpha", 0.001),
        inv_penalty_beta  = cfg["env"].get("inv_penalty_beta",  0.0005),
        seed              = seed,
    )


def main(config_path: str) -> None:
    cfg  = load_config(config_path)
    seed = cfg["training"]["seed"]
    use_vol = cfg["env"]["use_vol_features"]

    random.seed(seed)
    np.random.seed(seed)

    train_env = make_env(cfg, seed=seed,     use_vol=use_vol)
    eval_env  = make_env(cfg, seed=seed + 1, use_vol=use_vol)

    agent = DQNMarketMaker(
        env             = train_env,
        learning_rate   = cfg["agent"]["learning_rate"],
        buffer_size     = cfg["agent"]["buffer_size"],
        batch_size      = cfg["agent"]["batch_size"],
        gamma           = cfg["agent"]["gamma"],
        exploration_fraction   = cfg["agent"]["exploration_fraction"],
        exploration_final_eps  = cfg["agent"]["exploration_final_eps"],
        target_update_interval = cfg["agent"]["target_update_interval"],
        learning_starts        = cfg["agent"].get("learning_starts", 2000),
        tensorboard_log = "data/processed/simulations/tb_logs",
        seed            = seed,
    )

    agent.train(
        total_timesteps = cfg["training"]["total_timesteps"],
        eval_env        = eval_env,
        eval_freq       = cfg["training"]["eval_freq"],
        save_dir        = "data/processed/simulations",
    )
    agent.save("data/processed/simulations/dqn_mm_final")
    print("Training complete, model saved to data/processed/simulations/dqn_mm_final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dqn_default.yaml")
    args = parser.parse_args()
    main(args.config)