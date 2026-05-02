from __future__ import annotations
import argparse
import json
import numpy as np
from pathlib import Path

from src.envs.market_making_env import MarketMakingEnv
from src.integration.volatility_model import VolatilityModel
from src.policies.avellaneda_stoikov import AvellanedaStoikov
from src.policies.rl_dqn_agent import DQNMarketMaker
from src.utils.config import load_config
from src.utils.metrics import summarise


def make_env(cfg: dict, seed: int, use_vol: bool) -> MarketMakingEnv:
    vol_model = VolatilityModel(
        stub=not use_vol,
        default_vol=cfg["env"]["sim_kwargs"]["sigma"],
    )
    return MarketMakingEnv(
        episode_steps=cfg["env"]["episode_steps"],
        sim_kwargs={**cfg["env"]["sim_kwargs"], "seed": seed},
        as_kwargs=cfg["env"]["as_kwargs"],
        use_vol_features=use_vol,
        vol_model=vol_model,
        inventory_limit=cfg["env"]["inventory_limit"],
        inv_penalty_alpha=cfg["env"].get("inv_penalty_alpha", 0.001),
        inv_penalty_beta=cfg["env"].get("inv_penalty_beta", 0.0005),
        seed=seed,
    )


def run_episode_as(env: MarketMakingEnv, as_policy: AvellanedaStoikov) -> dict:
    obs, _ = env.reset()
    pnl, inv, b_fills, a_fills = [], [], [], []
    done = False
    while not done:
        action = 4  # neutral action — pure AS quotes, no RL adjustment
        obs, reward, terminated, truncated, info = env.step(action)
        pnl.append(info["pnl"])
        inv.append(info["inventory"])
        b_fills.append(info["bid_filled"])
        a_fills.append(info["ask_filled"])
        done = terminated or truncated
    return summarise(pnl, inv, b_fills, a_fills)


def run_episode_dqn(env: MarketMakingEnv, agent: DQNMarketMaker) -> dict:
    obs, _ = env.reset()
    pnl, inv, b_fills, a_fills = [], [], [], []
    done = False
    while not done:
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        pnl.append(info["pnl"])
        inv.append(info["inventory"])
        b_fills.append(info["bid_filled"])
        a_fills.append(info["ask_filled"])
        done = terminated or truncated
    return summarise(pnl, inv, b_fills, a_fills)


def main(config_path: str, n_seeds: int = 5) -> None:
    cfg = load_config(config_path)

    model_vol    = "data/processed/simulations/best_model_vol"
    model_no_vol = "data/processed/simulations/best_model_no_vol"

    results: dict[str, list] = {"AS": [], "DQN_no_vol": [], "DQN_vol": []}

    for seed in range(n_seeds):
        print(f"  running seed {seed}...")

        # AS baseline (no vol features needed)
        env_as = make_env(cfg, seed=seed + 100, use_vol=False)
        as_policy = AvellanedaStoikov(**cfg["env"]["as_kwargs"])
        results["AS"].append(run_episode_as(env_as, as_policy))

        # DQN without vol features
        try:
            env_nov = make_env(cfg, seed=seed + 100, use_vol=False)
            agent_nov = DQNMarketMaker.load(model_no_vol, env_nov)
            results["DQN_no_vol"].append(run_episode_dqn(env_nov, agent_nov))
        except Exception as e:
            print(f"    DQN_no_vol skipped: {e}")

        # DQN with vol features
        try:
            env_vol = make_env(cfg, seed=seed + 100, use_vol=True)
            agent_vol = DQNMarketMaker.load(model_vol, env_vol)
            results["DQN_vol"].append(run_episode_dqn(env_vol, agent_vol))
        except Exception as e:
            print(f"    DQN_vol skipped: {e}")

    # Save
    out_path = Path("data/processed/simulations/compare_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print table
    print("\n" + "=" * 76)
    print(f"{'Policy':<18} {'PnL mean':>10} {'PnL std':>10} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'InvVar':>8}")
    print("=" * 76)
    for policy, runs in results.items():
        if not runs:
            print(f"  {policy:<16}  no data (model not trained yet)")
            continue
        pnls    = [r["total_pnl"]          for r in runs]
        sharpes = [r["sharpe"]             for r in runs]
        sortin  = [r["sortino"]            for r in runs]
        mdds    = [r["max_drawdown"]       for r in runs]
        invvars = [r["inventory_variance"] for r in runs]
        print(
            f"{policy:<18}"
            f" {np.mean(pnls):>+10.4f}"
            f" {np.std(pnls):>10.4f}"
            f" {np.mean(sharpes):>8.3f}"
            f" {np.mean(sortin):>8.3f}"
            f" {np.mean(mdds):>8.4f}"
            f" {np.mean(invvars):>8.4f}"
        )
    print("=" * 76)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dqn_default.yaml")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    main(args.config, args.seeds)
