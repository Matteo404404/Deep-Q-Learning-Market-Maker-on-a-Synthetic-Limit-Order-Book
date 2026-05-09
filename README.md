# QUANT2 — Deep Q-Learning Market Maker on a Synthetic Limit Order Book

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.x-green.svg)](https://stable-baselines3.readthedocs.io/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-orange.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A reinforcement-learning market maker that learns to quote bid/ask spreads on a synthetic limit order book, compared against an Avellaneda–Stoikov baseline.**

This repository implements a full RL pipeline for single-asset market making:

- A **Gymnasium-compatible LOB environment** with inventory and risk penalties  
- An **Avellaneda–Stoikov (AS)** analytical market-making policy as a baseline  
- A **DQN agent** (Stable-Baselines3) trained on the same environment  
- A **comparison harness** that evaluates AS vs DQN with and without volatility features  
- A **Streamlit dashboard** to visualize PnL, Sharpe/Sortino ratios, and inventory risk

The environment is fully synthetic—no real market data is required—making the project easy to reproduce and extend.[^as]

[^as]: For background on Avellaneda–Stoikov-style market making and RL approaches, see e.g. deep RL market-making papers using DQN and related methods.[^mm]

[^mm]: Examples include recent work on RL-based market making that compares RL agents to analytical controls using PnL and Sharpe ratio as key metrics.[^mmref]

---

## 📊 Results (fixed-sigma baseline)

For the base configuration with constant volatility and 200k training timesteps, the learned DQN without volatility features outperforms the AS baseline on risk-adjusted returns:

| Policy        | PnL mean | PnL std | Sharpe | Sortino | MaxDD | Inventory variance |
|--------------|---------:|--------:|-------:|--------:|------:|-------------------:|
| **AS**       |  ≈ 35    |   ≈ 80  | 0.10   | 2.00    | ≈ 51  | ≈ 55               |
| **DQN_no_vol** | ≈ 127   |  ≈ 150 | **1.95** | **3.13** | ≈ 93 | ≈ 70               |
| **DQN_vol**  | ≈ 74     |  ≈ 135 | 0.74   | 1.29    | ≈ 90  | ≈ 110              |

- **DQN_no_vol** achieves the best Sharpe and Sortino ratios, learning to exploit spread revenue while managing inventory better than AS.  
- Adding a **volatility feature** from the QUANT1 GNN does not improve performance in this synthetic environment, highlighting a mismatch between the real-data-trained vol model and the simplified simulator.

A second set of experiments introduces **stochastic volatility regimes** (low / mid / high sigma) to stress-test the policies. Under the harder regime setup, AS becomes competitive in mean PnL while the DQN agents trade more conservatively with lower inventory variance.

---

## 🏗 Architecture

```text
Synthetic LOB Simulator (single asset)
  └── MarketMakingEnv (Gymnasium Env)
        -  State: mid-price, spread, depth, inventory, cash, unrealized PnL,
                 time-to-maturity, optional vol + systemic features
        -  Action: 9 discrete combinations of spread and skew adjustments
        -  Reward: spread revenue – quadratic + linear inventory penalties

Baselines
  ├── Avellaneda–Stoikov quoting policy
  └── DQN agent (Stable-Baselines3)
        -  Policy: MLP
        -  Replay buffer, epsilon-greedy exploration
        -  Target network, double DQN

Experiments
  ├── train_dqn.py          → train DQN on MarketMakingEnv
  └── compare_policies.py   → evaluate AS vs DQN across multiple seeds

Visualization
  └── dashboard/app.py      → Streamlit dashboard for PnL / Sharpe / inventory risk
```

---

## 📁 Project Structure

```text
QUANT2/
├── configs/
│   └── dqn_default.yaml            # Environment + DQN hyperparameters
│
├── data/
│   └── processed/
│       └── simulations/
│           ├── dqn_mm_final.zip    # Final DQN checkpoint (no_vol)
│           ├── best_model_no_vol.zip
│           ├── best_model_vol.zip
│           └── compare_results.json  # AS vs DQN_no_vol vs DQN_vol metrics
│
├── src/
│   ├── envs/
│   │   ├── lob_simulator.py        # Synthetic LOB simulator with vol regimes
│   │   └── market_making_env.py    # Gymnasium Env (reward + observation)
│   │
│   ├── policies/
│   │   ├── avellaneda_stoikov.py   # Analytical MM baseline
│   │   └── rl_dqn_agent.py         # Thin wrapper around SB3 DQN
│   │
│   └── experiments/
│       ├── train_dqn.py            # Training script
│       └── compare_policies.py     # Multi-seed evaluation script
│
├── dashboard/
│   └── app.py                      # Streamlit dashboard
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

### 1. Install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/QUANT2.git
cd QUANT2

pip install -r requirements.txt
```

Key libraries:

- `stable-baselines3` (DQN implementation)[^sb3]
- `gymnasium` (environment API)[^gym]
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `streamlit` (dashboard)

[^sb3]: Stable-Baselines3 docs provide DQN hyperparameter and reward-scaling tips for custom environments.[^sb3ref]

[^gym]: Gymnasium defines the `Env` interface and the `reset`/`step` signatures used here.[^gymref]

### 2. Set `PYTHONPATH`

From the repo root:

```bash
export PYTHONPATH="$(pwd):/home/plugga404/Documents/VS files/QUANTS/QUANT1/src"
```

Adjust the second path if QUANT1 (the Optiver GNN project) lives elsewhere. The volatility model from QUANT1 is optional; the environment will fall back to a stub vol model if not available.

### 3. Train the DQN agent

```bash
python src/experiments/train_dqn.py --config configs/dqn_default.yaml
```

This will:

- Instantiate `MarketMakingEnv` with parameters from the YAML config  
- Train a DQN agent for `training.total_timesteps` (default: 200,000)  
- Evaluate the trained policy on a fixed evaluation seed  
- Save the final model to `data/processed/simulations/dqn_mm_final.zip` and the best checkpoint to `best_model_no_vol.zip`

Training logs (episode reward, loss, epsilon) are written to `data/processed/simulations/tb_logs/` for TensorBoard.

### 4. Evaluate AS vs DQN (multi-seed)

```bash
python src/experiments/compare_policies.py --config configs/dqn_default.yaml --seeds 5
```

This script:

- Runs **AS**, **DQN_no_vol**, and **DQN_vol** over `N` random seeds  
- Collects per-episode PnL trajectories for each policy  
- Computes summary statistics: PnL mean/std, Sharpe, Sortino, max drawdown, inventory variance  
- Writes results to `data/processed/simulations/compare_results.json` and prints a formatted table in the terminal

---

## 📈 Dashboard

Launch the dashboard to explore results interactively:

```bash
streamlit run dashboard/app.py
```

Views:

- **Performance:** bar charts of Sharpe and Sortino by policy, PnL distributions, drawdowns  
- **Inventory risk:** average inventory variance and terminal inventory by policy  
- **Raw results:** table of per-seed metrics

---

## 🔍 Implementation Notes

- **Reward shaping:** The reward includes
  - spread revenue on filled volume  
  - a **quadratic inventory penalty** (`inv_penalty_alpha * inv²`)  
  - a **linear adverse selection penalty** (`inv_penalty_beta * |inv|`)  
  Clipping to a bounded range stabilizes DQN training, as recommended in SB3 tips.[^sb3tips]

- **Volatility regimes:** In some experiments, the LOB simulator samples a volatility regime (low/mid/high sigma) per episode. This is used to test robustness; the main headline results are reported in the fixed-sigma configuration.

- **Volatility features:** When available, the vol model from QUANT1 provides a `vol_forecast` and optional systemic risk score. These are appended to the observation vector. In the current synthetic environment, this additional signal does **not** improve performance, which is discussed in the results.

---

## 🧪 Possible Extensions

If you want to take this further:

- Replace the synthetic LOB simulator with a **replay-based environment** using real LOB data  
- Try alternative RL algorithms (e.g. C51, QR-DQN, PPO) and compare against DQN  
- Model multiple correlated assets and extend the AS baseline to a **multi-asset** setting  
- Add **transaction cost modeling** (fees, rebates, inventory holding costs)

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.
