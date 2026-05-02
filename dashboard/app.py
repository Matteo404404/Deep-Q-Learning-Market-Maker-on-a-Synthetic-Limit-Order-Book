"""
app.py — Streamlit dashboard for RL Market Maker results.
Run: PYTHONPATH=. streamlit run dashboard/app.py
"""
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="RL Market Maker", layout="wide")
st.title("RL Market Maker — Results Dashboard")

RESULTS_PATH = Path("data/processed/simulations/compare_results.json")

if not RESULTS_PATH.exists():
    st.warning("No results found. Run `compare_policies.py` first.")
    st.stop()

with open(RESULTS_PATH) as f:
    results = json.load(f)

# ── Summary metrics ──────────────────────────────────────────────────────────
st.subheader("Policy comparison (mean ± std across seeds)")

cols = st.columns(len(results))
for col, (policy, runs) in zip(cols, results.items()):
    if not runs:
        col.metric(policy, "No data")
        continue
    pnl_mean    = np.mean([r["total_pnl"]    for r in runs])
    sharpe_mean = np.mean([r["sharpe"]        for r in runs])
    mdd_mean    = np.mean([r["max_drawdown"]  for r in runs])
    inv_mean    = np.mean([r["inventory_variance"] for r in runs])
    col.markdown(f"### {policy}")
    col.metric("Total PnL",       f"{pnl_mean:+.2f}")
    col.metric("Sharpe",          f"{sharpe_mean:.3f}")
    col.metric("Max Drawdown",    f"{mdd_mean:.2f}")
    col.metric("Inventory Var",   f"{inv_mean:.3f}")

# ── PnL bar chart ─────────────────────────────────────────────────────────────
st.subheader("Mean Total PnL by policy")
fig, ax = plt.subplots(figsize=(8, 4))
labels  = [p for p, r in results.items() if r]
means   = [np.mean([x["total_pnl"] for x in results[p]]) for p in labels]
stds    = [np.std ([x["total_pnl"] for x in results[p]]) for p in labels]
colors  = ["#4C72B0", "#DD8452", "#55A868"][:len(labels)]
ax.bar(labels, means, yerr=stds, capsize=6, color=colors, alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_ylabel("Total PnL")
ax.set_title("AS vs DQN (no vol) vs DQN (vol)")
st.pyplot(fig)

# ── Sharpe / Sortino bar chart ────────────────────────────────────────────────
st.subheader("Risk-adjusted metrics")
fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, metric, title in zip(
    axes,
    ["sharpe", "sortino"],
    ["Sharpe Ratio", "Sortino Ratio"],
):
    vals = [np.mean([x[metric] for x in results[p]]) for p in labels]
    ax.bar(labels, vals, color=colors, alpha=0.85)
    ax.set_title(title)
    ax.set_ylabel(title)
st.pyplot(fig2)

# ── Inventory variance chart ──────────────────────────────────────────────────
st.subheader("Inventory control")
fig3, ax3 = plt.subplots(figsize=(8, 4))
inv_means = [np.mean([x["inventory_variance"] for x in results[p]]) for p in labels]
ax3.bar(labels, inv_means, color=colors, alpha=0.85)
ax3.set_ylabel("Inventory Variance (lower = better)")
ax3.set_title("Inventory risk by policy")
st.pyplot(fig3)

# ── Raw results table ─────────────────────────────────────────────────────────
st.subheader("Raw results (all seeds)")
for policy, runs in results.items():
    if not runs:
        continue
    st.markdown(f"**{policy}**")
    rows = []
    for i, r in enumerate(runs):
        rows.append({
            "seed":       i,
            "total_pnl":  round(r["total_pnl"], 4),
            "sharpe":     round(r["sharpe"], 3),
            "sortino":    round(r["sortino"], 3),
            "max_dd":     round(r["max_drawdown"], 4),
            "inv_var":    round(r["inventory_variance"], 4),
            "final_inv":  round(r["final_inventory"], 2),
        })
    st.dataframe(rows)
