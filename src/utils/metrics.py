from __future__ import annotations
import numpy as np
from typing import Sequence

def sharpe(pnl_series: Sequence[float], rf: float = 0.0) -> float:
    arr = np.diff(np.array(pnl_series))   # per-step changes
    if len(arr) == 0 or arr.std() < 1e-12:
        return 0.0
    return float((arr.mean() - rf) / arr.std() * np.sqrt(252))

def sortino(pnl_series: Sequence[float], rf: float = 0.0) -> float:
    arr = np.diff(np.array(pnl_series))
    downside = arr[arr < rf]
    if len(downside) == 0 or downside.std() < 1e-12:
        return 0.0
    return float((arr.mean() - rf) / downside.std() * np.sqrt(252))

def max_drawdown(pnl_series: Sequence[float]) -> float:
    arr = np.array(pnl_series)
    running_max = np.maximum.accumulate(arr)
    return float((running_max - arr).max())

def inventory_variance(inv_series: Sequence[float]) -> float:
    return float(np.var(inv_series))

def fill_rate(bid_fills: Sequence[float], ask_fills: Sequence[float]) -> float:
    total_filled = sum(bid_fills) + sum(ask_fills)
    total_placed = len(bid_fills) + len(ask_fills)
    return total_filled / max(total_placed, 1)

def summarise(
    pnl_series: Sequence[float],
    inv_series: Sequence[float],
    bid_fills: Sequence[float],
    ask_fills: Sequence[float],
) -> dict:
    arr = np.array(pnl_series)
    return {
        "total_pnl":          float(arr[-1]) if len(arr) > 0 else 0.0,  # final PnL
        "sharpe":             sharpe(pnl_series),
        "sortino":            sortino(pnl_series),
        "max_drawdown":       max_drawdown(pnl_series),
        "inventory_variance": inventory_variance(inv_series),
        "final_inventory":    float(inv_series[-1]) if inv_series else 0.0,
        "fill_rate":          fill_rate(bid_fills, ask_fills),
    }
