"""
avellaneda_stoikov.py
=====================
Avellaneda-Stoikov (2008) analytic market making policy.

Reservation price:  r = s - q·γ·σ²·(T-t)
Optimal half-spread: δ = γ·σ²·(T-t)/2 + (1/γ)·ln(1 + γ/κ)

Extensions:
  - Inventory hard cap (stop quoting on excess side)
  - Directional drift in reservation price
  - RL-controlled spread_mult and skew_mult applied on top
"""
from __future__ import annotations

import numpy as np
from typing import Optional


class AvellanedaStoikov:
    """
    Parameters
    ----------
    gamma         : risk-aversion coefficient (0.01 – 0.5 typical)
    kappa         : market-depth / fill-rate param (0.5 – 3.0 typical)
    inventory_cap : hard cap on |inventory|; disable quoting on excess side
    drift         : deterministic drift added to reservation price
                    (set externally from a directional signal)
    """

    def __init__(
        self,
        gamma:         float = 0.1,
        kappa:         float = 1.5,
        inventory_cap: float = 10.0,
        drift:         float = 0.0,
    ) -> None:
        self.gamma         = gamma
        self.kappa         = kappa
        self.inventory_cap = inventory_cap
        self.drift         = drift

    def compute_quotes(
        self,
        mid:         float,
        inventory:   float,
        t:           int,
        T:           int,
        sigma:       Optional[float] = None,
        spread_mult: float = 1.0,
        skew_mult:   float = 0.0,
    ) -> tuple[float, float]:
        """
        Returns (bid_price, ask_price).

        Parameters
        ----------
        mid          : current mid-price
        inventory    : current net inventory (+ve = long)
        t            : current step
        T            : total episode steps
        sigma        : per-step vol; falls back to 0.02 if None
        spread_mult  : RL multiplier on half-spread (1.0 = pure AS)
        skew_mult    : RL additional skew on reservation price
        """
        sigma = sigma or 0.02
        g     = self.gamma
        k     = self.kappa
        tau   = max(T - t, 1) / T   # normalised time remaining ∈ (0,1]

        # reservation price (AS eq. 7)
        r = (mid
             - inventory * g * sigma**2 * tau
             + self.drift
             + skew_mult * sigma * np.sqrt(tau))

        # optimal half-spread (AS eq. 8)
        delta = (g * sigma**2 * tau / 2.0
                 + (1.0 / g) * np.log(1.0 + g / k))
        delta = max(delta * spread_mult, 1e-6)

        bid = r - delta
        ask = r + delta

        # inventory hard cap: neutralise the side that worsens inventory
        INF = mid * 1e6
        if inventory >= self.inventory_cap:
            bid = mid - INF   # effectively withdrawn
        elif inventory <= -self.inventory_cap:
            ask = mid + INF

        return float(bid), float(ask)

    def set_drift(self, drift: float) -> None:
        """Update directional drift from an external signal (e.g., vol forecast)."""
        self.drift = drift