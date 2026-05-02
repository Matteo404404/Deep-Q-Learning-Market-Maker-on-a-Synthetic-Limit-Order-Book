"""
lob_simulator.py
================
Core single-stock limit order book simulator.

Mid-price follows a discrete-time log-normal (GBM) diffusion.
Market and limit order arrivals follow independent Poisson processes.
Matching engine uses price-time priority with partial fills.

References
----------
Avellaneda & Stoikov (2008). Quantitative Finance 8(3), 217-224.
LOB simulation review: arXiv 2402.17359
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Order:
    """A single resting limit order."""
    order_id:  int
    side:      str    # "bid" or "ask"
    price:     float
    quantity:  float
    timestamp: int    # step at which it was placed


@dataclass
class LOBState:
    """Top-of-book snapshot."""
    mid_price:  float
    best_bid:   float
    best_ask:   float
    bid_depth:  float   # total qty at best bid
    ask_depth:  float   # total qty at best ask
    spread:     float

    @classmethod
    def empty(cls) -> "LOBState":
        return cls(
            mid_price=0.0, best_bid=0.0, best_ask=0.0,
            bid_depth=0.0, ask_depth=0.0, spread=0.0,
        )


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class LOBSimulator:
    """
    Single-stock discrete-time LOB simulator.

    Parameters
    ----------
    initial_price : float
        Starting mid-price.
    sigma : float
        Per-step price volatility (GBM diffusion coefficient).
    tick_size : float
        Minimum price increment.
    dt : float
        Duration of each time step in seconds.
    lambda_market : float
        Poisson arrival rate for market orders per step.
    lambda_limit : float
        Poisson arrival rate for limit orders per step per side.
    lambda_cancel : float
        Per-step cancellation probability for each resting order.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        initial_price:  float           = 100.0,
        sigma:          float           = 0.02,
        tick_size:      float           = 0.01,
        dt:             float           = 1.0,
        lambda_market:  float           = 5.0,
        lambda_limit:   float           = 8.0,
        lambda_cancel:  float           = 0.1,
        seed:           Optional[int]   = None,
    ) -> None:
        self.initial_price = initial_price
        self.sigma         = sigma
        self.tick_size     = tick_size
        self.dt            = dt
        self.lambda_market = lambda_market
        self.lambda_limit  = lambda_limit
        self.lambda_cancel = lambda_cancel
        self.rng           = np.random.default_rng(seed)

        # mutable state
        self.mid_price:          float        = initial_price
        self.step:               int          = 0
        self._order_id_counter:  int          = 0
        self._bids:              list[Order]  = []
        self._asks:              list[Order]  = []
        self.trade_history:      list[dict]   = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> LOBState:
        """Reset the simulator to its initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.mid_price          = self.initial_price
        self.step               = 0
        self._order_id_counter  = 0
        self._bids              = []
        self._asks              = []
        self.trade_history      = []
        return self._get_state()

    def step_sim(self) -> LOBState:
        """
        Advance simulator by one time step.

        Order of operations:
          1. GBM mid-price update.
          2. Poisson market-order arrivals (may fill MM quotes).
          3. Poisson limit-order arrivals from background participants.
          4. Random cancellations.
        """
        self._update_mid_price()
        self._process_market_orders()
        self._process_limit_arrivals()
        self._process_cancellations()
        self.step += 1
        return self._get_state()

    def place_mm_quotes(
        self,
        bid_price: float,
        ask_price: float,
        bid_qty:   float = 1.0,
        ask_qty:   float = 1.0,
    ) -> tuple[int, int]:
        """
        Place market-maker bid and ask resting limit orders.

        Returns
        -------
        (bid_order_id, ask_order_id)
        """
        bid_price = self._round_tick(bid_price)
        ask_price = self._round_tick(ask_price)
        bid_id    = self._add_order("bid", bid_price, bid_qty)
        ask_id    = self._add_order("ask", ask_price, ask_qty)
        return bid_id, ask_id

    def cancel_order(self, order_id: int) -> bool:
        """Cancel a resting order by id.  Returns True if found."""
        for book in (self._bids, self._asks):
            for i, o in enumerate(book):
                if o.order_id == order_id:
                    book.pop(i)
                    return True
        return False

    def get_fills(self, order_id: int) -> float:
        """Return total filled quantity for a given order_id since last call."""
        return sum(
            t["qty"] for t in self.trade_history if t["order_id"] == order_id
        )

    # ------------------------------------------------------------------
    # State snapshot  (two aliases for backward compatibility)
    # ------------------------------------------------------------------

    def _get_state(self) -> LOBState:
        """Compute and return the current top-of-book state."""
        best_bid = max(
            (o.price for o in self._bids),
            default=self.mid_price - self.tick_size,
        )
        best_ask = min(
            (o.price for o in self._asks),
            default=self.mid_price + self.tick_size,
        )
        # safety: ensure no crossed book
        if best_ask <= best_bid:
            best_ask = best_bid + self.tick_size

        bid_depth = sum(o.quantity for o in self._bids if o.price == best_bid)
        ask_depth = sum(o.quantity for o in self._asks if o.price == best_ask)
        mid       = (best_bid + best_ask) / 2.0

        return LOBState(
            mid_price  = mid,
            best_bid   = best_bid,
            best_ask   = best_ask,
            bid_depth  = bid_depth,
            ask_depth  = ask_depth,
            spread     = best_ask - best_bid,
        )

    # Alias so both `sim._get_state()` and `sim._snap()` work.
    _snap = _get_state

    # ------------------------------------------------------------------
    # Internal mechanics
    # ------------------------------------------------------------------

    def _update_mid_price(self) -> None:
        """Log-normal (GBM) mid-price update."""
        dW = self.rng.normal(0.0, 1.0)
        self.mid_price = max(
            self.tick_size,
            self.mid_price * np.exp(self.sigma * np.sqrt(self.dt) * dW),
        )

    def _process_market_orders(self) -> None:
        """Simulate arriving market orders that may fill resting MM quotes."""
        n_mo = self.rng.poisson(self.lambda_market)
        for _ in range(n_mo):
            side = self.rng.choice(["buy", "sell"])
            qty  = self.rng.exponential(1.0)
            if side == "buy":
                self._fill_side(self._asks, qty, "ask")
            else:
                self._fill_side(self._bids, qty, "bid")

    def _fill_side(self, book: list[Order], qty: float, side: str) -> None:
        """Walk the book and fill resting orders up to qty."""
        remaining  = qty
        reverse    = side == "bid"   # bids: highest price first
        book.sort(key=lambda o: o.price, reverse=reverse)
        filled_ids: list[int] = []

        for order in book:
            if remaining <= 0:
                break
            fill_qty = min(order.quantity, remaining)
            self.trade_history.append({
                "step":     self.step,
                "order_id": order.order_id,
                "price":    order.price,
                "qty":      fill_qty,
                "side":     side,
            })
            order.quantity -= fill_qty
            remaining      -= fill_qty
            if order.quantity <= 1e-9:
                filled_ids.append(order.order_id)

        book[:] = [o for o in book if o.order_id not in filled_ids]

    def _process_limit_arrivals(self) -> None:
        """Background limit orders from other participants."""
        for side in ("bid", "ask"):
            n_lo = self.rng.poisson(self.lambda_limit)
            for _ in range(n_lo):
                offset = abs(self.rng.normal(0.0, 2.0 * self.tick_size))
                price  = (
                    self.mid_price - offset if side == "bid"
                    else self.mid_price + offset
                )
                qty = self.rng.exponential(1.0)
                self._add_order(side, self._round_tick(price), qty)

    def _process_cancellations(self) -> None:
        """Randomly cancel a fraction of resting orders each step."""
        for book in (self._bids, self._asks):
            book[:] = [
                o for o in book
                if self.rng.random() > self.lambda_cancel
            ]

    def _add_order(self, side: str, price: float, qty: float) -> int:
        oid   = self._order_id_counter
        self._order_id_counter += 1
        order = Order(oid, side, price, qty, self.step)
        (self._bids if side == "bid" else self._asks).append(order)
        return oid

    def _round_tick(self, price: float) -> float:
        return round(round(price / self.tick_size) * self.tick_size, 10)