"""
volatility_model.py
===================
Adapter around QUANT1's GraphSAGE volatility model for use in the RL env.

Integration path (QUANT1):
  - Architecture:  quant_optiver.models.gnn_volatility.SAGEVolModel
  - Weights:       sage_vol_best.pt (checkpoint with keys
                    ['model_state', 'cfg', 'y_mean', 'y_std'])

We:
  - Build a minimal single-node graph from the synthetic LOB history.
  - Infer in_channels, hidden, n_layers from the checkpoint shapes.
  - Apply y_mean/y_std to de-standardise the model output.
  - Clamp the resulting vol to a sane range.

If stub=True, we never touch QUANT1 and just return default_vol.
"""

from __future__ import annotations

import warnings
from typing import Optional, Any

import numpy as np


class VolatilityModel:
    """
    Wrapper around QUANT1's SAGEVolModel for realized volatility forecasts.

    Parameters
    ----------
    stub         : bool
        If True, always return default_vol (no model load).
    weights_path : str | None
        Path to the QUANT1 checkpoint (e.g. sage_vol_best.pt).
    min_history  : int
        Minimum number of LOB snapshots required before using the GNN.
    default_vol  : float
        Fallback volatility estimate.
    device       : str
        "cpu" or "cuda".
    """

    def __init__(
        self,
        stub: bool = False,
        weights_path: Optional[str] = None,
        min_history: int = 30,
        default_vol: float = 0.02,
        device: str = "cpu",
    ) -> None:
        self.stub = stub or (weights_path is None)
        self.min_history = min_history
        self.default_vol = default_vol
        self.device = device

        self._model: Any = None
        self._torch = None
        self._y_mean: Optional[float] = None
        self._y_std: Optional[float] = None

        if not self.stub:
            self._load_from_quantis(weights_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, lob_history: list[dict]) -> dict:
        """
        Parameters
        ----------
        lob_history : list of dict
            Each dict should contain at least:
              {"mid": float, "spread": float,
               "bid_depth": float, "ask_depth": float}

        Returns
        -------
        dict
            {"vol_forecast": float, "systemic_score": float}
        """
        if self.stub or len(lob_history) < self.min_history:
            return {"vol_forecast": self.default_vol, "systemic_score": 0.0}

        try:
            x = self._build_node_features(lob_history)
            vol = self._run_model(x)
            return {"vol_forecast": float(vol), "systemic_score": 0.0}
        except Exception as exc:
            warnings.warn(f"VolatilityModel.predict failed: {exc}. Using default.")
            return {"vol_forecast": self.default_vol, "systemic_score": 0.0}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_from_quantis(self, weights_path: str) -> None:
        """
        Load SAGEVolModel and checkpoint from QUANT1.

        We infer in_channels, hidden, n_layers from model_state shapes and
        de-standardisation params from y_mean/y_std.
        """
        try:
            import torch
            from torch import nn
            from quant_optiver.models.gnn_volatility import SAGEVolModel

            self._torch = torch
            ck = torch.load(weights_path, map_location=self.device)

            if not isinstance(ck, dict):
                raise ValueError("Checkpoint is not a dict")

            state = ck.get("model_state") or ck.get("model_state_dict") or ck.get("state_dict") or ck

            # infer architecture from weight shapes
            # W_self.0.weight : (hidden, in_channels)
            w0 = state["W_self.0.weight"]
            hidden = w0.shape[0]
            in_channels = w0.shape[1]
            n_layers = len([k for k in state.keys() if k.startswith("W_self.") and k.endswith(".weight")])

            model = SAGEVolModel(
                in_channels=in_channels,
                hidden=hidden,
                n_layers=n_layers,
                dropout=0.0,
            )
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()

            self._model = model
            self._y_mean = float(ck.get("y_mean", 0.0))
            self._y_std = float(ck.get("y_std", 1.0))

        except Exception as e:
            warnings.warn(f"Could not load QUANT1 SAGEVolModel from '{weights_path}': {e}. Stub mode.")
            self.stub = True

    def _build_node_features(self, lob_history: list[dict]) -> "torch.Tensor":
        """
        Build a (1, F) node feature tensor from LOB history.

        We construct a small feature vector aligned with QUANT1's
        LOB microstructure features (see lob_features.py), keeping
        the dimensionality fixed and letting the GNN handle scaling.

        Features (17-dim):
          0  rv_full       : std of log returns over full window
          1  rv_last       : std of last half of returns
          2  rv_first      : std of first half of returns
          3  rv_ratio      : rv_last / (rv_first + eps)
          4  mid_rv        : last mid * rv_full
          5  spread_mean   : mean spread
          6  spread_std    : std spread
          7  imb_mean      : mean imbalance
          8  imb_std       : std imbalance
          9  depth_mean    : mean (bid+ask depth)
          10 depth_std     : std (bid+ask depth)
          11 bid_mean      : mean bid depth
          12 ask_mean      : mean ask depth
          13 log_mid_last  : log(last mid)
          14 spread_last   : last spread
          15 depth_last    : last depth
          16 imb_last      : last imbalance
        """
        import torch

        mids = np.array([s["mid"] for s in lob_history], dtype=np.float32)
        sprs = np.array([s["spread"] for s in lob_history], dtype=np.float32)
        bds = np.array([s["bid_depth"] for s in lob_history], dtype=np.float32)
        ads = np.array([s["ask_depth"] for s in lob_history], dtype=np.float32)
        deps = bds + ads
        imbs = (bds - ads) / (deps + 1e-9)

        log_ret = np.diff(np.log(np.maximum(mids, 1e-9)))
        if log_ret.size == 0:
            log_ret = np.array([0.0], dtype=np.float32)

        n = log_ret.size
        half = max(n // 2, 1)

        rv_full = float(np.std(log_ret))
        rv_first = float(np.std(log_ret[:half]))
        rv_last = float(np.std(log_ret[half:]))
        rv_ratio = rv_last / (rv_first + 1e-9)
        mid_rv = float(mids[-1]) * rv_full

        feats = np.array([
            rv_full,
            rv_last,
            rv_first,
            rv_ratio,
            mid_rv,
            float(np.mean(sprs)),
            float(np.std(sprs)),
            float(np.mean(imbs)),
            float(np.std(imbs)),
            float(np.mean(deps)),
            float(np.std(deps)),
            float(np.mean(bds)),
            float(np.mean(ads)),
            float(np.log(mids[-1])),
            float(sprs[-1]),
            float(deps[-1]),
            float(imbs[-1]),
        ], dtype=np.float32)

        return torch.tensor(feats, device=self.device).unsqueeze(0)  # (1, F)

    def _run_model(self, x: "torch.Tensor") -> float:
        """
        Run SAGEVolModel on a 1-node graph, de-standardise using y_mean/y_std,
        then clamp to [1e-4, 1.0].
        """
        import torch
        from torch_geometric.data import Data

        if self._model is None or self._torch is None:
            return self.default_vol

        edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
        data = Data(x=x, edge_index=edge_index)

        with self._torch.no_grad():
            raw = self._model(data)[0].item()

        if self._y_mean is not None and self._y_std is not None:
            vol = raw * self._y_std + self._y_mean
        else:
            vol = raw

        # realised vol in QUANT1 is small (e.g. 0.001–0.05)
        vol = float(np.clip(vol, 1e-4, 1.0))
        return vol