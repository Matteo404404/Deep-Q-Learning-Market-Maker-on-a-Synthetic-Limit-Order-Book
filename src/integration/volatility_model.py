"""
volatility_model.py
===================
Adapter around QUANT1's GraphSAGE volatility model for use in the RL env.
"""
from __future__ import annotations
import warnings
from typing import Optional, Any
import numpy as np


class VolatilityModel:
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

    def predict(self, lob_history: list[dict]) -> dict:
        if self.stub or len(lob_history) < self.min_history:
            return {"vol_forecast": self.default_vol, "systemic_score": 0.0}
        try:
            x = self._build_node_features(lob_history)
            vol = self._run_model(x)
            return {"vol_forecast": float(vol), "systemic_score": 0.0}
        except Exception as exc:
            warnings.warn(f"VolatilityModel.predict failed: {exc}. Using default.")
            return {"vol_forecast": self.default_vol, "systemic_score": 0.0}

    def _load_from_quantis(self, weights_path: str) -> None:
        try:
            import torch
            from quant_optiver.models.gnn_volatility import SAGEVolModel
            self._torch = torch
            ck = torch.load(weights_path, map_location=self.device)
            if not isinstance(ck, dict):
                raise ValueError("Checkpoint is not a dict")
            state = ck.get("model_state") or ck.get("model_state_dict") or ck.get("state_dict") or ck
            w0 = state["W_self.0.weight"]
            hidden = w0.shape[0]
            in_channels = w0.shape[1]
            n_layers = len([k for k in state.keys() if k.startswith("W_self.") and k.endswith(".weight")])
            model = SAGEVolModel(in_channels=in_channels, hidden=hidden, n_layers=n_layers, dropout=0.0)
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            self._model = model
            self._y_mean = float(ck.get("y_mean", 0.0))
            self._y_std  = float(ck.get("y_std",  1.0))
        except Exception as e:
            warnings.warn(f"Could not load SAGEVolModel from '{weights_path}': {e}. Stub mode.")
            self.stub = True

    def _build_node_features(self, lob_history: list[dict]) -> "torch.Tensor":
        import torch
        mids = np.array([s["mid"]       for s in lob_history], dtype=np.float32)
        sprs = np.array([s["spread"]    for s in lob_history], dtype=np.float32)
        bds  = np.array([s["bid_depth"] for s in lob_history], dtype=np.float32)
        ads  = np.array([s["ask_depth"] for s in lob_history], dtype=np.float32)
        deps = bds + ads
        imbs = (bds - ads) / (deps + 1e-9)
        log_ret = np.diff(np.log(np.maximum(mids, 1e-9)))
        if log_ret.size == 0:
            log_ret = np.array([0.0], dtype=np.float32)
        n    = log_ret.size
        half = max(n // 2, 1)
        rv_full  = float(np.std(log_ret))
        rv_first = float(np.std(log_ret[:half]))
        rv_last  = float(np.std(log_ret[half:]))
        rv_ratio = rv_last / (rv_first + 1e-9)
        feats = np.array([
            rv_full, rv_last, rv_first, rv_ratio,
            float(mids[-1]) * rv_full,
            float(np.mean(sprs)), float(np.std(sprs)),
            float(np.mean(imbs)), float(np.std(imbs)),
            float(np.mean(deps)), float(np.std(deps)),
            float(np.mean(bds)),  float(np.mean(ads)),
            float(np.log(mids[-1])),
            float(sprs[-1]), float(deps[-1]), float(imbs[-1]),
        ], dtype=np.float32)
        return torch.tensor(feats, device=self.device).unsqueeze(0)

    def _run_model(self, x: "torch.Tensor") -> float:
        import torch
        from torch_geometric.data import Data
        if self._model is None:
            return self.default_vol
        edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
        data = Data(x=x, edge_index=edge_index)
        with self._torch.no_grad():
            raw = self._model(data)[0].item()
        vol = raw * self._y_std + self._y_mean if self._y_mean is not None else raw
        return float(np.clip(vol, 1e-4, 1.0))
