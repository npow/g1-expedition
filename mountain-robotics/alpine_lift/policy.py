"""Numpy inference for the trained residual policy.

Training happens on a GPU box with PyTorch; the demo runs on a laptop that
does not need to have PyTorch installed at all. The trainer exports plain
arrays, and this file is the whole runtime -- three matrix multiplies and a
tanh. It also means the live demo has no framework that can fail to import
five minutes before a pitch.
"""

from __future__ import annotations

import os

import numpy as np


class ResidualPolicy:
    """Deterministic MLP policy: normalised observation -> tanh action."""

    def __init__(self, weights: dict):
        self.w = [weights[f"w{i}"] for i in range(weights["n_layers"].item())]
        self.b = [weights[f"b{i}"] for i in range(weights["n_layers"].item())]
        self.obs_mean = weights["obs_mean"]
        self.obs_std = np.maximum(weights["obs_std"], 1e-6)
        self.obs_dim = int(self.obs_mean.shape[0])
        self.act_dim = int(self.b[-1].shape[0])
        self.meta = {
            k: weights[k] for k in weights.files
            if k not in {"obs_mean", "obs_std", "n_layers"}
            and not k.startswith(("w", "b"))
        }

    @classmethod
    def load(cls, path: str) -> "ResidualPolicy":
        return cls(np.load(path))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = (np.asarray(obs, dtype=np.float64) - self.obs_mean) / self.obs_std
        x = np.clip(x, -10.0, 10.0)
        for i, (w, b) in enumerate(zip(self.w, self.b)):
            x = x @ w.T + b
            if i < len(self.w) - 1:
                x = np.tanh(x)
        return np.tanh(x)


def load_if_present(path: str | None) -> ResidualPolicy | None:
    if path and os.path.exists(path):
        return ResidualPolicy.load(path)
    return None
