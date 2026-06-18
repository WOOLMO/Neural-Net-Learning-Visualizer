from __future__ import annotations

import numpy as np


def make_dataset(name: str, n_samples: int = 240, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    name = name.lower()

    if name == "moons":
        angles = rng.uniform(0, np.pi, n_samples // 2)
        upper = np.c_[np.cos(angles), np.sin(angles)]
        lower = np.c_[1 - np.cos(angles), 0.5 - np.sin(angles)]
        x = np.vstack([upper, lower])
        y = np.r_[np.zeros(len(upper)), np.ones(len(lower))]
        x += rng.normal(0, 0.09, x.shape)
    elif name == "spiral":
        half = n_samples // 2
        t = np.sqrt(rng.uniform(0, 1, half)) * 3.8
        first = np.c_[np.cos(t) * t, np.sin(t) * t]
        second = np.c_[np.cos(t + np.pi) * t, np.sin(t + np.pi) * t]
        x = np.vstack([first, second])
        y = np.r_[np.zeros(half), np.ones(half)]
        x += rng.normal(0, 0.18, x.shape)
    elif name == "xor":
        x = rng.uniform(-1.2, 1.2, (n_samples, 2))
        y = ((x[:, 0] * x[:, 1]) > 0).astype(float)
        x += rng.normal(0, 0.05, x.shape)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    x = (x - x.mean(axis=0)) / x.std(axis=0)
    return x.astype(np.float64), y.reshape(-1, 1).astype(np.float64)
