"""Uji statistik berpasangan (Bagian 3.7, persamaan (14)).

- paired_bootstrap_ci   : interval kepercayaan untuk Delta = rata-rata selisih metrik per kueri
- paired_randomization  : nilai-p dua sisi (uji permutasi berpasangan / sign-flip)
- holm                  : koreksi perbandingan berganda Holm-Bonferroni
"""
from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_resamples: int = 10000,
                        confidence: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        means[i] = d[rng.integers(0, len(d), len(d))].mean()
    lo, hi = np.quantile(means, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
    return float(d.mean()), float(lo), float(hi)


def paired_randomization(a: np.ndarray, b: np.ndarray, n_resamples: int = 10000,
                         seed: int = 42) -> float:
    """H0: sistem A dan B dapat dipertukarkan per kueri. Statistik uji: |mean(d)|."""
    d = np.asarray(a, float) - np.asarray(b, float)
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_resamples):
        signs = rng.choice((-1.0, 1.0), size=len(d))
        if abs((signs * d).mean()) >= observed - 1e-12:
            count += 1
    return (count + 1) / (n_resamples + 1)


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist()
