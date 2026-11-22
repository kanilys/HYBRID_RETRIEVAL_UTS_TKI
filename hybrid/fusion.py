"""Normalisasi dan fusi skor (Bagian 3.5, persamaan (5), (6), (7)).

Input setiap fungsi: daftar kandidat satu jalur sebagai pasangan (indeks dokumen, skor),
urut menurun (peringkat 1 = elemen pertama). Himpunan kandidat gabungan = union kedua daftar.
"""
from __future__ import annotations

from typing import Sequence

Ranked = Sequence[tuple[int, float]]


def minmax(ranked: Ranked) -> dict[int, float]:
    """Persamaan (5). Jika max == min, semua skor ternormalisasi = 0."""
    if not ranked:
        return {}
    scores = [s for _, s in ranked]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return {d: 0.0 for d, _ in ranked}
    return {d: (s - lo) / (hi - lo) for d, s in ranked}


def _sorted(scores: dict[int, float]) -> list[tuple[int, float]]:
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def fuse_linear(bm25: Ranked, dense: Ranked, alpha: float) -> list[tuple[int, float]]:
    """Persamaan (6): alpha * s'_dense + (1 - alpha) * s'_BM25; jalur yang tidak memuat d bernilai 0."""
    nb, nd = minmax(bm25), minmax(dense)
    docs = set(nb) | set(nd)
    return _sorted({d: alpha * nd.get(d, 0.0) + (1 - alpha) * nb.get(d, 0.0) for d in docs})


def fuse_rrf(bm25: Ranked, dense: Ranked, k: float = 60) -> list[tuple[int, float]]:
    """Persamaan (7): jumlah 1/(k + rank) dari daftar yang memuat d (rank mulai dari 1)."""
    out: dict[int, float] = {}
    for ranked in (bm25, dense):
        for rank, (d, _) in enumerate(ranked, start=1):
            out[d] = out.get(d, 0.0) + 1.0 / (k + rank)
    return _sorted(out)


def fuse(mode: str, bm25: Ranked, dense: Ranked, alpha: float, rrf_k: float):
    if mode == "linear":
        return fuse_linear(bm25, dense, alpha)
    if mode == "rrf":
        return fuse_rrf(bm25, dense, rrf_k)
    raise ValueError(mode)
