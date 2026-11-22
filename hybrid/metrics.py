"""Metrik evaluasi (Bagian 3.7, persamaan (8)-(13)).

Konvensi (mengikuti trec_eval):
- Dokumen relevan = rel > 0 pada qrels. Dokumen yang tidak ada di qrels dianggap tidak relevan.
- AP dinormalisasi dengan |Rel| = jumlah dokumen relevan pada qrels (bukan yang ditemukan).
- Kueri tanpa dokumen relevan di qrels tidak dievaluasi.
- Gain nDCG = 2^rel - 1 (persamaan (8)). Pada MIRACL penilaian bersifat biner, sehingga
  hasilnya identik dengan gain linear yang dipakai trec_eval.
"""
from __future__ import annotations

import math
from typing import Sequence

import pandas as pd


def ndcg_at_k(ranked: Sequence[str], qrel: dict[str, int], k: int) -> float:
    dcg = sum((2 ** qrel.get(d, 0) - 1) / math.log2(i + 2)
              for i, d in enumerate(ranked[:k]) if qrel.get(d, 0) > 0)
    ideal = sorted((r for r in qrel.values() if r > 0), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(ranked: Sequence[str], qrel: dict[str, int], k: int) -> float:
    return sum(1 for d in ranked[:k] if qrel.get(d, 0) > 0) / k


def recall_at_k(ranked: Sequence[str], qrel: dict[str, int], k: int) -> float:
    n_rel = sum(1 for r in qrel.values() if r > 0)
    return sum(1 for d in ranked[:k] if qrel.get(d, 0) > 0) / n_rel if n_rel else 0.0


def ap_at_k(ranked: Sequence[str], qrel: dict[str, int], k: int) -> float:
    n_rel = sum(1 for r in qrel.values() if r > 0)
    if not n_rel:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(ranked[:k], start=1):
        if qrel.get(d, 0) > 0:
            hits += 1
            total += hits / i
    return total / n_rel


METRICS = {
    "nDCG@10": lambda r, q: ndcg_at_k(r, q, 10),
    "MAP@100": lambda r, q: ap_at_k(r, q, 100),
    "Recall@100": lambda r, q: recall_at_k(r, q, 100),
    "P@10": lambda r, q: precision_at_k(r, q, 10),
}


def evaluate_run(run: dict[str, list[str]], qrels: dict[str, dict[str, int]],
                 qids: Sequence[str]) -> pd.DataFrame:
    """Metrik per kueri. Kolom: qid + setiap metrik. Rata-ratanya = MAP, mean nDCG, dst."""
    rows = []
    for qid in qids:
        qrel = qrels.get(qid, {})
        if not any(r > 0 for r in qrel.values()):
            continue
        ranked = run.get(qid, [])
        rows.append({"qid": qid, **{m: f(ranked, qrel) for m, f in METRICS.items()}})
    return pd.DataFrame(rows).set_index("qid")
