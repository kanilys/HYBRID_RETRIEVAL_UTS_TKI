"""Uji unit untuk komponen yang dapat diverifikasi tanpa model/GPU.

Jalankan:  python -m pytest tests -q
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hybrid.bm25 import BM25Index, remap_vocab, topk  # noqa: E402
from hybrid.dense import DenseIndex  # noqa: E402
from hybrid.fusion import fuse_linear, fuse_rrf, minmax  # noqa: E402
from hybrid.metrics import ap_at_k, ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402
from hybrid.stats import holm, paired_bootstrap_ci, paired_randomization  # noqa: E402

DOCS = [["mobil", "listrik", "baterai"],
        ["kendaraan", "listrik", "listrik", "kota"],
        ["sepeda", "kota"],
        ["mobil", "bensin", "mesin", "mobil", "jalan"]]


def build_toy(k1=1.2, b=0.75):
    vocab = sorted({t for d in DOCS for t in d})
    tid = {t: i for i, t in enumerate(vocab)}
    flat = np.array([tid[t] for d in DOCS for t in d], dtype=np.int32)
    offsets = np.cumsum([0] + [len(d) for d in DOCS]).astype(np.int64)
    return BM25Index.build(flat, offsets, vocab, k1, b)


def manual_bm25(query, doc, k1=1.2, b=0.75):
    N = len(DOCS)
    avgdl = sum(map(len, DOCS)) / N
    score = 0.0
    for t in set(query):
        n_t = sum(t in d for d in DOCS)
        f = doc.count(t)
        if not f:
            continue
        idf = math.log(1 + (N - n_t + 0.5) / (n_t + 0.5))
        score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(doc) / avgdl))
    return score


def test_bm25_matches_formula():
    idx = build_toy()
    q = ["mobil", "listrik", "tidakada"]
    np.testing.assert_allclose(idx.scores(q), [manual_bm25(q, d) for d in DOCS], rtol=1e-5)


def test_bm25_search_excludes_zero_and_sorts():
    ids, scores = build_toy().search(["listrik"], K=10)
    assert set(ids.tolist()) == {0, 1}
    assert scores[0] >= scores[1]


def test_bm25_save_load(tmp_path):
    idx = build_toy()
    idx.save(tmp_path)
    again = BM25Index.load(tmp_path)
    np.testing.assert_allclose(again.scores(["kota"]), idx.scores(["kota"]))


def test_remap_vocab_merges_terms():
    mapping, new_vocab = remap_vocab(["mencari", "cari", "pencarian"], ["cari", "cari", "cari"])
    assert new_vocab == ["cari"] and mapping.tolist() == [0, 0, 0]


def test_topk_tiebreak_deterministic():
    ids, _ = topk(np.array([1.0, 3.0, 3.0, 2.0]), 3)
    assert ids.tolist() == [1, 2, 3]


def test_dense_blockwise_equals_bruteforce():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(1000, 16)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    q = emb[:3] + 0.01
    ids, s = DenseIndex(emb, block_size=97).search(q, 25)
    full = q @ emb.T
    for i in range(3):
        expected = np.argsort(-full[i], kind="stable")[:25]
        assert ids[i].tolist() == expected.tolist()


def test_minmax_constant_scores_zero():
    assert minmax([(1, 5.0), (2, 5.0)]) == {1: 0.0, 2: 0.0}


def test_linear_fusion_missing_path_is_zero():
    bm25 = [(1, 10.0), (2, 5.0), (3, 0.0)]
    dense = [(2, 0.9), (4, 0.5)]
    out = dict(fuse_linear(bm25, dense, alpha=0.5))
    assert out[1] == pytest.approx(0.5 * 0 + 0.5 * 1.0)
    assert out[2] == pytest.approx(0.5 * 1.0 + 0.5 * 0.5)
    assert out[4] == pytest.approx(0.0)


def test_rrf():
    out = dict(fuse_rrf([(1, 9), (2, 8)], [(2, 0.9), (3, 0.1)], k=60))
    assert out[2] == pytest.approx(1 / 62 + 1 / 61)
    assert out[1] == pytest.approx(1 / 61)
    assert out[3] == pytest.approx(1 / 62)


def test_metrics_known_values():
    qrel = {"a": 1, "b": 1, "c": 0}
    ranked = ["x", "a", "c", "b"]
    assert precision_at_k(ranked, qrel, 2) == 0.5
    assert recall_at_k(ranked, qrel, 2) == 0.5
    assert ap_at_k(ranked, qrel, 100) == pytest.approx((1 / 2 + 2 / 4) / 2)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3)
    assert ndcg_at_k(ranked, qrel, 10) == pytest.approx(dcg / idcg)
    # AP dinormalisasi dengan |Rel| di qrels, bukan yang ditemukan
    assert ap_at_k(["a"], qrel, 100) == pytest.approx(0.5)


def test_stats():
    a = np.array([0.5, 0.6, 0.7, 0.8] * 10)
    delta, lo, hi = paired_bootstrap_ci(a, a, 500)
    assert delta == 0 and lo == 0 and hi == 0
    assert paired_randomization(a + 0.1, a, 500) < 0.01
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_dense_gpu_equals_cpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("tanpa GPU")
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(5000, 32)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb.astype(np.float16)
    q = emb[:4].astype(np.float32)
    ci, cs = DenseIndex(emb, block_size=999).search(q, 50)
    gi, gs = DenseIndex(emb, device="cuda")._search_gpu(q, 50, block=1234)
    np.testing.assert_allclose(cs, gs, atol=1e-5)
    assert (ci == gi).mean() > 0.98  # hanya boleh beda pada skor yang seri/hampir seri
