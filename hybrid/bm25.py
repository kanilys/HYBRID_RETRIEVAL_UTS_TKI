"""Jalur sparse: BM25 (Bagian 3.3, persamaan (1) dan (2)).

Implementasi "eager scoring" berbasis scipy.sparse (pendekatan yang sama dengan pustaka bm25s):
bobot BM25 tiap pasangan (dokumen, istilah) dihitung sekali saat indexing dan disimpan dalam
matriks jarang W (N dokumen x V istilah). Skor kueri = jumlah kolom W untuk istilah kueri.

    BM25(q,d) = sum_{t in q} IDF(t) * f(t,d)(k1+1) / (f(t,d) + k1(1 - b + b|d|/avgdl))
    IDF(t)    = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))          (varian Lucene)

Catatan: istilah kueri diperlakukan sebagai himpunan (istilah yang berulang dihitung sekali),
dan |d| = jumlah token dokumen setelah penghapusan stopword.
"""
from __future__ import annotations

import json
from array import array
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .data import doc_to_text, iter_corpus
from .preprocess import tokenize


def tokenize_corpus(data_dir: Path, stopwords: frozenset[str]):
    """Tokenisasi korpus SEKALI (tanpa stemming).

    Return: flat_ids (int32, semua token berurutan), offsets (int64, batas tiap dokumen), vocab (list).
    """
    from tqdm import tqdm

    vocab: dict[str, int] = {}
    flat = array("i")
    offsets = array("q", [0])
    for doc in tqdm(iter_corpus(data_dir), desc="tokenisasi korpus", unit="doc"):
        for tok in tokenize(doc_to_text(doc)):
            if tok in stopwords:
                continue
            tid = vocab.get(tok)
            if tid is None:
                tid = vocab[tok] = len(vocab)
            flat.append(tid)
        offsets.append(len(flat))
    return (np.frombuffer(flat, dtype=np.int32).copy(),
            np.frombuffer(offsets, dtype=np.int64).copy(),
            list(vocab))


def remap_vocab(vocab: list[str], new_terms: list[str]) -> tuple[np.ndarray, list[str]]:
    """Petakan id lama -> id baru setelah setiap istilah diubah (mis. di-stem)."""
    new_vocab: dict[str, int] = {}
    mapping = np.empty(len(vocab), dtype=np.int32)
    for old_id, term in enumerate(new_terms):
        nid = new_vocab.get(term)
        if nid is None:
            nid = new_vocab[term] = len(new_vocab)
        mapping[old_id] = nid
    return mapping, list(new_vocab)


class BM25Index:
    def __init__(self, W: sp.csc_matrix, vocab: list[str], k1: float, b: float, meta: dict | None = None):
        self.W = W.tocsc()
        self.vocab = vocab
        self.term_to_id = {t: i for i, t in enumerate(vocab)}
        self.k1, self.b = k1, b
        self.meta = meta or {}

    @property
    def n_docs(self) -> int:
        return self.W.shape[0]

    # ---- pembangunan -----------------------------------------------------------------------
    @classmethod
    def build(cls, flat_ids: np.ndarray, offsets: np.ndarray, vocab: list[str],
              k1: float = 1.2, b: float = 0.75, meta: dict | None = None) -> "BM25Index":
        n_docs, n_terms = len(offsets) - 1, len(vocab)
        lengths = np.diff(offsets)
        doc_idx = np.repeat(np.arange(n_docs, dtype=np.int32), lengths)
        # matriks frekuensi istilah f(t,d); duplikat dijumlahkan saat konversi ke CSR
        tf = sp.csr_matrix((np.ones(len(flat_ids), dtype=np.float32), (doc_idx, flat_ids)),
                           shape=(n_docs, n_terms))
        del doc_idx
        tf.sum_duplicates()

        dl = lengths.astype(np.float32)
        avgdl = float(dl.mean()) if n_docs else 0.0
        df = np.bincount(tf.indices, minlength=n_terms).astype(np.float64)
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

        rows = np.repeat(np.arange(n_docs, dtype=np.int32), np.diff(tf.indptr))
        f = tf.data
        denom = f + k1 * (1.0 - b + b * dl[rows] / max(avgdl, 1e-9))
        tf.data = (idf[tf.indices] * f * (k1 + 1.0) / denom).astype(np.float32)
        meta = dict(meta or {}, n_docs=n_docs, n_terms=n_terms, avgdl=avgdl, k1=k1, b=b)
        return cls(tf.tocsc(), vocab, k1, b, meta)

    # ---- pencarian -------------------------------------------------------------------------
    def scores(self, query_tokens: list[str]) -> np.ndarray:
        ids = sorted({self.term_to_id[t] for t in query_tokens if t in self.term_to_id})
        if not ids:
            return np.zeros(self.n_docs, dtype=np.float32)
        return np.asarray(self.W[:, ids].sum(axis=1)).ravel()

    def search(self, query_tokens: list[str], K: int) -> tuple[np.ndarray, np.ndarray]:
        """K dokumen teratas (skor > 0), urut menurun. Return (indeks dokumen, skor)."""
        s = self.scores(query_tokens)
        return topk(s, K, min_score=0.0)

    # ---- simpan/muat -----------------------------------------------------------------------
    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        sp.save_npz(directory / "W.npz", self.W, compressed=False)
        (directory / "vocab.json").write_text(json.dumps(self.vocab, ensure_ascii=False), encoding="utf-8")
        (directory / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "BM25Index":
        W = sp.load_npz(directory / "W.npz")
        vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        return cls(W, vocab, meta["k1"], meta["b"], meta)


def topk(scores: np.ndarray, K: int, min_score: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Top-K urut menurun; seri dipecah dengan indeks dokumen terkecil (deterministik)."""
    if min_score is not None:
        cand = np.flatnonzero(scores > min_score)
    else:
        cand = np.arange(len(scores))
    if len(cand) > K:
        part = np.argpartition(-scores[cand], K - 1)[:K]
        cand = cand[part]
    order = np.lexsort((cand, -scores[cand]))
    cand = cand[order]
    return cand, scores[cand]
