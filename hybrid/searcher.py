"""Penggabungan komponen menjadi satu mesin pencari (Algoritma 2)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import config
from .bm25 import BM25Index
from .dense import DenseIndex, E5Encoder
from .fusion import fuse
from .preprocess import BM25Analyzer, CachedStemmer, clean_text, load_stopwords

MODES = ("bm25", "dense", "rrf", "linear")


class HybridSearcher:
    def __init__(self, index_dir: Path, device: str | None = None, load_dense: bool = True,
                 dense_in_memory: bool = True, stopwords_path: str | None = None,
                 dense_subdir: str = "dense"):
        index_dir = Path(index_dir)
        self.docids: list[str] = json.loads((index_dir / "docids.json").read_text(encoding="utf-8"))
        stopwords = load_stopwords(stopwords_path)
        self.bm25 = {False: BM25Index.load(index_dir / "bm25_nostem"),
                     True: BM25Index.load(index_dir / "bm25_stem")}
        self.analyzer = {False: BM25Analyzer(stopwords), True: BM25Analyzer(stopwords, CachedStemmer())}
        self.dense = self.encoder = None
        if load_dense:
            meta = json.loads((index_dir / dense_subdir / "meta.json").read_text(encoding="utf-8"))
            self.encoder = E5Encoder(meta["model"], device=device)
            gpu = self.encoder.device if self.encoder.device.startswith("cuda") else None
            self.dense = DenseIndex.load(index_dir / dense_subdir, in_memory=dense_in_memory, device=gpu)
        for name, n in [("bm25_stem", self.bm25[True].n_docs),
                        ("dense", self.dense.n_docs if self.dense else len(self.docids))]:
            if n != len(self.docids):
                raise ValueError(f"Jumlah dokumen {name} ({n}) != docstore ({len(self.docids)})")

    # ---- kandidat per jalur ----------------------------------------------------------------
    def bm25_candidates(self, query: str, stem: bool, K: int) -> list[tuple[int, float]]:
        idx, s = self.bm25[stem].search(self.analyzer[stem](query), K)
        return list(zip(idx.tolist(), s.tolist()))

    def dense_candidates_batch(self, queries: list[str], K: int, batch_size: int = 64):
        q = self.encoder.encode_queries([clean_text(x) for x in queries], batch_size=batch_size)
        out = []
        for start in range(0, len(q), 256):
            idx, s = self.dense.search(q[start:start + 256], K)
            out.extend(list(zip(i.tolist(), v.tolist())) for i, v in zip(idx, s))
        return out

    # ---- pencarian interaktif (dipakai Streamlit) ------------------------------------------
    def search(self, query: str, mode: str = "rrf", stem: bool = False, alpha: float = config.ALPHA,
               rrf_k: float = config.RRF_K, K: int = config.K_CANDIDATES, top_k: int = 10) -> dict:
        timings = {}
        bm25 = dense = None
        if mode in ("bm25", "rrf", "linear"):
            t = time.perf_counter()
            bm25 = self.bm25_candidates(query, stem, max(K, top_k))
            timings["bm25_ms"] = (time.perf_counter() - t) * 1000
        if mode in ("dense", "rrf", "linear"):
            t = time.perf_counter()
            dense = self.dense_candidates_batch([query], max(K, top_k))[0]
            timings["dense_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        if mode == "bm25":
            final = bm25
        elif mode == "dense":
            final = dense
        else:
            final = fuse(mode, bm25[:K], dense[:K], alpha, rrf_k)
        timings["fusion_ms"] = (time.perf_counter() - t) * 1000

        def rank_map(lst):
            return {d: (r, s) for r, (d, s) in enumerate(lst or [], start=1)}

        rb, rd = rank_map(bm25), rank_map(dense)
        results = []
        for rank, (d, score) in enumerate(final[:top_k], start=1):
            results.append({"rank": rank, "idx": d, "docid": self.docids[d], "score": score,
                            "bm25_rank": rb.get(d, (None, None))[0], "bm25_score": rb.get(d, (None, None))[1],
                            "dense_rank": rd.get(d, (None, None))[0], "dense_score": rd.get(d, (None, None))[1]})
        return {"results": results, "timings": timings,
                "query_tokens": self.analyzer[stem](query) if bm25 is not None else None}
