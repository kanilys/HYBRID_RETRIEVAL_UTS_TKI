"""Pemuatan data dalam format sederhana yang netral terhadap dataset.

Struktur folder data:
    <data_dir>/corpus/*.jsonl atau *.jsonl.gz   -> {"docid", "title", "text"} per baris
    <data_dir>/queries_<split>.tsv              -> qid \t teks kueri
    <data_dir>/qrels_<split>.tsv                -> qid Q0 docid rel   (format TREC)

Urutan dokumen (indeks internal 0..N-1) = urutan baris pada file korpus yang diurutkan namanya.
Semua komponen (BM25, dense, docstore) memakai urutan yang sama.
"""
from __future__ import annotations

import gzip
import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from .preprocess import clean_text


def corpus_files(data_dir: Path) -> list[Path]:
    files = sorted((Path(data_dir) / "corpus").glob("*.jsonl*"))
    if not files:
        raise FileNotFoundError(f"Tidak ada file korpus di {Path(data_dir) / 'corpus'}")
    return files


def iter_corpus(data_dir: Path) -> Iterator[dict]:
    for f in corpus_files(data_dir):
        opener = gzip.open if f.suffix == ".gz" else open
        with opener(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def doc_to_text(doc: dict) -> str:
    """Teks dokumen yang dipakai kedua jalur: judul + isi, setelah pembersihan umum."""
    title = clean_text(doc.get("title"))
    text = clean_text(doc.get("text"))
    return f"{title} {text}".strip() if title else text


def load_queries(path: Path) -> dict[str, str]:
    queries = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                queries[parts[0]] = parts[1]
    return queries


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 4:
                qid, _, docid, rel = parts
                qrels[qid][docid] = int(rel)
    return dict(qrels)


def split_qids(qids: list[str], fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Bagi kueri menjadi (tune, test) secara acak tetapi deterministik."""
    qids = sorted(qids)
    rng = random.Random(seed)
    rng.shuffle(qids)
    n_tune = int(round(len(qids) * fraction))
    return sorted(qids[:n_tune]), sorted(qids[n_tune:])


# --- Docstore (untuk menampilkan judul/potongan teks di antarmuka) -------------------------
class DocStore:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)

    @staticmethod
    def build(data_dir: Path, path: Path) -> list[str]:
        from tqdm import tqdm

        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        conn = sqlite3.connect(str(path))
        conn.execute("CREATE TABLE docs (idx INTEGER PRIMARY KEY, docid TEXT, title TEXT, text TEXT)")
        docids, batch = [], []
        for i, doc in enumerate(tqdm(iter_corpus(data_dir), desc="docstore", unit="doc")):
            docids.append(doc["docid"])
            batch.append((i, doc["docid"], doc.get("title", ""), doc.get("text", "")))
            if len(batch) >= 50000:
                conn.executemany("INSERT INTO docs VALUES (?,?,?,?)", batch)
                batch.clear()
        if batch:
            conn.executemany("INSERT INTO docs VALUES (?,?,?,?)", batch)
        conn.commit()
        conn.close()
        return docids

    def get(self, idxs: list[int]) -> dict[int, dict]:
        if not idxs:
            return {}
        q = f"SELECT idx, docid, title, text FROM docs WHERE idx IN ({','.join('?' * len(idxs))})"
        return {r[0]: {"docid": r[1], "title": r[2], "text": r[3]}
                for r in self.conn.execute(q, [int(i) for i in idxs])}
