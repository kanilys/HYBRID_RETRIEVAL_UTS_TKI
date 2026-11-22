"""Prapemrosesan teks (Bagian 3.2).

- clean_text      : pembersihan umum untuk KEDUA jalur (normalisasi Unicode, buang markup, rapikan spasi)
- tokenize        : case folding + tokenisasi (khusus jalur BM25)
- stopword        : daftar stopword PySastrawi (khusus jalur BM25)
- CachedStemmer   : stemmer PySastrawi dengan cache per kata unik (khusus jalur BM25, opsional)
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)  # huruf/angka; tanda baca dan underscore dibuang


def clean_text(text: str | None) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = _TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Case folding + tokenisasi. Diasumsikan teks sudah melalui clean_text."""
    return _TOKEN_RE.findall(text.casefold())


def load_stopwords(path: str | Path | None = None) -> frozenset[str]:
    """Stopword PySastrawi, atau file teks (satu kata per baris) bila path diberikan."""
    if path:
        words = Path(path).read_text(encoding="utf-8").split()
    else:
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        words = StopWordRemoverFactory().get_stop_words()
    return frozenset(w.casefold() for w in words)


def _should_stem(token: str) -> bool:
    # Token berisi angka (tahun, kode) dibiarkan apa adanya.
    return token.isalpha()


class CachedStemmer:
    """Stemmer PySastrawi. Setiap kata unik hanya di-stem sekali."""

    def __init__(self):
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        self._stemmer = StemmerFactory().create_stemmer()
        self.cache: dict[str, str] = {}

    def stem(self, token: str) -> str:
        out = self.cache.get(token)
        if out is None:
            out = (self._stemmer.stem(token) if _should_stem(token) else token) or token
            self.cache[token] = out
        return out


# --- stemming paralel untuk kosakata korpus (dipakai saat indexing) ---------------------
_worker_stemmer = None


def _init_worker():
    global _worker_stemmer
    _worker_stemmer = CachedStemmer()


def _stem_chunk(tokens: list[str]) -> list[str]:
    return [_worker_stemmer.stem(t) for t in tokens]


def stem_vocabulary(vocab: list[str], workers: int = 1, chunk: int = 20000) -> list[str]:
    """Stem seluruh kosakata unik (jauh lebih cepat daripada stem per token korpus)."""
    from tqdm import tqdm

    chunks = [vocab[i:i + chunk] for i in range(0, len(vocab), chunk)]
    out: list[str] = []
    if workers <= 1:
        _init_worker()
        for c in tqdm(chunks, desc="stemming kosakata", unit="chunk"):
            out.extend(_stem_chunk(c))
        return out
    from multiprocessing import Pool
    with Pool(workers, initializer=_init_worker) as pool:
        for res in tqdm(pool.imap(_stem_chunk, chunks), total=len(chunks),
                        desc=f"stemming kosakata ({workers} proses)", unit="chunk"):
            out.extend(res)
    return out


class BM25Analyzer:
    """Analyzer kueri untuk jalur BM25; identik dengan yang dipakai saat indexing."""

    def __init__(self, stopwords: frozenset[str], stemmer: CachedStemmer | None = None):
        self.stopwords = stopwords
        self.stemmer = stemmer

    def __call__(self, text: str) -> list[str]:
        toks = [t for t in tokenize(clean_text(text)) if t not in self.stopwords]
        if self.stemmer is not None:
            toks = [self.stemmer.stem(t) for t in toks]
        return toks
