"""Jalur dense: multilingual-E5 tanpa fine-tuning (Bagian 3.4, persamaan (3) dan (4)).

- Awalan "query: " untuk kueri dan "passage: " untuk dokumen (sesuai kartu model multilingual-e5-base)
- Average pooling dengan attention mask, lalu normalisasi L2  -> kosinus = hasil kali dalam
- Teks lebih dari 512 token dipotong (truncation) oleh tokenizer.
- Embedding korpus disimpan sebagai float16 (N x 768) untuk menghemat memori.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


class E5Encoder:
    def __init__(self, model_name: str, device: str | None = None, max_seq_length: int = 512,
                 fp16: bool = True):
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_seq_length
        if fp16 and device.startswith("cuda"):
            self.model.half()
        self._check_pooling()

    def _check_pooling(self):
        """Pastikan konfigurasi pooling = mean, sesuai persamaan (3)."""
        from sentence_transformers.models import Pooling
        for module in self.model:
            if isinstance(module, Pooling):
                cfg = module.get_config_dict()
                is_mean = cfg.get("pooling_mode") == "mean" or cfg.get("pooling_mode_mean_tokens") is True
                if not is_mean:
                    raise ValueError(f"Pooling {self.model_name} bukan mean pooling: {cfg}")

    def _encode(self, texts: list[str], batch_size: int, show_progress: bool) -> np.ndarray:
        emb = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=show_progress)
        return emb.astype(np.float32)

    def encode_queries(self, queries: list[str], batch_size: int = 64, show_progress: bool = False):
        return self._encode([QUERY_PREFIX + q for q in queries], batch_size, show_progress)

    def encode_passages(self, passages: list[str], batch_size: int = 128, show_progress: bool = False):
        return self._encode([PASSAGE_PREFIX + p for p in passages], batch_size, show_progress)


def encode_corpus(encoder: E5Encoder, texts_iter, out_dir: Path, part_size: int = 50000,
                  batch_size: int = 128) -> Path:
    """Encode korpus per bagian (bisa dilanjutkan bila proses terputus), lalu gabungkan."""
    from tqdm import tqdm

    parts_dir = out_dir / "emb_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    buf, part, total = [], 0, 0

    def flush(buf, part):
        f = parts_dir / f"part_{part:05d}.npy"
        if not f.exists():  # bagian yang sudah ada dilewati (resume)
            # urutkan berdasarkan panjang agar batching efisien, lalu kembalikan urutannya
            order = np.argsort([len(t) for t in buf])[::-1]
            emb = encoder.encode_passages([buf[i] for i in order], batch_size=batch_size)
            out = np.empty_like(emb)
            out[order] = emb
            np.save(f, out.astype(np.float16))

    for text in tqdm(texts_iter, desc="encode korpus", unit="doc"):
        buf.append(text)
        if len(buf) == part_size:
            flush(buf, part)
            total += len(buf)
            buf, part = [], part + 1
    if buf:
        flush(buf, part)
        total += len(buf)
        part += 1

    parts = [parts_dir / f"part_{i:05d}.npy" for i in range(part)]
    dim = np.load(parts[0], mmap_mode="r").shape[1]
    final = out_dir / "embeddings.npy"
    mm = np.lib.format.open_memmap(final, mode="w+", dtype=np.float16, shape=(total, dim))
    pos = 0
    for p in parts:
        a = np.load(p)
        mm[pos:pos + len(a)] = a
        pos += len(a)
    mm.flush()
    del mm
    (out_dir / "meta.json").write_text(json.dumps(
        {"model": encoder.model_name, "n_docs": total, "dim": int(dim), "dtype": "float16",
         "query_prefix": QUERY_PREFIX, "passage_prefix": PASSAGE_PREFIX,
         "max_seq_length": encoder.model.max_seq_length, "pooling": "mean", "normalize": "L2"},
        indent=2), encoding="utf-8")
    return final


class DenseIndex:
    """Pencarian eksak (brute force) dengan perkalian matriks per bagian (NumPy)."""

    def __init__(self, embeddings: np.ndarray, block_size: int = 65536, device: str | None = None):
        self.emb = embeddings
        self.block_size = block_size
        self.gpu_emb = None
        if device and device.startswith("cuda"):
            import torch
            # disimpan float16 di GPU (±2.2 GB); dihitung float32 per blok agar presisinya sama dengan CPU
            self.gpu_emb = torch.from_numpy(np.ascontiguousarray(embeddings)).to(device)

    @classmethod
    def load(cls, directory: Path, in_memory: bool = True, device: str | None = None) -> "DenseIndex":
        emb = np.load(directory / "embeddings.npy", mmap_mode=None if in_memory or device else "r")
        return cls(emb, device=device)

    @property
    def n_docs(self) -> int:
        return self.emb.shape[0]

    def _search_gpu(self, q_emb: np.ndarray, K: int, block: int = 262144):
        import torch
        q = torch.from_numpy(np.atleast_2d(q_emb).astype(np.float32)).to(self.gpu_emb.device)
        K = min(K, self.n_docs)
        best_s = best_i = None
        for start in range(0, self.n_docs, block):
            s = q @ self.gpu_emb[start:start + block].float().T
            v, i = torch.topk(s, min(K, s.shape[1]), dim=1)
            i = i + start
            if best_s is None:
                best_s, best_i = v, i
            else:
                cs, ci = torch.cat([best_s, v], 1), torch.cat([best_i, i], 1)
                best_s, idx = torch.topk(cs, K, dim=1)
                best_i = torch.gather(ci, 1, idx)
        s, i = best_s.cpu().numpy(), best_i.cpu().numpy()
        order = np.lexsort((i, -s), axis=1)
        return np.take_along_axis(i, order, 1), np.take_along_axis(s, order, 1)

    def search(self, q_emb: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
        """q_emb: (B, d) ternormalisasi. Return (indeks B x K, skor B x K), urut menurun."""
        if self.gpu_emb is not None:
            return self._search_gpu(q_emb, K)
        q = np.atleast_2d(q_emb).astype(np.float32)
        B = q.shape[0]
        K = min(K, self.n_docs)
        best_s = np.full((B, 0), -np.inf, dtype=np.float32)
        best_i = np.zeros((B, 0), dtype=np.int64)
        for start in range(0, self.n_docs, self.block_size):
            block = np.asarray(self.emb[start:start + self.block_size], dtype=np.float32)
            s = q @ block.T                                       # persamaan (4): q^T d
            k = min(K, s.shape[1])
            idx = np.argpartition(-s, k - 1, axis=1)[:, :k]
            cand_s = np.concatenate([best_s, np.take_along_axis(s, idx, 1)], axis=1)
            cand_i = np.concatenate([best_i, idx + start], axis=1)
            if cand_s.shape[1] > K:
                keep = np.argpartition(-cand_s, K - 1, axis=1)[:, :K]
                cand_s = np.take_along_axis(cand_s, keep, 1)
                cand_i = np.take_along_axis(cand_i, keep, 1)
            best_s, best_i = cand_s, cand_i
        order = np.lexsort((best_i, -best_s), axis=1)
        return np.take_along_axis(best_i, order, 1), np.take_along_axis(best_s, order, 1)
