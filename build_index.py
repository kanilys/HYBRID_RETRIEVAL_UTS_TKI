"""Fase 1 - Pengindeksan luring (Algoritma 1).

Contoh:
    python build_index.py                                     # semua langkah
    python build_index.py --steps bm25                        # hanya BM25 (CPU)
    python build_index.py --steps dense --device cuda         # hanya embedding (GPU)
    python build_index.py --data-dir data/miracl-id-sub50k --index-dir indexes/miracl-id-sub50k

Hasil di <index-dir>:
    docids.json, docstore.sqlite     pemetaan indeks -> docid, teks untuk ditampilkan
    bm25_nostem/, bm25_stem/          indeks BM25 untuk tiap opsi stemming
    dense/embeddings.npy              matriks embedding float16 (N x 768) + meta.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid import config  # noqa: E402
from hybrid.bm25 import BM25Index, remap_vocab, tokenize_corpus  # noqa: E402
from hybrid.data import DocStore, doc_to_text, iter_corpus  # noqa: E402
from hybrid.preprocess import load_stopwords, stem_vocabulary  # noqa: E402


def step_docstore(args):
    t = time.time()
    docids = DocStore.build(args.data_dir, args.index_dir / "docstore.sqlite")
    (args.index_dir / "docids.json").write_text(json.dumps(docids), encoding="utf-8")
    print(f"[docstore] {len(docids)} dokumen, {time.time() - t:.0f} detik")


def step_bm25(args):
    stopwords = load_stopwords(args.stopwords)
    t = time.time()
    flat, offsets, vocab = tokenize_corpus(args.data_dir, stopwords)
    print(f"[bm25] {len(offsets) - 1} dokumen, {len(flat)} token, {len(vocab)} istilah unik "
          f"({time.time() - t:.0f} detik)")
    common = {"stopwords": args.stopwords or "PySastrawi", "n_stopwords": len(stopwords),
              "idf": "ln(1 + (N - n_t + 0.5)/(n_t + 0.5))"}

    t = time.time()
    BM25Index.build(flat, offsets, vocab, args.k1, args.b, dict(common, stem=False)).save(
        args.index_dir / "bm25_nostem")
    print(f"[bm25] indeks tanpa stemming tersimpan ({time.time() - t:.0f} detik)")

    t = time.time()
    stems = stem_vocabulary(vocab, workers=args.workers)
    mapping, stem_vocab = remap_vocab(vocab, stems)
    print(f"[bm25] stemming: {len(vocab)} -> {len(stem_vocab)} istilah ({time.time() - t:.0f} detik)")
    k1s = args.k1 if args.k1_stem is None else args.k1_stem
    bs = args.b if args.b_stem is None else args.b_stem
    BM25Index.build(mapping[flat], offsets, stem_vocab, k1s, bs,
                    dict(common, stem=True, stemmer="PySastrawi")).save(args.index_dir / "bm25_stem")
    print("[bm25] indeks dengan stemming tersimpan")


def step_dense(args):
    from hybrid.dense import E5Encoder, encode_corpus

    t = time.time()
    model = str(Path(args.model).resolve()) if Path(args.model).exists() else args.model
    enc = E5Encoder(model, device=args.device, max_seq_length=config.MAX_SEQ_LENGTH)
    print(f"[dense] model {args.model} pada {enc.device}")
    texts = (doc_to_text(d) for d in iter_corpus(args.data_dir))
    path = encode_corpus(enc, texts, args.index_dir / args.dense_subdir, part_size=args.part_size,
                         batch_size=args.batch_size)
    print(f"[dense] {path} ({(time.time() - t) / 60:.1f} menit)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    ap.add_argument("--index-dir", type=Path, default=config.INDEX_DIR)
    ap.add_argument("--steps", nargs="+", default=["docstore", "bm25", "dense"],
                    choices=["docstore", "bm25", "dense"])
    ap.add_argument("--k1", type=float, default=config.K1)
    ap.add_argument("--b", type=float, default=config.B)
    ap.add_argument("--k1-stem", type=float, default=None, help="k1 untuk varian stemming (default = --k1)")
    ap.add_argument("--b-stem", type=float, default=None, help="b untuk varian stemming (default = --b)")
    ap.add_argument("--stopwords", default=None, help="file stopword kustom (default: PySastrawi)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--model", default=config.MODEL_NAME)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dense-subdir", default="dense", help="subfolder indeks dense (mis. dense_ft)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--part-size", type=int, default=50000)
    args = ap.parse_args()
    args.index_dir.mkdir(parents=True, exist_ok=True)

    for step in ("docstore", "bm25", "dense"):
        if step in args.steps:
            globals()[f"step_{step}"](args)


if __name__ == "__main__":
    main()
