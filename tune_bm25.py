"""Penyetelan parameter BM25 (k1, b) HANYA pada kueri TUNE (Bagian 3.3).

Tokenisasi korpus dilakukan sekali, lalu indeks dibangun ulang di memori untuk setiap (k1, b).
Pembagian kueri identik dengan run_experiments.py (seed dan fraksi yang sama).

Contoh:
    python tune_bm25.py
    python build_index.py --steps bm25 --k1 <k1 terbaik> --b <b terbaik>
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid import config  # noqa: E402
from hybrid.bm25 import BM25Index, remap_vocab, tokenize_corpus  # noqa: E402
from hybrid.data import load_qrels, load_queries, split_qids  # noqa: E402
from hybrid.metrics import evaluate_run  # noqa: E402
from hybrid.preprocess import BM25Analyzer, CachedStemmer, load_stopwords, stem_vocabulary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    ap.add_argument("--index-dir", type=Path, default=config.INDEX_DIR)
    ap.add_argument("--out", type=Path, default=config.OUTPUT_DIR / "tuning_bm25.csv")
    ap.add_argument("--k1", type=float, nargs="+", default=[0.6, 0.9, 1.2, 1.5])
    ap.add_argument("--b", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.75])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    q = load_queries(args.data_dir / "queries_dev.tsv")
    qr = load_qrels(args.data_dir / "qrels_dev.tsv")
    ids = [k for k in q if any(r > 0 for r in qr.get(k, {}).values())]
    tune_ids, _ = split_qids(ids, config.TUNE_FRACTION, config.SEED)
    docids = json.loads((args.index_dir / "docids.json").read_text(encoding="utf-8"))
    print(f"Kueri TUNE: {len(tune_ids)}")

    sw = load_stopwords()
    flat, off, vocab = tokenize_corpus(args.data_dir, sw)
    mapping, stem_vocab = remap_vocab(vocab, stem_vocabulary(vocab, workers=args.workers))
    variants = {False: (flat, vocab, BM25Analyzer(sw)),
                True: (mapping[flat], stem_vocab, BM25Analyzer(sw, CachedStemmer()))}

    rows = []
    for stem, (f, v, an) in variants.items():
        qtok = {k: an(q[k]) for k in tune_ids}
        for k1, b in itertools.product(args.k1, args.b):
            t = time.time()
            idx = BM25Index.build(f, off, v, k1, b)
            run = {k: [docids[i] for i in idx.search(qtok[k], 100)[0]] for k in tune_ids}
            m = evaluate_run(run, qr, tune_ids).mean()
            rows.append({"stem": stem, "k1": k1, "b": b, **m.to_dict()})
            print(f"stem={stem!s:5s} k1={k1:<4} b={b:<4} nDCG@10={m['nDCG@10']:.4f} "
                  f"R@100={m['Recall@100']:.4f} ({time.time() - t:.0f} dtk)", flush=True)
            del idx
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    for stem in (False, True):
        best = df[df.stem == stem].sort_values("nDCG@10", ascending=False).iloc[0]
        print(f"TERBAIK stem={stem}: k1={best.k1} b={best.b} nDCG@10(tune)={best['nDCG@10']:.4f}")


if __name__ == "__main__":
    main()
