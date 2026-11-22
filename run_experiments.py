"""Fase 3 - Evaluasi tujuh konfigurasi (Bagian 3.1, 3.5, 3.7).

Alur:
  1. Kueri dev dibagi menjadi bagian TUNE dan TEST (atau: tuning memakai kueri train, test = seluruh dev).
  2. Kandidat tiap jalur diambil sekali untuk semua kueri.
  3. alpha (linear) dan k (RRF) dipilih HANYA berdasarkan nDCG@10 pada bagian TUNE.
  4. Tujuh konfigurasi dievaluasi pada bagian TEST: nDCG@10, MAP@100, Recall@100, P@10.
  5. Uji berpasangan: bootstrap CI + uji randomisasi, koreksi Holm.
  6. Analisis sensitivitas K (pada TUNE), latensi per kueri, dan informasi lingkungan (Tabel 2).

Contoh:
    python run_experiments.py
    python run_experiments.py --tune-source train
    python run_experiments.py --data-dir data/miracl-id-sub50k --index-dir indexes/miracl-id-sub50k
"""
import argparse
import json
import pickle
import platform
import statistics
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid import config  # noqa: E402
from hybrid.data import load_qrels, load_queries, split_qids  # noqa: E402
from hybrid.fusion import fuse  # noqa: E402
from hybrid.metrics import evaluate_run  # noqa: E402
from hybrid.searcher import HybridSearcher  # noqa: E402
from hybrid.stats import holm, paired_bootstrap_ci, paired_randomization  # noqa: E402

CONFIGS = [  # (nama, mode, stemming)
    ("BM25", "bm25", False),
    ("BM25+stem", "bm25", True),
    ("Dense", "dense", False),
    ("Hybrid-RRF", "rrf", False),
    ("Hybrid-RRF+stem", "rrf", True),
    ("Hybrid-Linear", "linear", False),
    ("Hybrid-Linear+stem", "linear", True),
]
COMPARISONS = [  # (A, B): Delta = A - B
    ("Hybrid-RRF", "BM25"), ("Hybrid-RRF", "Dense"),
    ("Hybrid-Linear", "BM25"), ("Hybrid-Linear", "Dense"),
    ("Hybrid-RRF+stem", "BM25+stem"), ("Hybrid-RRF+stem", "Dense"),
    ("Hybrid-Linear+stem", "BM25+stem"), ("Hybrid-Linear+stem", "Dense"),
    ("Hybrid-Linear", "Hybrid-RRF"), ("Hybrid-Linear+stem", "Hybrid-RRF+stem"),
    ("BM25+stem", "BM25"), ("Hybrid-RRF+stem", "Hybrid-RRF"), ("Hybrid-Linear+stem", "Hybrid-Linear"),
]
MAIN_METRICS = ["nDCG@10", "MAP@100", "Recall@100"]


def to_markdown(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def env_info(args, searcher) -> dict:
    def ver(pkg):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None
    info = {"python": platform.python_version(), "os": platform.platform(), "cpu": platform.processor(),
            "packages": {p: ver(p) for p in ["numpy", "scipy", "pandas", "torch", "sentence-transformers",
                                              "transformers", "PySastrawi", "streamlit"]},
            "model": searcher.encoder.model_name if searcher.encoder else None,
            "device": searcher.encoder.device if searcher.encoder else "cpu",
            "seed": args.seed, "K": args.K, "eval_depth": config.EVAL_DEPTH,
            "bm25": searcher.bm25[False].meta, "bm25_stem": searcher.bm25[True].meta,
            "n_docs": len(searcher.docids), "data_dir": str(args.data_dir)}
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    ap.add_argument("--index-dir", type=Path, default=config.INDEX_DIR)
    ap.add_argument("--out-dir", type=Path, default=config.OUTPUT_DIR)
    ap.add_argument("--eval-split", default="dev")
    ap.add_argument("--tune-source", choices=["split", "train"], default="split",
                    help="split: bagi kueri dev menjadi tune/test; train: tuning memakai kueri train")
    ap.add_argument("--tune-fraction", type=float, default=config.TUNE_FRACTION)
    ap.add_argument("--max-tune-queries", type=int, default=1000)
    ap.add_argument("--K", type=int, default=config.K_CANDIDATES, help="kandidat per jalur")
    ap.add_argument("--alpha-grid", type=float, nargs="+", default=config.ALPHA_GRID)
    ap.add_argument("--rrf-grid", type=float, nargs="+", default=config.RRF_K_GRID)
    ap.add_argument("--K-sensitivity", type=int, nargs="*", default=[20, 50, 100, 200, 500])
    ap.add_argument("--n-resamples", type=int, default=10000)
    ap.add_argument("--latency-queries", type=int, default=50)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dense-subdir", default="dense", help="subfolder indeks dense (mis. dense_ft)")
    ap.add_argument("--bm25-only", action="store_true",
                    help="hanya konfigurasi BM25 (tanpa model dense), mis. sebelum embedding tersedia")
    args = ap.parse_args()
    configs = [c for c in CONFIGS if not args.bm25_only or c[1] == "bm25"]
    out = args.out_dir
    (out / "runs").mkdir(parents=True, exist_ok=True)

    # ---- 1. kueri dan pembagian data ---------------------------------------------------------
    eval_q = load_queries(args.data_dir / f"queries_{args.eval_split}.tsv")
    eval_qrels = load_qrels(args.data_dir / f"qrels_{args.eval_split}.tsv")
    eval_ids = [q for q in eval_q if any(r > 0 for r in eval_qrels.get(q, {}).values())]
    if args.tune_source == "split":
        tune_ids, test_ids = split_qids(eval_ids, args.tune_fraction, args.seed)
        tune_q, tune_qrels = eval_q, eval_qrels
    else:
        tune_q = load_queries(args.data_dir / "queries_train.tsv")
        tune_qrels = load_qrels(args.data_dir / "qrels_train.tsv")
        ids = [q for q in tune_q if any(r > 0 for r in tune_qrels.get(q, {}).values())]
        tune_ids, _ = split_qids(ids, min(1.0, args.max_tune_queries / max(len(ids), 1)), args.seed)
        test_ids = sorted(eval_ids)
    print(f"Kueri TUNE: {len(tune_ids)} | kueri TEST: {len(test_ids)}")
    json.dump({"tune": tune_ids, "test": test_ids, "tune_source": args.tune_source},
              open(out / "query_split.json", "w"), indent=1)

    # ---- 2. kandidat per jalur (sekali untuk semua kueri) -------------------------------------
    searcher = HybridSearcher(args.index_dir, device=args.device, load_dense=not args.bm25_only,
                              dense_subdir=args.dense_subdir)
    docids = searcher.docids
    depth = max([args.K, config.EVAL_DEPTH] + list(args.K_sensitivity))
    texts = {("tune", q): tune_q[q] for q in tune_ids} | {("test", q): eval_q[q] for q in test_ids}
    keys = list(texts)
    cand = {}
    for stem in (False, True):
        t = time.time()
        cand[f"bm25_{int(stem)}"] = {k: searcher.bm25_candidates(texts[k], stem, depth) for k in keys}
        print(f"Kandidat BM25 (stem={stem}): {time.time() - t:.1f} detik")
    if args.bm25_only:
        cand["dense"] = {k: [] for k in keys}
    else:
        t = time.time()
        dense = searcher.dense_candidates_batch([texts[k] for k in keys], depth)
        cand["dense"] = dict(zip(keys, dense))
        print(f"Kandidat dense: {time.time() - t:.1f} detik")
    with open(out / "candidates.pkl", "wb") as fh:
        pickle.dump({"keys": keys, "cand": cand, "docids_len": len(docids)}, fh)

    def make_run(part, qids, mode, stem, alpha=config.ALPHA, rrf_k=config.RRF_K, K=args.K):
        run = {}
        for q in qids:
            b, d = cand[f"bm25_{int(stem)}"][(part, q)], cand["dense"][(part, q)]
            if mode == "bm25":
                ranked = b
            elif mode == "dense":
                ranked = d
            else:
                ranked = fuse(mode, b[:K], d[:K], alpha, rrf_k)
            run[q] = [docids[i] for i, _ in ranked[:config.EVAL_DEPTH]]
        return run

    def mean_ndcg(run, qids, qrels):
        return evaluate_run(run, qrels, qids)["nDCG@10"].mean()

    # ---- 3. tuning pada TUNE -----------------------------------------------------------------
    tuning, best = [], {}
    for stem in (() if args.bm25_only else (False, True)):
        for mode, grid, key in (("linear", args.alpha_grid, "alpha"), ("rrf", args.rrf_grid, "rrf_k")):
            scores = []
            for v in grid:
                s = mean_ndcg(make_run("tune", tune_ids, mode, stem, **{key: v}), tune_ids, tune_qrels)
                scores.append(s)
                tuning.append({"mode": mode, "stem": stem, "param": key, "value": v, "nDCG@10_tune": s})
            best[(mode, stem)] = {key: grid[int(np.argmax(scores))]}
            print(f"  terbaik {mode:6s} stem={stem!s:5s}: {key} = {best[(mode, stem)][key]} "
                  f"(nDCG@10 tune = {max(scores):.4f})")
    pd.DataFrame(tuning).to_csv(out / "tuning.csv", index=False)

    # ---- 4. evaluasi TEST --------------------------------------------------------------------
    per_query, rows = {}, []
    for name, mode, stem in configs:
        params = best.get((mode, stem), {})
        run = make_run("test", test_ids, mode, stem, **params)
        df = evaluate_run(run, eval_qrels, test_ids)
        per_query[name] = df
        rows.append({"Konfigurasi": name, **df.mean().to_dict(),
                     "Parameter": ", ".join(f"{k}={v}" for k, v in params.items()) or "-"})
        with open(out / "runs" / f"{name}.trec", "w", encoding="utf-8") as fh:
            for q, docs in run.items():
                for r, d in enumerate(docs, start=1):
                    fh.write(f"{q} Q0 {d} {r} {config.EVAL_DEPTH - r + 1} {name}\n")
    summary = pd.DataFrame(rows).set_index("Konfigurasi")
    summary.to_csv(out / "results_test.csv")
    pd.concat(per_query, names=["Konfigurasi"]).to_csv(out / "per_query_test.csv")

    # ---- 5. uji statistik ---------------------------------------------------------------------
    stat_rows = []
    for metric in MAIN_METRICS:
        block = []
        for a, b in [c for c in COMPARISONS if c[0] in per_query and c[1] in per_query]:
            va, vb = per_query[a][metric].values, per_query[b][metric].values
            delta, lo, hi = paired_bootstrap_ci(va, vb, args.n_resamples, seed=args.seed)
            p = paired_randomization(va, vb, args.n_resamples, seed=args.seed)
            block.append({"metrik": metric, "A": a, "B": b, "delta(A-B)": delta,
                          "CI95_bawah": lo, "CI95_atas": hi, "p": p})
        for row, padj in zip(block, holm([r["p"] for r in block])):
            row["p_holm"] = padj
            row["signifikan(0.05)"] = padj < 0.05
        stat_rows.extend(block)
    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(out / "significance_test.csv", index=False)

    # ---- 6a. sensitivitas K pada TUNE --------------------------------------------------------
    sens = []
    for K in ([] if args.bm25_only else args.K_sensitivity):
        for mode in ("rrf", "linear"):
            for stem in (False, True):
                s = mean_ndcg(make_run("tune", tune_ids, mode, stem, K=K, **best[(mode, stem)]),
                              tune_ids, tune_qrels)
                sens.append({"K": K, "mode": mode, "stem": stem, "nDCG@10_tune": s})
    pd.DataFrame(sens).to_csv(out / "sensitivity_K.csv", index=False)

    # ---- 6b. latensi per kueri (kueri tunggal, seperti di antarmuka) -------------------------
    lat_rows = []
    sample = test_ids[:args.latency_queries]
    searcher.search(eval_q[sample[0]], configs[-1][1])  # pemanasan
    for name, mode, stem in configs:
        times = []
        for q in sample:
            t = time.perf_counter()
            searcher.search(eval_q[q], mode, stem, K=args.K, top_k=10, **best.get((mode, stem), {}))
            times.append((time.perf_counter() - t) * 1000)
        lat_rows.append({"Konfigurasi": name, "median_ms": statistics.median(times),
                         "mean_ms": statistics.mean(times), "p95_ms": float(np.percentile(times, 95))})
    pd.DataFrame(lat_rows).to_csv(out / "latency.csv", index=False)

    # ---- 6c. lingkungan + parameter terpilih -------------------------------------------------
    info = env_info(args, searcher)
    info["best_params"] = {f"{m}{'+stem' if s else ''}": v for (m, s), v in best.items()}
    info["n_tune"], info["n_test"] = len(tune_ids), len(test_ids)
    (out / "environment.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    report = ["# Hasil evaluasi (kueri TEST)", "", to_markdown(summary), "",
              "## Uji signifikansi (nDCG@10)", "",
              to_markdown(stats_df[stats_df.metrik == "nDCG@10"].drop(columns="metrik").set_index("A")), "",
              "## Latensi per kueri", "", to_markdown(pd.DataFrame(lat_rows).set_index("Konfigurasi"), "{:.1f}")]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n" + "\n".join(report))
    print(f"\nSemua keluaran tersimpan di {out}")


if __name__ == "__main__":
    main()
