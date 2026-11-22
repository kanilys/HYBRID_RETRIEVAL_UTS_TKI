"""Fase 2 - Antarmuka pencarian Streamlit (Bagian 3.6).

Jalankan:
    streamlit run app.py
    streamlit run app.py -- --index-dir indexes/miracl-id-sub50k
"""
import argparse
import html
import re
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid import config  # noqa: E402
from hybrid.data import DocStore  # noqa: E402
from hybrid.searcher import HybridSearcher  # noqa: E402

MODE_LABELS = {"BM25": "bm25", "Dense (multilingual-E5)": "dense",
               "Hibrida - RRF": "rrf", "Hibrida - Linear": "linear"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", type=Path, default=config.INDEX_DIR)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dense-subdir", default="dense", help="dense (E5 asli) atau dense_ft (fine-tuned)")
    return ap.parse_known_args(sys.argv[1:])[0]


@st.cache_resource(show_spinner="Memuat indeks dan model (hanya sekali) ...")
def load(index_dir: str, device: str | None, dense_subdir: str = "dense"):
    has_dense = (Path(index_dir) / dense_subdir / "embeddings.npy").exists()
    searcher = HybridSearcher(Path(index_dir), device=device, load_dense=has_dense, dense_subdir=dense_subdir)
    return searcher, DocStore(Path(index_dir) / "docstore.sqlite")


def highlight(text: str, terms: list[str], limit: int = 400) -> str:
    text = html.escape(text[:limit] + ("..." if len(text) > limit else ""))
    for t in sorted(set(terms or []), key=len, reverse=True):
        if len(t) > 2:
            text = re.sub(rf"(?i)\b({re.escape(t)}\w*)", r"<mark>\1</mark>", text)
    return text


def fmt(v, digits=4):
    return "-" if v is None else f"{v:.{digits}f}"


def main():
    args = parse_args()
    st.set_page_config(page_title="Pencarian Hibrida BM25 + E5", layout="wide")
    st.title("Pencarian Dokumen Bahasa Indonesia")
    st.caption("Hybrid retrieval: BM25 + multilingual-E5 (RRF / kombinasi linear)")

    searcher, store = load(str(args.index_dir), args.device, args.dense_subdir)

    with st.sidebar:
        st.header("Pengaturan")
        labels = list(MODE_LABELS) if searcher.dense is not None else ["BM25"]
        if searcher.dense is None:
            st.info("Indeks dense belum tersedia: hanya mode BM25.")
        mode = MODE_LABELS[st.radio("Mode pencarian", labels, index=min(2, len(labels) - 1))]
        stem = st.checkbox("Stemming pada jalur BM25", value=False, disabled=(mode == "dense"))
        alpha = st.slider("α (bobot dense)", 0.0, 1.0, config.ALPHA, 0.05, disabled=(mode != "linear"))
        rrf_k = st.number_input("k (konstanta RRF)", 1, 1000, config.RRF_K, disabled=(mode != "rrf"))
        K = st.select_slider("K (kandidat per jalur)", [20, 50, 100, 200, 500, 1000], config.K_CANDIDATES,
                             disabled=mode in ("bm25", "dense"))
        top_k = st.slider("Jumlah hasil (top-k)", 5, 50, 10)
        st.caption(f"{len(searcher.docids):,} dokumen terindeks")

    query = st.text_input("Kueri", placeholder="contoh: kendaraan listrik di Indonesia")
    if not query.strip():
        return

    out = searcher.search(query, mode=mode, stem=stem, alpha=alpha, rrf_k=rrf_k, K=K, top_k=top_k)
    docs = store.get([r["idx"] for r in out["results"]])
    t = out["timings"]
    st.caption(" · ".join(f"{k.replace('_ms', '')}: {v:.0f} ms" for k, v in t.items())
               + (f" · token BM25: {out['query_tokens']}" if out["query_tokens"] is not None else ""))

    if not out["results"]:
        st.warning("Tidak ada dokumen yang cocok.")
        return

    for r in out["results"]:
        doc = docs.get(r["idx"], {})
        with st.container(border=True):
            st.markdown(f"**{r['rank']}. {html.escape(doc.get('title') or '(tanpa judul)')}**  "
                        f"<span style='opacity:.6'>`{r['docid']}`</span>", unsafe_allow_html=True)
            st.markdown(highlight(doc.get("text", ""), out["query_tokens"] or query.lower().split()),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Skor akhir", fmt(r["score"]))
            c2.metric("BM25 (peringkat · skor)",
                      f"#{r['bm25_rank']} · {fmt(r['bm25_score'], 2)}" if r["bm25_rank"] else "-")
            c3.metric("Dense (peringkat · kosinus)",
                      f"#{r['dense_rank']} · {fmt(r['dense_score'])}" if r["dense_rank"] else "-")


if __name__ == "__main__":
    main()
