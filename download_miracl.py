"""Unduh MIRACL bahasa Indonesia dan ubah ke format folder data proyek ini.

Contoh:
    python download_miracl.py                         # korpus penuh (~1,4 juta paragraf)
    python download_miracl.py --subset 50000          # korpus kecil untuk uji cepat

Mode --subset menyimpan SEMUA dokumen yang dinilai pada qrels (relevan & tidak relevan)
ditambah dokumen acak, sehingga evaluasi tetap bermakna tetapi jauh lebih ringan.
Angka pada mode subset TIDAK boleh dilaporkan sebagai hasil MIRACL.

Catatan: dataset miracl/miracl di Hugging Face mungkin perlu login (`huggingface-cli login`)
dan persetujuan syarat penggunaan di halaman dataset.
"""
import argparse
import gzip
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid import config  # noqa: E402

LANG = "id"


def find_one(root: Path, pattern: str) -> Path | None:
    hits = sorted(root.rglob(pattern))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--subset", type=int, default=0, help="jumlah dokumen acak tambahan (0 = korpus penuh)")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    out = args.out or (config.DATA_DIR if not args.subset else
                       config.DATA_DIR.parent / f"miracl-id-sub{args.subset // 1000}k")

    from huggingface_hub import snapshot_download

    print("Mengunduh topik dan qrels ...")
    topics_snap = Path(snapshot_download("miracl/miracl", repo_type="dataset",
                                         allow_patterns=[f"miracl-v1.0-{LANG}/*"]))
    print("Mengunduh korpus ...")
    corpus_snap = Path(snapshot_download("miracl/miracl-corpus", repo_type="dataset",
                                         allow_patterns=[f"miracl-corpus-v1.0-{LANG}/*"]))

    out.mkdir(parents=True, exist_ok=True)
    judged = set()
    for split in ("train", "dev"):
        t = find_one(topics_snap, f"topics*{LANG}-{split}.tsv")
        q = find_one(topics_snap, f"qrels*{LANG}-{split}.tsv")
        if not (t and q):
            print(f"  [lewati] split {split} tidak ditemukan")
            continue
        shutil.copy(t, out / f"queries_{split}.tsv")
        shutil.copy(q, out / f"qrels_{split}.tsv")
        for line in open(q, encoding="utf-8"):
            parts = line.split()
            if len(parts) == 4:
                judged.add(parts[2])
        print(f"  {split}: {t.name}, {q.name}")

    files = sorted(corpus_snap.rglob("*.jsonl.gz"))
    if not files:
        raise SystemExit("File korpus tidak ditemukan pada snapshot.")
    corpus_dir = out / "corpus"
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    corpus_dir.mkdir()

    if not args.subset:
        for f in files:
            shutil.copy(f, corpus_dir / f.name)
        print(f"Korpus penuh disalin: {len(files)} file -> {corpus_dir}")
    else:
        total = sum(1 for f in files for _ in gzip.open(f, "rt", encoding="utf-8"))
        p = args.subset / max(total - len(judged), 1)
        rng = random.Random(args.seed)
        kept = 0
        with gzip.open(corpus_dir / "docs.jsonl.gz", "wt", encoding="utf-8") as w:
            for f in files:
                for line in gzip.open(f, "rt", encoding="utf-8"):
                    docid = json.loads(line)["docid"]
                    if docid in judged or rng.random() < p:
                        w.write(line)
                        kept += 1
        print(f"Subset: {kept} dari {total} dokumen ({len(judged)} dokumen berlabel) -> {corpus_dir}")
    print(f"Selesai. Folder data: {out}")


if __name__ == "__main__":
    main()
