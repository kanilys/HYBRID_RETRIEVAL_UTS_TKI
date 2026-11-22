# Hybrid Retrieval BM25 + Multilingual-E5 (Bahasa Indonesia)

Implementasi rancangan pada *Laporan_Hybrid_Retrieval*: BM25 (dengan/tanpa stemming) +
`intfloat/multilingual-e5-base` tanpa fine-tuning, fusi RRF dan kombinasi linear, evaluasi tujuh
konfigurasi pada MIRACL bahasa Indonesia, dan antarmuka Streamlit.

## Struktur

| File | Isi | Bagian laporan |
|---|---|---|
| `hybrid/preprocess.py` | pembersihan umum, tokenisasi, stopword, stemmer PySastrawi (dengan cache) | 3.2 |
| `hybrid/bm25.py` | BM25 berbasis matriks jarang (varian IDF Lucene) | 3.3, pers. (1)–(2) |
| `hybrid/dense.py` | encoder E5 (prefix, mean pooling, L2), pencarian eksak per blok | 3.4, pers. (3)–(4) |
| `hybrid/fusion.py` | min-max, kombinasi linear, RRF | 3.5, pers. (5)–(7) |
| `hybrid/metrics.py` | nDCG@10, MAP@100, Recall@100, P@10 (konvensi trec_eval) | 3.7, pers. (8)–(13) |
| `hybrid/stats.py` | paired bootstrap CI, uji randomisasi berpasangan, koreksi Holm | 3.7, pers. (14) |
| `hybrid/searcher.py` | penggabungan semua komponen | Algoritma 2 |
| `download_miracl.py` | unduh MIRACL-id dan ubah ke format proyek | 3.2 |
| `build_index.py` | pengindeksan luring | Algoritma 1 |
| `run_experiments.py` | tuning, evaluasi 7 konfigurasi, uji statistik, latensi, lingkungan | 3.1, 3.7, 3.8 |
| `app.py` | antarmuka Streamlit | 3.6 |
| `tests/` | uji unit (rumus BM25, fusi, metrik, statistik) | – |

## Langkah menjalankan

```bash
# 0. Instalasi (untuk GPU NVIDIA, pasang torch versi CUDA dulu: https://pytorch.org/get-started)
pip install -r requirements.txt
python -m pytest tests -q

# 1. Data (dataset miracl/miracl mungkin perlu: huggingface-cli login + setujui syarat di halaman HF)
python download_miracl.py --subset 50000        # uji cepat dulu (data/miracl-id-sub50k)
python download_miracl.py                       # korpus penuh (data/miracl-id)

# 2. Indexing
python build_index.py --data-dir data/miracl-id-sub50k --index-dir indexes/miracl-id-sub50k
python build_index.py                           # korpus penuh

# 3. Eksperimen
python run_experiments.py --data-dir data/miracl-id-sub50k --index-dir indexes/miracl-id-sub50k --out-dir outputs/sub50k
python run_experiments.py                       # korpus penuh -> outputs/

# 3b. (Opsional) penyetelan k1/b BM25 dan eksperimen fine-tuning E5
python tune_bm25.py                                   # pilih k1, b pada kueri TUNE
python build_index.py --steps bm25 --k1 0.9 --b 0.2 --k1-stem 0.6 --b-stem 0.3
python finetune_e5.py                                 # ±6 menit di RTX 5050
python build_index.py --steps dense --model models/e5-base-miracl-id --dense-subdir dense_ft
python run_experiments.py --dense-subdir dense_ft --out-dir outputs/ft

# 3c. Laporan (Word) dan slide (PowerPoint) dari hasil eksperimen
python laporan/build_report.py
cd laporan/slides && python make_data.py && node build_deck.js   # perlu NODE_PATH dan PPTX_SKILL

# 4. Antarmuka
streamlit run app.py
streamlit run app.py -- --index-dir indexes/miracl-id-sub50k
```

**Uji cepat dulu di subset.** Angka dari subset tidak boleh dilaporkan sebagai hasil MIRACL.

## Embedding di Kaggle/Colab (opsional)

GPU laptop 8 GB sudah cukup. Kalau ingin memakai Kaggle/Colab, salin folder proyek beserta
`data/miracl-id/corpus`, lalu jalankan `python build_index.py --steps dense`. Setelah selesai,
unduh folder `indexes/miracl-id/dense/` ke laptop. Proses ini **bisa dilanjutkan** bila sesi
terputus, karena bagian `emb_parts/part_*.npy` yang sudah ada akan dilewati.

## Perkiraan sumber daya (korpus penuh ±1,45 juta paragraf)

- Embedding: float16, sekitar 2,2 GB di disk/RAM. Waktu encode di GPU laptop kira-kira 1–3 jam.
- Indeks BM25: sekitar 1 GB per varian. Proses indexing butuh RAM puncak beberapa GB.
- Stemming hanya dilakukan pada **kosakata unik**, dijalankan paralel (`--workers`).
- Kalau RAM kurang, buka embedding dengan memmap: `HybridSearcher(..., dense_in_memory=False)`. Pencarian akan lebih lambat.

## Keluaran `run_experiments.py`

| File | Untuk laporan |
|---|---|
| `results_test.csv`, `report.md` | tabel utama 7 konfigurasi (Bagian Hasil) |
| `significance_test.csv` | Δ, CI 95%, p, p-Holm untuk setiap perbandingan |
| `tuning.csv` | sensitivitas α dan k RRF (pada kueri TUNE) |
| `sensitivity_K.csv` | sensitivitas jumlah kandidat K |
| `latency.csv` | waktu pencarian per kueri |
| `environment.json` | pengisian Tabel 2 (versi pustaka, GPU, parameter terpilih) |
| `per_query_test.csv`, `runs/*.trec` | analisis per kueri / contoh berhasil-gagal; dapat dicek ulang dengan trec_eval |

## Keputusan desain yang perlu disebut di laporan

1. **Pembagian data:** qrels test MIRACL tidak dipakai. Kueri **dev** dibagi 50/50 (seed 42) menjadi TUNE (untuk memilih α dan k) dan TEST (untuk pelaporan). Alternatifnya `--tune-source train`: tuning memakai kueri train, dan seluruh dev menjadi TEST.
2. **BM25** diimplementasikan sendiri dengan `scipy.sparse` (metode *eager scoring* yang sama dengan bm25s) agar rumus (1)–(2) persis terkontrol. Perbarui baris "Pustaka BM25" pada Tabel 2.
3. Istilah kueri diperlakukan sebagai himpunan, dan |d| dihitung setelah penghapusan stopword.
4. Teks dokumen = judul + isi, untuk kedua jalur. Jalur dense tidak diberi stemming maupun penghapusan stopword.
5. K = 100 kandidat per jalur (bisa diubah dengan `--K`). Pada kombinasi linear, skor jalur yang tidak memuat dokumen diisi 0.
6. Pencarian vektor eksak memakai perkalian matriks NumPy per blok, tanpa FAISS.
