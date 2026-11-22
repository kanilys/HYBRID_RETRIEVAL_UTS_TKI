"""Konfigurasi default. Semua nilai dapat ditimpa lewat argumen command line."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "miracl-id"
INDEX_DIR = ROOT / "indexes" / "miracl-id"
OUTPUT_DIR = ROOT / "outputs"

# Jalur dense (Bagian 3.4)
MODEL_NAME = "intfloat/multilingual-e5-base"
MAX_SEQ_LENGTH = 512

# Jalur BM25 (Bagian 3.3) - nilai awal, bukan nilai optimal universal
K1 = 1.2
B = 0.75

# Fusi (Bagian 3.5)
K_CANDIDATES = 100      # K: jumlah kandidat per jalur
RRF_K = 60              # konstanta RRF awal
ALPHA = 0.5             # bobot jalur dense pada kombinasi linear
ALPHA_GRID = [round(x * 0.1, 1) for x in range(11)]
RRF_K_GRID = [1, 5, 10, 20, 40, 60, 80, 100]

# Evaluasi (Bagian 3.7)
EVAL_DEPTH = 100
SEED = 42
TUNE_FRACTION = 0.5     # porsi kueri dev untuk tuning bila tidak memakai split train
