"""Eksperimen tambahan: fine-tuning multilingual-E5 pada kueri TRAIN MIRACL bahasa Indonesia.

- Data: setiap pasangan (kueri train, passage relevan) + satu hard negative dari passage yang dinilai
  TIDAK relevan oleh anotator (qrels rel = 0); jika tidak ada, negatif diambil acak dari korpus.
- Loss: InfoNCE / contrastive dengan negatif dalam batch + hard negative, temperatur 0.01 (seperti E5).
- Kueri dev (TUNE maupun TEST) tidak dipakai sama sekali. Hiperparameter ditetapkan di awal (tidak disetel).
- Keluaran: folder model yang dapat dimuat sentence-transformers (models/e5-base-miracl-id).

Contoh:
    python finetune_e5.py
    python build_index.py --steps dense --model models/e5-base-miracl-id --dense-subdir dense_ft
    python run_experiments.py --dense-subdir dense_ft --out-dir outputs/ft
"""
import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid import config  # noqa: E402
from hybrid.data import DocStore, doc_to_text, load_qrels, load_queries  # noqa: E402
from hybrid.dense import PASSAGE_PREFIX, QUERY_PREFIX  # noqa: E402
from hybrid.preprocess import clean_text  # noqa: E402


def build_examples(data_dir: Path, index_dir: Path, seed: int):
    q = load_queries(data_dir / "queries_train.tsv")
    qr = load_qrels(data_dir / "qrels_train.tsv")
    docids = json.loads((index_dir / "docids.json").read_text(encoding="utf-8"))
    id2idx = {d: i for i, d in enumerate(docids)}
    store = DocStore(index_dir / "docstore.sqlite")
    rng = random.Random(seed)

    triples = []
    for qid, judged in qr.items():
        if qid not in q:
            continue
        pos = [d for d, r in judged.items() if r > 0 and d in id2idx]
        neg = [d for d, r in judged.items() if r == 0 and d in id2idx]
        for p in pos:
            n = rng.choice(neg) if neg else docids[rng.randrange(len(docids))]
            triples.append((qid, p, n))
    need = sorted({id2idx[d] for _, p, n in triples for d in (p, n)})
    texts = {}
    for i in range(0, len(need), 900):
        for idx, doc in store.get(need[i:i + 900]).items():
            texts[doc["docid"]] = doc_to_text(doc)
    rng.shuffle(triples)
    return [(QUERY_PREFIX + clean_text(q[a]), PASSAGE_PREFIX + texts[b], PASSAGE_PREFIX + texts[c])
            for a, b, c in triples], len(qr), sum(1 for v in qr.values() if any(r == 0 for r in v.values()))


def mean_pool(last_hidden, mask):
    m = mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    ap.add_argument("--index-dir", type=Path, default=config.INDEX_DIR)
    ap.add_argument("--base", default=config.MODEL_NAME)
    ap.add_argument("--out", type=Path, default=config.ROOT / "models" / "e5-base-miracl-id")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temperature", type=float, default=0.01)
    ap.add_argument("--max-q-len", type=int, default=64)
    ap.add_argument("--max-p-len", type=int, default=256)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    examples, n_q, n_q_hard = build_examples(args.data_dir, args.index_dir, args.seed)
    print(f"Kueri train: {n_q} (dengan hard negative berlabel: {n_q_hard}) | triplet: {len(examples)}")

    base_dir = Path(snapshot_download(args.base))
    tok = AutoTokenizer.from_pretrained(base_dir)
    model = AutoModel.from_pretrained(base_dir).cuda()
    model.gradient_checkpointing_enable()
    model.train()

    steps_per_epoch = math.ceil(len(examples) / args.batch_size)
    total = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * total), total)

    def encode(texts, max_len):
        b = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**b).last_hidden_state
        return F.normalize(mean_pool(out.float(), b["attention_mask"]), dim=-1)

    log = []
    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(examples)
        for i in range(0, len(examples), args.batch_size):
            batch = examples[i:i + args.batch_size]
            qs, ps, ns = zip(*batch)
            q = encode(list(qs), args.max_q_len)
            d = encode(list(ps) + list(ns), args.max_p_len)
            scores = q @ d.T / args.temperature          # (B, 2B): positif ke-i ada di kolom i
            loss = F.cross_entropy(scores, torch.arange(len(batch), device="cuda"))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 20 == 0 or step == total:
                log.append({"step": step, "loss": loss.item()})
                print(f"epoch {epoch + 1} langkah {step}/{total} loss {loss.item():.4f} "
                      f"({time.time() - t0:.0f} dtk)", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    for name in ("modules.json", "sentence_bert_config.json"):
        shutil.copy(base_dir / name, args.out / name)
    shutil.copytree(base_dir / "1_Pooling", args.out / "1_Pooling", dirs_exist_ok=True)
    (args.out / "training_info.json").write_text(json.dumps(
        {"base": args.base, "n_train_queries": n_q, "n_triples": len(examples), "epochs": args.epochs,
         "batch_size": args.batch_size, "lr": args.lr, "temperature": args.temperature,
         "max_q_len": args.max_q_len, "max_p_len": args.max_p_len, "warmup": args.warmup, "seed": args.seed,
         "train_seconds": time.time() - t0, "loss_log": log}, indent=1), encoding="utf-8")
    print(f"Model tersimpan di {args.out} ({(time.time() - t0) / 60:.1f} menit)")


if __name__ == "__main__":
    main()
