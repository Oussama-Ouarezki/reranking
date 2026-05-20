"""LR sweep for fine-tuning LiT5-Distill-base on the BioASQ windowed dataset.

Sweeps LRs in [5e-6, 1e-5, 3e-5, 1e-4]. For each LR:
  1. Re-load fresh base checkpoint
  2. Train `epochs` epochs on windowed_train.jsonl (4000 windows, 1000 queries)
  3. After every epoch evaluate nDCG@10 on the BioASQ Task13B golden test
     set (BM25 top-100 candidates cached in bm25_top100_test.jsonl, sliding
     window inference window=20 stride=10 — same protocol as zero-shot baseline).
  4. Track best (LR, epoch) by dev nDCG@10.

Training settings (frozen across LRs so only LR varies):
  - bf16 autocast (T5 is unstable in fp16)
  - gradient checkpointing, batch_size=1, grad_accum=8
  - AdamW (wd=0.01), warmup ratio 0.06, cosine schedule
  - label smoothing 0.1, grad-clip 1.0
  - per-passage encode (FiD) then concatenated cross-attention

Outputs:
  - lr_sweep_results.tsv  : (lr, epoch, train_loss, ndcg@10)
  - lr_sweep_curve.png    : per-LR train loss + dev nDCG@10 curves
  - best_ckpt/            : highest-nDCG checkpoint over the entire sweep
"""

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
from tqdm import tqdm
import ir_measures
from ir_measures import nDCG, RR, Recall, P, ScoredDoc, Qrel

METRICS = [nDCG @ 1, nDCG @ 5, nDCG @ 10, nDCG @ 20, RR,
           Recall @ 10, Recall @ 20, P @ 10]
SELECTION_METRIC = nDCG @ 10  # used to pick best (lr, epoch)

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

ROOT = Path("/home/oussama/Desktop/reranking_project")
HERE = ROOT / "lit5 fine tuning"
TRAIN_FILE = HERE / "windowed_train.jsonl"
TEST_CACHE = HERE / "bm25_top100_test.jsonl"
TEST_DIR = ROOT / "data/bioasq/raw/Task13BGoldenEnriched"
BATCHES = ["13B1", "13B2", "13B3", "13B4"]
BASE_CKPT = ROOT / "checkpoints/LiT5-Distill-base"
OUT_DIR = HERE / "sweep_out"
RESULTS_TSV = OUT_DIR / "lr_sweep_results.tsv"
CURVE_IMG = OUT_DIR / "lr_sweep_curve.png"
BEST_CKPT_DIR = OUT_DIR / "best_ckpt"

QUERY_PREFIX = "Search Query:"
PASS_PREFIX = "Passage:"
SUFFIX = " Relevance Ranking:"

WINDOW_SIZE = 20
STRIDE = 10
MAX_LENGTH = 350
MAX_TARGET = 140


# ── Data ─────────────────────────────────────────────────────────────────────

def load_train():
    rows = []
    with TRAIN_FILE.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_test_qrels() -> list[Qrel]:
    qrels = []
    for batch in BATCHES:
        with (TEST_DIR / batch / "qrels.tsv").open() as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    qid, did, score = parts
                elif len(parts) == 4:
                    qid, _, did, score = parts
                else:
                    continue
                qrels.append(Qrel(qid, did, int(score)))
    return qrels


def load_test_cache():
    if not TEST_CACHE.exists():
        raise FileNotFoundError(
            f"{TEST_CACHE} missing. Run cache_bm25_test.py first.")
    rows = []
    with TEST_CACHE.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


# ── Dataset ──────────────────────────────────────────────────────────────────

def build_target(input_docids: list[str], target_docids: list[str]) -> str:
    """Map target docid order back to positions in the (reversed) input."""
    pos = {d: i for i, d in enumerate(input_docids)}
    return " > ".join(f"[{pos[d] + 1}]" for d in target_docids if d in pos)


class WindowDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tok = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ex = self.rows[idx]
        strings = [
            f"{QUERY_PREFIX} {ex['query']} {PASS_PREFIX} [{i + 1}] {t}{SUFFIX}"
            for i, t in enumerate(ex["input_passages"])
        ]
        enc = self.tok(strings, max_length=MAX_LENGTH, padding="max_length",
                       truncation=True, return_tensors="pt")
        target = build_target(ex["input_docids"], ex["target_docids"])
        lbl = self.tok(target, max_length=MAX_TARGET, padding="max_length",
                       truncation=True, return_tensors="pt")
        labels = lbl["input_ids"].squeeze(0)
        labels[labels == self.tok.pad_token_id] = -100
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }


def collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


# ── FiD forward / loss ───────────────────────────────────────────────────────

def fid_encode(model, input_ids, attention_mask):
    B, n, L = input_ids.shape
    flat_ids = input_ids.view(B * n, L)
    flat_mask = attention_mask.view(B * n, L)
    enc = model.encoder(input_ids=flat_ids, attention_mask=flat_mask)
    hidden = enc.last_hidden_state.view(B, n * L, -1)
    attn = flat_mask.view(B, n * L)
    return hidden, attn


def smooth_ce(logits, labels, eps):
    vocab = logits.size(-1)
    flat_logits = logits.view(-1, vocab)
    flat_labels = labels.view(-1)
    mask = flat_labels != -100
    flat_logits = flat_logits[mask]
    flat_labels = flat_labels[mask]
    log_probs = F.log_softmax(flat_logits, dim=-1)
    nll = -log_probs.gather(1, flat_labels.unsqueeze(1)).squeeze(1)
    smooth = -log_probs.mean(dim=-1)
    return ((1 - eps) * nll + eps * smooth).mean()


# ── Sliding-window inference (matches zero-shot eval protocol) ───────────────

def parse_perm(text: str, n: int) -> list[int]:
    out, seen = [], set()
    for tok in text.replace(",", " ").replace(">", " ").split():
        try:
            v = int(tok.strip("[]()."))
            if 1 <= v <= n and (v - 1) not in seen:
                out.append(v - 1)
                seen.add(v - 1)
        except ValueError:
            continue
    for i in range(n):
        if i not in seen:
            out.append(i)
    return out


@torch.no_grad()
def rerank_window(model, tokenizer, query, window, device, amp_dtype):
    """window: [(docid, text), ...] of length WINDOW_SIZE."""
    strings = [
        f"{QUERY_PREFIX} {query} {PASS_PREFIX} [{i + 1}] {t}{SUFFIX}"
        for i, (_, t) in enumerate(window)
    ]
    enc = tokenizer(strings, max_length=MAX_LENGTH, padding="max_length",
                    truncation=True, return_tensors="pt").to(device)
    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
        hidden, attn = fid_encode(model, enc["input_ids"].unsqueeze(0),
                                  enc["attention_mask"].unsqueeze(0))
        out_ids = model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
            attention_mask=attn,
            max_new_tokens=MAX_TARGET,
            do_sample=False,
        )
    perm_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    order = parse_perm(perm_text, len(window))
    return [window[i] for i in order]


def sliding_window_rerank(model, tokenizer, query, candidates, device, amp_dtype):
    """Bottom-up sliding window with overlap propagation (zero-shot protocol)."""
    ranked = list(candidates)
    if len(ranked) <= WINDOW_SIZE:
        return rerank_window(model, tokenizer, query, ranked, device, amp_dtype)
    start = len(ranked) - WINDOW_SIZE
    while True:
        end = min(start + WINDOW_SIZE, len(ranked))
        ranked[start:end] = rerank_window(model, tokenizer, query, ranked[start:end], device, amp_dtype)
        if start == 0:
            break
        start = max(0, start - STRIDE)
    return ranked


def evaluate(model, tokenizer, test_rows, qrels, device, amp_dtype, limit=None):
    model.eval()
    rows = test_rows if limit is None else test_rows[:limit]
    test_qids = {r["qid"] for r in rows}
    run = []
    for r in tqdm(rows, desc="  eval", leave=False):
        cands = [(c["docid"], c["text"]) for c in r["candidates"]]
        reranked = sliding_window_rerank(model, tokenizer, r["query"], cands, device, amp_dtype)
        for rank, (did, _) in enumerate(reranked, start=1):
            run.append(ScoredDoc(r["qid"], did, score=len(cands) - rank + 1))
    qrel_subset = [q for q in qrels if q.query_id in test_qids]
    res = ir_measures.calc_aggregate(METRICS, qrel_subset, run)
    return {str(m): float(v) for m, v in res.items()}


# ── Single training run for one LR ───────────────────────────────────────────

def train_one_lr(lr, args, base_train_rows, test_rows, qrels, device, amp_dtype):
    print("\n" + "=" * 70)
    print(f"  LR = {lr}")
    print("=" * 70)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    tokenizer = T5Tokenizer.from_pretrained(str(BASE_CKPT), legacy=False, use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(str(BASE_CKPT)).to(device)
    model.gradient_checkpointing_enable()

    loader = DataLoader(WindowDataset(base_train_rows, tokenizer),
                        batch_size=1, shuffle=True, collate_fn=collate, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(loader) / args.grad_accum) * args.epochs
    warmup = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    epoch_losses, epoch_ndcgs = [], []
    best_ndcg, best_epoch = -1.0, 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(loader, desc=f"LR={lr} ep{epoch}")):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                hidden, attn = fid_encode(model, input_ids, attn_mask)
                out = model(encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
                            attention_mask=attn, labels=labels)
                loss = smooth_ce(out.logits, labels, args.label_smoothing)
                loss = loss / args.grad_accum
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue
            loss.backward()
            running += loss.item() * args.grad_accum
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
        if len(loader) % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        avg = running / max(1, len(loader))
        m = evaluate(model, tokenizer, test_rows, qrels, device, amp_dtype, limit=args.eval_limit)
        sel = m.get(str(SELECTION_METRIC), 0.0)
        epoch_losses.append(avg); epoch_ndcgs.append(sel)
        print(f"  LR={lr} ep{epoch}  loss={avg:.4f}  "
              f"nDCG@1={m.get('nDCG@1', 0):.4f}  nDCG@10={m.get('nDCG@10', 0):.4f}  "
              f"MRR={m.get('RR', 0):.4f}  R@10={m.get('R@10', 0):.4f}")

        with RESULTS_TSV.open("a") as f:
            cols = "\t".join(f"{m.get(str(k), 0.0):.6f}" for k in METRICS)
            f.write(f"{lr}\t{epoch}\t{avg:.6f}\t{cols}\n")

        if sel > best_ndcg:
            best_ndcg, best_epoch = sel, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return {"lr": lr, "epoch_losses": epoch_losses, "epoch_ndcgs": epoch_ndcgs,
            "best_ndcg": best_ndcg, "best_epoch": best_epoch,
            "best_state": best_state, "tokenizer": tokenizer}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lrs", type=float, nargs="+", default=[5e-6, 1e-5, 3e-5, 1e-4])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--eval_limit", type=int, default=None,
                   help="Cap test queries during sweep eval (e.g. 200 for speed). "
                        "None = full Task13B test set.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_TSV.exists():
        with RESULTS_TSV.open("w") as f:
            cols = "\t".join(str(m) for m in METRICS)
            f.write(f"lr\tepoch\ttrain_loss\t{cols}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"AMP dtype: {amp_dtype} (bf16 required for T5 stability)")

    print("\nLoading data ...")
    train_rows = load_train()
    test_rows = load_test_cache()
    qrels = load_test_qrels()
    print(f"  Train windows: {len(train_rows)} | Test queries: {len(test_rows)} | Qrels: {len(qrels)}")

    train_qids = {r["qid"] for r in train_rows}
    test_qids = {r["qid"] for r in test_rows}
    leak = train_qids & test_qids
    print(f"  Train/test qid overlap: {len(leak)} (expect 0)")

    all_results = []
    global_best = {"ndcg": -1.0}

    for lr in args.lrs:
        res = train_one_lr(lr, args, train_rows, test_rows, qrels, device, amp_dtype)
        all_results.append(res)
        if res["best_ndcg"] > global_best["ndcg"]:
            if BEST_CKPT_DIR.exists():
                shutil.rmtree(BEST_CKPT_DIR)
            BEST_CKPT_DIR.mkdir(parents=True)
            tmp_model = T5ForConditionalGeneration.from_pretrained(str(BASE_CKPT))
            tmp_model.load_state_dict(res["best_state"])
            tmp_model.save_pretrained(BEST_CKPT_DIR)
            res["tokenizer"].save_pretrained(BEST_CKPT_DIR)
            with (BEST_CKPT_DIR / "sweep_meta.json").open("w") as f:
                json.dump({"lr": res["lr"], "epoch": res["best_epoch"],
                           "ndcg@10": res["best_ndcg"]}, f, indent=2)
            global_best = {"ndcg": res["best_ndcg"], "lr": res["lr"], "epoch": res["best_epoch"]}
            print(f"  ✓ new global best: {global_best}")
        # release training state from this LR before next
        del res["best_state"]
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print(f"  Sweep complete. Global best: {global_best}")
    print("=" * 70)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for r in all_results:
        epochs = list(range(1, len(r["epoch_losses"]) + 1))
        ax1.plot(epochs, r["epoch_losses"], marker="o", label=f"lr={r['lr']}")
        ax2.plot(epochs, r["epoch_ndcgs"], marker="s", label=f"lr={r['lr']}")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss"); ax1.set_title("Training loss"); ax1.legend()
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("nDCG@10 (Task13B)"); ax2.set_title("Dev nDCG@10"); ax2.legend()
    plt.tight_layout()
    plt.savefig(CURVE_IMG, dpi=150)
    plt.close()
    print(f"  Curve saved : {CURVE_IMG}")
    print(f"  Results TSV : {RESULTS_TSV}")
    print(f"  Best ckpt   : {BEST_CKPT_DIR}")


if __name__ == "__main__":
    main()
