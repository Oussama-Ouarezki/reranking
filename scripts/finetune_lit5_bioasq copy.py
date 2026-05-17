"""
Fine-tune LiT5-Distill on BioASQ using oracle-hybrid permutations as teacher signal.

Dataset  : lit5 fine tuning/windowed_train.jsonl
           Pre-built windowed examples (2 windows per query: [0:20] and [10:30]).
           Each line has: qid, query, window_start, input_docids, input, target
           where `input` is the LiT5-formatted prompt string and `target` is
           the position-based permutation string "[j] > [k] > ...".

Method   : Supervised listwise distillation.
           2 windows per query × 1000 queries = 2000 training examples.

Input    : FiD-encoded (query, passage_i) pairs, n=20 passages (reversed BM25 order)
Target   : permutation string "[j] > [k] > ..." — positions sorted by hybrid rank

Anti-overfitting:
  - 80/20 train/val split stratified by query
  - Weight decay 0.01, label smoothing 0.1
  - Gradient checkpointing (saves ~4 GB)
  - Early stopping on validation nDCG@5 (patience=3)  ← now nDCG-based
  - Cosine LR schedule with warmup
  - bfloat16 (stable on Ampere+ GPUs; no GradScaler needed)

Validation metrics:
  - Validation loss
  - nDCG@1, nDCG@5, nDCG@10  (early stopping uses nDCG@5)

Checkpoints:
  - ALL checkpoints are saved (one per epoch)
  - Best checkpoint symlink updated whenever nDCG@5 improves
  - Training can be resumed from the last saved checkpoint

Outputs:
  - checkpoints/lit5_finetune_oracle/ep{N}_ndcg5{X.XXXX}/   (all epochs)
  - checkpoints/lit5_finetune_oracle/best/                   (symlink → best)
  - checkpoints/lit5_finetune_oracle/training_results.txt    (per-epoch table)
  - checkpoints/lit5_finetune_oracle/training_loss.png       (train + val loss)
  - checkpoints/lit5_finetune_oracle/ndcg_metrics.png        (nDCG@1/5/10)

NaN fixes vs original:
  - bfloat16 replaces fp16 → eliminates most overflow-induced NaN
  - smooth_ce casts logits to fp32 before log_softmax
  - Guard against all-masked label tensors in smooth_ce
  - hidden states clamped before decoder cross-attention
  - Gradient update skipped when grad norm is non-finite
  - TEXT_MAXLENGTH reduced to 256 (n*L = 5120 instead of 7000)

Usage:
    python scripts/finetune_lit5_bioasq.py
    python scripts/finetune_lit5_bioasq.py --lr 1e-5 --epochs 15
    python scripts/finetune_lit5_bioasq.py --resume   # auto-resume from last ckpt
"""

import argparse
import json
import math
import random
import re
import os
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

# ── Plotting theme ────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
DATASET_FILE = ROOT / "lit5 fine tuning/windowed_train.jsonl"
BASE_CKPT    = ROOT / "checkpoints/LiT5-Distill-base"
OUT_DIR      = ROOT / "checkpoints/lit5_finetune_oracle"
LOSS_IMG     = OUT_DIR / "training_loss.png"
NDCG_IMG     = OUT_DIR / "ndcg_metrics.png"
RESULTS_TXT  = OUT_DIR / "training_results.txt"

# ── Default hyperparameters ───────────────────────────────────────────────────
WINDOW_SIZE    = 20
TEXT_MAXLENGTH = 350
MAX_NEW_TOKENS = 140
LR             = 1e-5
WEIGHT_DECAY   = 0.01
WARMUP_RATIO   = 0.06
EPOCHS         = 15
GRAD_ACCUM     = 8
PATIENCE       = 3     # early stopping based on nDCG@5
VAL_SPLIT      = 0.20
LABEL_SMOOTH   = 0.1
SEED           = 42

BF16 = torch.bfloat16


# ─────────────────────────────────────────────────────────────────────────────
# nDCG utilities
# ─────────────────────────────────────────────────────────────────────────────

def parse_permutation(perm_str: str) -> list[int]:
    """
    Parse a permutation string like "[3] > [1] > [2]" into a 0-indexed list
    [2, 0, 1] (the rank ordering of positions).

    Returns [] on parse failure.
    """
    tokens = re.findall(r'\[(\d+)\]', perm_str)
    if not tokens:
        return []
    return [int(t) - 1 for t in tokens]   # convert 1-indexed → 0-indexed


def permutation_to_ranking(perm: list[int], n: int) -> list[int]:
    """
    Given a permutation [p0, p1, ..., pk] (best-first), return a rank array
    `rank[i]` = rank of document i (0 = best).

    Documents not mentioned are assigned rank n (worst).
    """
    rank = [n] * n
    for pos, doc_idx in enumerate(perm):
        if 0 <= doc_idx < n:
            rank[doc_idx] = pos
    return rank


def ndcg_at_k(predicted_perm: list[int], target_perm: list[int], k: int, n: int) -> float:
    """
    Compute nDCG@k.

    Relevance of document i is determined by its rank in `target_perm`:
        rel(i) = n - rank_target(i)   (so the top-ranked doc gets rel = n)

    DCG@k is computed over the first k positions of `predicted_perm`.
    IDCG@k is the maximum possible DCG (ideal ordering = target_perm[:k]).
    """
    if not target_perm or not predicted_perm:
        return 0.0

    # Relevance scores based on target ranking (higher rank → higher relevance)
    target_rank = permutation_to_ranking(target_perm, n)
    rel = [n - target_rank[i] for i in range(n)]   # rel[i] in [0, n]

    def dcg(ordering: list[int], cutoff: int) -> float:
        score = 0.0
        for pos, doc_idx in enumerate(ordering[:cutoff]):
            if 0 <= doc_idx < n:
                score += rel[doc_idx] / math.log2(pos + 2)   # log2(pos+2): pos is 0-indexed
        return score

    dcg_pred  = dcg(predicted_perm, k)
    # Ideal: order docs by descending relevance
    ideal_order = sorted(range(n), key=lambda i: -rel[i])
    dcg_ideal  = dcg(ideal_order, k)

    return dcg_pred / dcg_ideal if dcg_ideal > 0 else 0.0


def score_permutation_string(pred_str: str, target_str: str, n: int) -> dict:
    """
    Parse predicted and target permutation strings, return nDCG@1/5/10.
    """
    pred   = parse_permutation(pred_str)
    target = parse_permutation(target_str)
    return {
        "ndcg@1":  ndcg_at_k(pred, target, 1,  n),
        "ndcg@5":  ndcg_at_k(pred, target, 5,  n),
        "ndcg@10": ndcg_at_k(pred, target, 10, n),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: Path) -> list[dict]:
    examples = []
    with path.open() as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"  Loaded {len(examples):,} windowed examples from {path.name}")
    return examples


def parse_examples(raw: list[dict]) -> list[dict]:
    examples = []
    skipped  = 0

    for rec in raw:
        query        = rec["query"]
        target       = rec["target"]
        input_str    = rec["input"]
        qid          = rec["qid"]
        window_start = rec["window_start"]

        prefix = f"Query: {query} Document: "
        suffix = " Relevant Document:"
        body   = input_str
        if body.startswith(prefix):
            body = body[len(prefix):]
        if body.endswith(suffix):
            body = body[: -len(suffix)]

        parts = re.split(r'\[(\d+)\]', body)
        passage_texts = []
        i = 1
        while i < len(parts) - 1:
            passage_texts.append(parts[i + 1].strip())
            i += 2

        if not passage_texts:
            skipped += 1
            continue

        examples.append({
            "qid":           qid,
            "query":         query,
            "passage_texts": passage_texts,
            "target":        target,
            "window_start":  window_start,
        })

    if skipped:
        print(f"  WARNING: skipped {skipped} records (could not parse input string)")
    return examples


def train_val_split(
    examples: list[dict], val_ratio: float, seed: int
) -> tuple[list[dict], list[dict]]:
    rng  = random.Random(seed)
    qids = list({ex["qid"] for ex in examples})
    rng.shuffle(qids)
    n_val    = max(1, int(len(qids) * val_ratio))
    val_qids = set(qids[:n_val])
    train    = [ex for ex in examples if ex["qid"] not in val_qids]
    val      = [ex for ex in examples if ex["qid"] in val_qids]
    return train, val


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

QUERY_PREFIX = "Query:"
PASS_PREFIX  = "Document:"
SUFFIX       = " Relevant Document:"


class LiT5FiDDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer, max_length: int):
        self.examples   = examples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex     = self.examples[idx]
        query  = ex["query"]
        texts  = ex["passage_texts"]
        target = ex["target"]

        strings = [
            f"{QUERY_PREFIX} {query} {PASS_PREFIX} [{i + 1}] {text}{SUFFIX}"
            for i, text in enumerate(texts)
        ]
        enc = self.tokenizer(
            strings,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        lbl = self.tokenizer(
            target,
            max_length=MAX_NEW_TOKENS,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = lbl["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels":         labels,
            "target_str":     target,
            "n_passages":     len(texts),
        }


def collate_fn(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
        "target_strs":    [b["target_str"]  for b in batch],
        "n_passages":     [b["n_passages"]  for b in batch],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Label-smoothed cross-entropy  (fp32 stable)
# ─────────────────────────────────────────────────────────────────────────────

def smooth_ce(logits: torch.Tensor, labels: torch.Tensor, eps: float) -> torch.Tensor:
    vocab       = logits.size(-1)
    flat_logits = logits.view(-1, vocab)
    flat_labels = labels.view(-1)

    mask        = flat_labels != -100
    if mask.sum() == 0:
        return logits.new_tensor(0.0)

    flat_logits = flat_logits[mask]
    flat_labels = flat_labels[mask]
    flat_logits = flat_logits.float().clamp(-1e4, 1e4)

    log_probs = F.log_softmax(flat_logits, dim=-1)
    nll       = -log_probs.gather(1, flat_labels.unsqueeze(1)).squeeze(1)
    smooth    = -log_probs.mean(dim=-1)
    return ((1 - eps) * nll + eps * smooth).mean()


# ─────────────────────────────────────────────────────────────────────────────
# FiD forward
# ─────────────────────────────────────────────────────────────────────────────

def fid_encode(model, input_ids, attention_mask):
    B, n, L   = input_ids.shape
    flat_ids  = input_ids.view(B * n, L)
    flat_mask = attention_mask.view(B * n, L)

    enc_out = model.encoder(input_ids=flat_ids, attention_mask=flat_mask)
    hidden  = enc_out.last_hidden_state.view(B, n * L, -1)
    hidden  = hidden.clamp(-1e4, 1e4)
    attn    = flat_mask.view(B, n * L)
    return hidden, attn


def fid_forward_train(model, input_ids, attention_mask, labels):
    hidden, attn = fid_encode(model, input_ids, attention_mask)
    return model(
        encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
        attention_mask=attn,
        labels=labels,
    )


def fid_generate(model, input_ids, attention_mask):
    hidden, attn = fid_encode(model, input_ids, attention_mask)
    return model.generate(
        encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
        attention_mask=attn,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation: loss + nDCG@1/5/10
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_validation(
    model,
    tokenizer,
    val_examples: list[dict],
    device: torch.device,
    max_length: int,
    label_smoothing: float,
    use_bf16: bool,
) -> dict:
    """
    Compute validation loss AND nDCG@1/5/10 over all val windows.

    Returns a dict with keys:
        val_loss, ndcg@1, ndcg@5, ndcg@10
    """
    model.eval()
    val_ds     = LiT5FiDDataset(val_examples, tokenizer, max_length)
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    total_loss   = 0.0
    n_loss       = 0
    ndcg1_scores = []
    ndcg5_scores = []
    ndcg10_scores = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="  validation", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            target_strs    = batch["target_strs"]
            n_passages     = batch["n_passages"][0]

            # ── Loss ──────────────────────────────────────────────────────────
            with torch.amp.autocast("cuda", enabled=use_bf16, dtype=BF16):
                out  = fid_forward_train(model, input_ids, attention_mask, labels)
                loss = (
                    smooth_ce(out.logits, labels, label_smoothing)
                    if label_smoothing > 0
                    else out.loss
                )

            if torch.isfinite(loss):
                total_loss += loss.item()
                n_loss     += 1

            # ── Generation for nDCG ───────────────────────────────────────────
            with torch.amp.autocast("cuda", enabled=use_bf16, dtype=BF16):
                gen_ids = fid_generate(model, input_ids, attention_mask)

            pred_str = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            target_str = target_strs[0]

            scores = score_permutation_string(pred_str, target_str, n_passages)
            ndcg1_scores.append(scores["ndcg@1"])
            ndcg5_scores.append(scores["ndcg@5"])
            ndcg10_scores.append(scores["ndcg@10"])

    model.train()

    return {
        "val_loss": total_loss / max(1, n_loss),
        "ndcg@1":   float(np.mean(ndcg1_scores))  if ndcg1_scores  else 0.0,
        "ndcg@5":   float(np.mean(ndcg5_scores))  if ndcg5_scores  else 0.0,
        "ndcg@10":  float(np.mean(ndcg10_scores)) if ndcg10_scores else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint resume helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_last_checkpoint(out_dir: Path) -> tuple[Path | None, int]:
    """
    Scan out_dir for epoch checkpoints and return (path, epoch_number) of the
    latest one, or (None, 0) if none exist.

    Checkpoint dirs are named:  ep{N:02d}_ndcg5{X.XXXX}
    """
    pattern = re.compile(r'^ep(\d+)_ndcg5')
    candidates = []
    for d in out_dir.iterdir():
        if d.is_dir():
            m = pattern.match(d.name)
            if m:
                candidates.append((int(m.group(1)), d))
    if not candidates:
        return None, 0
    candidates.sort(key=lambda x: x[0])
    last_epoch, last_path = candidates[-1]
    return last_path, last_epoch


def load_training_state(results_txt: Path) -> list[dict]:
    """
    Re-parse the results text file to restore per-epoch history.
    Returns a list of dicts with keys: epoch, train_loss, val_loss, ndcg@1, ndcg@5, ndcg@10
    """
    history = []
    if not results_txt.exists():
        return history
    with results_txt.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Epoch"):
                continue
            parts = line.split("|")
            if len(parts) < 6:
                continue
            try:
                history.append({
                    "epoch":      int(parts[0].strip()),
                    "train_loss": float(parts[1].strip()),
                    "val_loss":   float(parts[2].strip()),
                    "ndcg@1":     float(parts[3].strip()),
                    "ndcg@5":     float(parts[4].strip()),
                    "ndcg@10":    float(parts[5].strip()),
                })
            except ValueError:
                continue
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def save_plots(history: list[dict], loss_img: Path, ndcg_img: Path, best_epoch: int):
    """
    Save two figures:
      1. training_loss.png  — train loss + val loss on the same axes
      2. ndcg_metrics.png   — nDCG@1, nDCG@5, nDCG@10 on the same axes with legend
    Both use seaborn darkgrid + ggplot styling.
    """
    epochs       = [h["epoch"]      for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses   = [h["val_loss"]   for h in history]
    ndcg1s       = [h["ndcg@1"]     for h in history]
    ndcg5s       = [h["ndcg@5"]     for h in history]
    ndcg10s      = [h["ndcg@10"]    for h in history]

    # ── Figure 1: Loss ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_losses, marker="o", linewidth=2,
            label="Train Loss",  color="#E05C5C")
    ax.plot(epochs, val_losses,   marker="s", linewidth=2,
            label="Val Loss",    color="#5C9EE0")
    if best_epoch:
        ax.axvline(best_epoch, linestyle="--", color="grey", alpha=0.7,
                   label=f"Best nDCG@5 (ep {best_epoch})")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss",  fontsize=12)
    ax.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(loss_img, dpi=150)
    plt.close(fig)
    print(f"  Loss curve saved : {loss_img}")

    # ── Figure 2: nDCG@1 / 5 / 10 ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, ndcg1s,  marker="^", linewidth=2, label="nDCG@1",  color="#E0A85C")
    ax.plot(epochs, ndcg5s,  marker="o", linewidth=2, label="nDCG@5",  color="#5CBF7A")
    ax.plot(epochs, ndcg10s, marker="s", linewidth=2, label="nDCG@10", color="#A05CE0")
    if best_epoch:
        ax.axvline(best_epoch, linestyle="--", color="grey", alpha=0.7,
                   label=f"Best nDCG@5 (ep {best_epoch})")
    ax.set_xlabel("Epoch",  fontsize=12)
    ax.set_ylabel("nDCG",   fontsize=12)
    ax.set_title("Validation nDCG@1 / @5 / @10", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(ndcg_img, dpi=150)
    plt.close(fig)
    print(f"  nDCG curve saved : {ndcg_img}")


# ─────────────────────────────────────────────────────────────────────────────
# Results text writer
# ─────────────────────────────────────────────────────────────────────────────

def write_results(history: list[dict], results_txt: Path, best_epoch: int,
                  best_ndcg5: float, skipped_steps: int):
    """Write / overwrite the results text file."""
    col_w = [6, 12, 10, 10, 10, 10]
    header = (
        f"{'Epoch':>{col_w[0]}} | "
        f"{'TrainLoss':>{col_w[1]}} | "
        f"{'ValLoss':>{col_w[2]}} | "
        f"{'nDCG@1':>{col_w[3]}} | "
        f"{'nDCG@5':>{col_w[4]}} | "
        f"{'nDCG@10':>{col_w[5]}}"
    )
    sep = "-" * len(header)

    with results_txt.open("w") as f:
        f.write("LiT5 Fine-tuning Results\n")
        f.write("=" * len(header) + "\n")
        f.write(header + "\n")
        f.write(sep + "\n")
        for h in history:
            f.write(
                f"{h['epoch']:>{col_w[0]}} | "
                f"{h['train_loss']:>{col_w[1]}.4f} | "
                f"{h['val_loss']:>{col_w[2]}.4f} | "
                f"{h['ndcg@1']:>{col_w[3]}.4f} | "
                f"{h['ndcg@5']:>{col_w[4]}.4f} | "
                f"{h['ndcg@10']:>{col_w[5]}.4f}\n"
            )
        f.write(sep + "\n")
        f.write(f"\nBest nDCG@5 : {best_ndcg5:.4f}  (epoch {best_epoch})\n")
        f.write(f"Total skipped optimizer steps : {skipped_steps}\n")
    print(f"  Results saved  : {results_txt}")


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda"
    print(f"Device  : {device}")
    if use_bf16:
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
        print(f"Dtype   : bfloat16 (autocast)")
    else:
        print("Dtype   : float32 (CPU — bfloat16 autocast disabled)")

    # ── Load dataset ───────────────────────────────────────────────────────────
    print(f"\nLoading windowed dataset from {DATASET_FILE} ...")
    raw      = load_dataset(DATASET_FILE)
    examples = parse_examples(raw)
    print(f"  Parsed {len(examples):,} examples")

    train_ex, val_ex = train_val_split(examples, args.val_split, args.seed)
    val_qids = {e["qid"] for e in val_ex}
    print(
        f"  Train : {len(train_ex):,} windows  |  "
        f"Val : {len(val_ex):,} windows ({len(val_qids)} queries)"
    )

    # ── Determine starting checkpoint (resume vs base) ─────────────────────────
    start_epoch = 0
    ckpt_source = str(args.checkpoint)

    if args.resume:
        last_ckpt, last_epoch = find_last_checkpoint(OUT_DIR)
        if last_ckpt is not None:
            ckpt_source = str(last_ckpt)
            start_epoch = last_epoch
            print(f"\n  Resuming from checkpoint : {last_ckpt.name} (epoch {last_epoch})")
        else:
            print("\n  No checkpoint found — starting from base model")

    # ── Restore history if resuming ────────────────────────────────────────────
    history = load_training_state(RESULTS_TXT) if args.resume else []
    if history:
        # Trim any history beyond start_epoch (handles partial resume)
        history = [h for h in history if h["epoch"] <= start_epoch]
        best_ndcg5  = max((h["ndcg@5"] for h in history), default=0.0)
        best_epoch  = max((h for h in history), key=lambda h: h["ndcg@5"],
                          default={"epoch": 0})["epoch"]
        # Patience: count consecutive non-improving epochs at the tail
        patience_count = 0
        for h in reversed(history):
            if h["ndcg@5"] < best_ndcg5 - 1e-4:
                patience_count += 1
            else:
                break
        print(
            f"  Restored history: {len(history)} epochs  |  "
            f"best nDCG@5={best_ndcg5:.4f} (ep {best_epoch})  |  "
            f"patience={patience_count}"
        )
    else:
        best_ndcg5     = 0.0
        best_epoch     = 0
        patience_count = 0

    # ── Model ──────────────────────────────────────────────────────────────────
    print(f"\nLoading model from {ckpt_source} ...")
    tokenizer = T5Tokenizer.from_pretrained(ckpt_source, legacy=False, use_fast=True)
    model     = T5ForConditionalGeneration.from_pretrained(ckpt_source).to(device)
    model.gradient_checkpointing_enable()

    # ── DataLoader ─────────────────────────────────────────────────────────────
    train_ds     = LiT5FiDDataset(train_ex, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )

    # ── Optimizer / scheduler ──────────────────────────────────────────────────
    optimizer    = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps  = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    print(f"\nTotal optimizer steps : {total_steps:,}  |  warmup : {warmup_steps}")

    # How many optimizer steps already done (for cosine schedule alignment)
    steps_done = math.ceil(len(train_loader) / args.grad_accum) * start_epoch

    def lr_lambda(step: int) -> float:
        step = step + steps_done   # shift by already-completed steps
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler     = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    global_step   = steps_done
    skipped_steps = 0

    print("\n" + "=" * 65)
    print(f"  LiT5 fine-tuning  |  epochs {start_epoch + 1}–{args.epochs}")
    print(f"  LR={args.lr}  WD={args.weight_decay}  GradAccum={args.grad_accum}")
    print(f"  Early-stop metric : nDCG@5  (patience={args.patience})")
    print(f"  Saving ALL checkpoints to {OUT_DIR}")
    print("=" * 65)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        epoch_loss  = 0.0
        nan_batches = 0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=use_bf16, dtype=BF16):
                out  = fid_forward_train(model, input_ids, attention_mask, labels)
                loss = (
                    smooth_ce(out.logits, labels, args.label_smoothing)
                    if args.label_smoothing > 0
                    else out.loss
                )

            if not torch.isfinite(loss):
                nan_batches += 1
                print(
                    f"\n  ⚠ NaN/Inf loss at epoch {epoch} step {step} — "
                    f"logits [{out.logits.min():.2f}, {out.logits.max():.2f}]"
                )
                optimizer.zero_grad()
                continue

            scaled = loss / args.grad_accum
            scaled.backward()
            epoch_loss += loss.item()

            if (step + 1) % args.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if torch.isfinite(grad_norm):
                    optimizer.step()
                    scheduler.step()
                else:
                    skipped_steps += 1
                    print(
                        f"\n  ⚠ Skipping optimizer step {global_step} — "
                        f"non-finite grad norm: {grad_norm:.2f}"
                    )
                optimizer.zero_grad()
                global_step += 1

        # ── Flush remaining gradients ──────────────────────────────────────────
        if len(train_loader) % args.grad_accum != 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(grad_norm):
                optimizer.step()
                scheduler.step()
            else:
                skipped_steps += 1
                print(f"\n  ⚠ Skipping final flush step — non-finite grad norm")
            optimizer.zero_grad()
            global_step += 1

        n_valid        = max(1, len(train_loader) - nan_batches)
        avg_train_loss = epoch_loss / n_valid

        # ── Validation ─────────────────────────────────────────────────────────
        val_metrics = evaluate_validation(
            model, tokenizer, val_ex, device,
            args.max_length, args.label_smoothing, use_bf16,
        )

        lr_now  = scheduler.get_last_lr()[0]
        nan_msg = f"  ({nan_batches} NaN batches skipped)" if nan_batches else ""
        print(
            f"\n  Epoch {epoch:3d} | "
            f"train={avg_train_loss:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"nDCG@1={val_metrics['ndcg@1']:.4f} | "
            f"nDCG@5={val_metrics['ndcg@5']:.4f} | "
            f"nDCG@10={val_metrics['ndcg@10']:.4f} | "
            f"lr={lr_now:.2e}{nan_msg}"
        )

        # ── Record history ─────────────────────────────────────────────────────
        history.append({
            "epoch":      epoch,
            "train_loss": avg_train_loss,
            "val_loss":   val_metrics["val_loss"],
            "ndcg@1":     val_metrics["ndcg@1"],
            "ndcg@5":     val_metrics["ndcg@5"],
            "ndcg@10":    val_metrics["ndcg@10"],
        })

        # ── Save checkpoint (ALL epochs) ───────────────────────────────────────
        ckpt_path = OUT_DIR / f"ep{epoch:02d}_ndcg5{val_metrics['ndcg@5']:.4f}"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        print(f"  Checkpoint saved : {ckpt_path.name}")

        # ── Early stopping on nDCG@5 ───────────────────────────────────────────
        if val_metrics["ndcg@5"] > best_ndcg5 + 1e-4:
            best_ndcg5     = val_metrics["ndcg@5"]
            best_epoch     = epoch
            patience_count = 0

            # Update "best" symlink
            best_link = OUT_DIR / "best"
            if best_link.is_symlink() or best_link.exists():
                best_link.unlink(missing_ok=True)
            try:
                best_link.symlink_to(ckpt_path.resolve())
            except OSError:
                # Symlinks may not be supported on all systems — copy instead
                import shutil
                if best_link.exists():
                    shutil.rmtree(best_link)
                shutil.copytree(ckpt_path, best_link)

            print(f"  ✓ New best nDCG@5={best_ndcg5:.4f} → {ckpt_path.name}")
        else:
            patience_count += 1
            print(
                f"  No improvement on nDCG@5  "
                f"({patience_count}/{args.patience})  |  best={best_ndcg5:.4f}"
            )
            if patience_count >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} (best epoch={best_epoch})")
                break

        # ── Overfit warning ────────────────────────────────────────────────────
        if len(history) >= 3:
            train_drop = history[-3]["train_loss"] - history[-1]["train_loss"]
            ndcg5_drop = history[-1]["ndcg@5"]     - history[-3]["ndcg@5"]
            if train_drop > 0.05 and ndcg5_drop < -0.01:
                print("  ⚠  Train loss dropping but nDCG@5 falling → watch for overfit")

        # ── Save plots + results after every epoch ────────────────────────────
        save_plots(history, LOSS_IMG, NDCG_IMG, best_epoch)
        write_results(history, RESULTS_TXT, best_epoch, best_ndcg5, skipped_steps)

    # ── Final report ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  Best nDCG@5      : {best_ndcg5:.4f}  (epoch {best_epoch})")
    print(f"  Best checkpoint  : {OUT_DIR}/best  (→ ep{best_epoch:02d}_*)")
    print(f"  Total skipped optimizer steps : {skipped_steps}")
    print(f"  Results          : {RESULTS_TXT}")
    print(f"  Loss plot        : {LOSS_IMG}")
    print(f"  nDCG plot        : {NDCG_IMG}")
    print("=" * 65)

    # Final save (in case the loop ended without triggering the in-loop save)
    save_plots(history, LOSS_IMG, NDCG_IMG, best_epoch)
    write_results(history, RESULTS_TXT, best_epoch, best_ndcg5, skipped_steps)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune LiT5-Distill on BioASQ (bfloat16, nDCG early-stop)"
    )
    p.add_argument("--checkpoint",      default=str(BASE_CKPT),
                   help="Path to base or starting checkpoint")
    p.add_argument("--resume",          action="store_true",
                   help="Auto-resume from the latest epoch checkpoint in OUT_DIR")
    p.add_argument("--max_length",      type=int,   default=TEXT_MAXLENGTH)
    p.add_argument("--lr",              type=float, default=LR)
    p.add_argument("--weight_decay",    type=float, default=WEIGHT_DECAY)
    p.add_argument("--warmup_ratio",    type=float, default=WARMUP_RATIO)
    p.add_argument("--epochs",          type=int,   default=EPOCHS)
    p.add_argument("--grad_accum",      type=int,   default=GRAD_ACCUM)
    p.add_argument("--patience",        type=int,   default=PATIENCE,
                   help="Early-stopping patience on nDCG@5")
    p.add_argument("--val_split",       type=float, default=VAL_SPLIT)
    p.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTH)
    p.add_argument("--seed",            type=int,   default=SEED)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)