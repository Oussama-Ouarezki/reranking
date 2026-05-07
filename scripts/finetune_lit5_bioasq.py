"""
Fine-tune LiT5-Distill on BioASQ using oracle-hybrid permutations as teacher signal.

Dataset  : data/bioasq/reranked/deepseek_oracle_hybrid.jsonl
           2 000 training queries; bm25_order is model input, permutation has
           gold-relevant docs first (in DeepSeek order) then non-relevant.

Method   : Supervised listwise distillation.
           Rolling-window approach (window=20, stride=10, top-50) generates up to
           5 independent training examples per query → ~10 000 total examples.

Input    : FiD-encoded (query, passage_i) pairs, n=20 passages (in BM25 order)
Target   : permutation string "[j] > [k] > ..." — positions sorted by hybrid rank

Anti-overfitting:
  - 80/20 train/val split stratified by query
  - Weight decay 0.01, label smoothing 0.1
  - Gradient checkpointing (saves ~4 GB)
  - Early stopping on validation nDCG@10 (patience=3)
  - Cosine LR schedule with warmup
  - Mixed precision (fp16) for RTX 3060 12 GB

Usage:
    python scripts/finetune_lit5_bioasq.py
    python scripts/finetune_lit5_bioasq.py --lr 5e-5 --epochs 15 --window 20 --stride 10
"""

import argparse
import json
import math
import random
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
from ir_measures import nDCG, ScoredDoc, Qrel

# ── Plotting theme ────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RERANKED_FILE  = ROOT / "data/bioasq/reranked/deepseek_oracle_hybrid.jsonl"
FULL_CORPUS    = ROOT / "data/bioasq/pubmed_full/full/corpus_full_processed.jsonl"
QUERIES_FILE   = ROOT / "data/bioasq/processed/queries.jsonl"
QRELS_FILE     = ROOT / "data/bioasq/processed/qrels.tsv"
BASE_CKPT      = ROOT / "checkpoints/LiT5-Distill-base"
OUT_DIR        = ROOT / "checkpoints/lit5_finetune_oracle"
CURVE_IMG      = OUT_DIR / "training_curve.png"

MAX_SAVED_CKPTS = 2   # keep only the 2 best checkpoints to save disk space

# ── Default hyperparameters ───────────────────────────────────────────────────
WINDOW_SIZE    = 20
STRIDE         = 10
TOP_K          = 50
TEXT_MAXLENGTH = 350
MAX_NEW_TOKENS = 140
LR             = 1e-4
WEIGHT_DECAY   = 0.01
WARMUP_RATIO   = 0.06
EPOCHS         = 15
GRAD_ACCUM     = 8      # effective batch = 8 queries
PATIENCE       = 3      # early stopping epochs without improvement
VAL_SPLIT      = 0.20
LABEL_SMOOTH   = 0.1
SEED           = 42
VAL_EVAL_K     = 10     # nDCG@10 for early stopping

QUERY_PREFIX = "Search Query:"
PASS_PREFIX  = "Passage:"
SUFFIX       = " Relevance Ranking:"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_corpus() -> dict[str, str]:
    """Load corpus_full_processed into {docid: title+text}."""
    corpus: dict[str, str] = {}
    with FULL_CORPUS.open() as f:
        for line in tqdm(f, desc="Loading corpus"):
            d = json.loads(line)
            did = d.get("_id") or d.get("id", "")
            corpus[did] = (d.get("title", "") + " " + d.get("text", "")).strip()
    print(f"  Corpus: {len(corpus):,} documents loaded")
    return corpus


def load_queries() -> dict[str, str]:
    queries: dict[str, str] = {}
    with QUERIES_FILE.open() as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    return queries


def load_qrels() -> list[Qrel]:
    qrels = []
    with QRELS_FILE.open() as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, did, score = parts[0], parts[1], parts[2]
                qrels.append(Qrel(qid, did, int(score)))
    return qrels


def load_reranked() -> list[dict]:
    entries = []
    with RERANKED_FILE.open() as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


# ── Window generation ─────────────────────────────────────────────────────────

def build_target(window_docids: list[str], permutation: list[str]) -> str:
    """Derive permutation target string from DeepSeek ranking within a window.

    window_docids : docs in BM25 order within the window (input order)
    permutation   : full 50-doc DeepSeek ranking (best first)
    Returns "[j] > [k] > ..." where j,k are 1-based positions in window_docids
    """
    perm_rank = {docid: rank for rank, docid in enumerate(permutation)}
    # Sort window positions by DeepSeek rank (lower = more relevant)
    sorted_pos = sorted(
        range(len(window_docids)),
        key=lambda i: perm_rank.get(window_docids[i], len(permutation)),
    )
    return " > ".join(f"[{p + 1}]" for p in sorted_pos)


def generate_examples(
    entries: list[dict],
    queries: dict[str, str],
    corpus: dict[str, str],
    window_size: int,
    stride: int,
    top_k: int,
) -> list[dict]:
    """Generate all sliding-window training examples.

    Each example: {qid, query, passage_texts, target, window_start}
    passage_texts: list of window_size strings (BM25 order within window)
    target       : "[j] > [k] > ..." permutation string
    """
    examples = []
    missing_docs = 0
    missing_queries = 0

    for entry in entries:
        qid = entry["qid"]
        if qid not in queries:
            missing_queries += 1
            continue
        query = queries[qid]
        bm25_order = entry["bm25_order"][:top_k]
        permutation = entry["permutation"][:top_k]

        # Rolling window over BM25 order
        start = 0
        while start < len(bm25_order):
            end = min(start + window_size, len(bm25_order))
            window_docids = bm25_order[start:end]

            texts = []
            for did in window_docids:
                text = corpus.get(did, "")
                if not text:
                    missing_docs += 1
                texts.append(text)

            # Skip window if more than 30% of docs are missing
            if sum(1 for t in texts if not t) > 0.3 * len(texts):
                if end == len(bm25_order):
                    break
                start += stride
                continue

            target = build_target(window_docids, permutation)
            examples.append(
                {
                    "qid": qid,
                    "query": query,
                    "passage_texts": texts,
                    "target": target,
                    "window_start": start,
                }
            )
            if end == len(bm25_order):
                break
            start += stride

    print(f"  Generated {len(examples):,} windows | "
          f"missing_queries={missing_queries} missing_doc_slots={missing_docs}")
    return examples


def train_val_split(
    examples: list[dict], val_ratio: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Split by query ID so all windows for a query go to the same partition."""
    rng = random.Random(seed)
    qids = list({ex["qid"] for ex in examples})
    rng.shuffle(qids)
    n_val = max(1, int(len(qids) * val_ratio))
    val_qids = set(qids[:n_val])
    train = [ex for ex in examples if ex["qid"] not in val_qids]
    val   = [ex for ex in examples if ex["qid"] in val_qids]
    return train, val


# ── Dataset ───────────────────────────────────────────────────────────────────

class LiT5FiDDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer, max_length: int):
        self.examples   = examples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        query  = ex["query"]
        texts  = ex["passage_texts"]
        target = ex["target"]

        # Build FiD input strings — one per passage
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
            "input_ids":       enc["input_ids"],       # [window_size, max_length]
            "attention_mask":  enc["attention_mask"],  # [window_size, max_length]
            "labels":          labels,                 # [target_len]
        }


def collate_fn(batch):
    """Stack items into batch tensors; each item already has fixed-size tensors."""
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
    }


# ── FiD forward pass ──────────────────────────────────────────────────────────

def fid_forward(model, input_ids, attention_mask, labels=None):
    """Fusion-in-Decoder forward.

    input_ids      : [B, n, L]
    attention_mask : [B, n, L]
    labels         : [B, T]  or None (inference)

    Returns loss (train) or generated ids (inference).
    """
    B, n, L = input_ids.shape
    # Encode all passages independently
    flat_ids  = input_ids.view(B * n, L)
    flat_mask = attention_mask.view(B * n, L)
    encoder_out = model.encoder(input_ids=flat_ids, attention_mask=flat_mask)
    # Concatenate: [B, n*L, hidden]
    hidden = encoder_out.last_hidden_state.view(B, n * L, -1)
    attn   = flat_mask.view(B, n * L)

    if labels is not None:
        out = model(
            encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
            attention_mask=attn,
            labels=labels,
        )
        return out.loss
    else:
        return model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
            attention_mask=attn,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )


# ── Label smoothing loss ──────────────────────────────────────────────────────

def smooth_ce(logits: torch.Tensor, labels: torch.Tensor, eps: float) -> torch.Tensor:
    """Cross-entropy with label smoothing, ignoring -100 labels."""
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


# ── nDCG evaluation ───────────────────────────────────────────────────────────

def parse_ranking(perm_text: str, n: int) -> list[int]:
    nums, seen = [], set()
    for tok in perm_text.replace(",", " ").replace(">", " ").split():
        try:
            v = int(tok.strip("[]()."))
            if 1 <= v <= n and (v - 1) not in seen:
                nums.append(v - 1)
                seen.add(v - 1)
        except ValueError:
            continue
    for i in range(n):
        if i not in seen:
            nums.append(i)
    return nums


def evaluate_ndcg(
    model,
    tokenizer,
    val_entries: list[dict],
    queries: dict[str, str],
    corpus: dict[str, str],
    qrels: list[Qrel],
    device,
    window_size: int,
    stride: int,
    top_k: int,
    max_length: int,
) -> float:
    """Run full sliding-window inference on val queries and compute nDCG@10."""
    model.eval()
    val_qids = {e["qid"] for e in val_entries}
    metric   = nDCG @ VAL_EVAL_K
    scored   = []

    entry_map = {e["qid"]: e for e in val_entries}   # build once, not per query
    for qid in tqdm(val_qids, desc="  val nDCG", leave=False):
        if qid not in queries:
            continue
        query = queries[qid]

        if qid not in entry_map:
            continue
        bm25_order = entry_map[qid]["bm25_order"][:top_k]
        ranked = [(did, corpus.get(did, "")) for did in bm25_order]

        # Sliding window (same as inference, tail-to-head)
        n     = len(ranked)
        start = max(0, n - window_size)
        while True:
            end    = min(start + window_size, n)
            window = ranked[start:end]
            strings = [
                f"{QUERY_PREFIX} {query} {PASS_PREFIX} [{i + 1}] {text}{SUFFIX}"
                for i, (_, text) in enumerate(window)
            ]
            enc = tokenizer(
                strings,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out = fid_forward(
                    model,
                    enc["input_ids"].unsqueeze(0),
                    enc["attention_mask"].unsqueeze(0),
                )
            perm_text = tokenizer.decode(out[0], skip_special_tokens=True)
            perm      = parse_ranking(perm_text, len(window))
            ranked[start:end] = [window[i] for i in perm]
            if start == 0:
                break
            start = max(0, start - stride)

        for rank, (did, _) in enumerate(ranked):
            scored.append(ScoredDoc(qid, did, float(n - rank)))

    if not scored:
        return 0.0
    qrel_subset = [q for q in qrels if q.query_id in val_qids]
    results     = ir_measures.calc_aggregate([metric], qrel_subset, scored)
    return float(dict(results).get(metric, 0.0))


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\nLoading corpus …")
    corpus  = load_corpus()
    print("Loading queries …")
    queries = load_queries()
    print("Loading qrels …")
    qrels   = load_qrels()
    print("Loading reranked entries …")
    entries = load_reranked()

    # ── Build examples ─────────────────────────────────────────────────────────
    print("\nGenerating sliding-window examples …")
    examples = generate_examples(
        entries, queries, corpus, args.window, args.stride, args.top_k
    )
    train_ex, val_ex = train_val_split(examples, args.val_split, args.seed)

    # For nDCG evaluation we need one entry per val query (not per window)
    val_qids   = {ex["qid"] for ex in val_ex}
    val_entries = [e for e in entries if e["qid"] in val_qids]

    print(f"  Train examples : {len(train_ex):,}")
    print(f"  Val   examples : {len(val_ex):,}  ({len(val_qids)} queries)")

    # ── Model ──────────────────────────────────────────────────────────────────
    print(f"\nLoading model from {args.checkpoint} …")
    tokenizer = T5Tokenizer.from_pretrained(
        args.checkpoint, legacy=False, use_fast=True
    )
    model = T5ForConditionalGeneration.from_pretrained(
        args.checkpoint,
    ).to(device)
    model.gradient_checkpointing_enable()

    # ── DataLoader ─────────────────────────────────────────────────────────────
    train_ds = LiT5FiDDataset(train_ex, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    # val_loader unused — we run sliding-window inference directly

    # ── Optimizer / scheduler ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps   = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    warmup_steps  = max(1, int(total_steps * args.warmup_ratio))
    print(f"\nTotal optimizer steps : {total_steps:,}  |  warmup : {warmup_steps}")

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── Training state ─────────────────────────────────────────────────────────
    best_ndcg      = -1.0
    best_epoch     = 0
    patience_count = 0
    global_step    = 0

    train_losses:    list[float] = []
    val_ndcgs:       list[float] = []
    saved_ckpts:     list[tuple[float, Path]] = []   # (ndcg, path) sorted best-first

    print("\n" + "=" * 60)
    print(f"  LiT5 fine-tuning  |  {args.epochs} epochs  |  window={args.window} stride={args.stride}")
    print(f"  LR={args.lr}  WD={args.weight_decay}  GradAccum={args.grad_accum}")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss   = 0.0
        epoch_tokens = 0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            input_ids      = batch["input_ids"].to(device)       # [1, n, L]
            attention_mask = batch["attention_mask"].to(device)  # [1, n, L]
            labels         = batch["labels"].to(device)          # [1, T]

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                # FiD encode → get logits for label smoothing
                B, n, L = input_ids.shape
                flat_ids   = input_ids.view(B * n, L)
                flat_mask  = attention_mask.view(B * n, L)
                encoder_out = model.encoder(input_ids=flat_ids, attention_mask=flat_mask)
                hidden = encoder_out.last_hidden_state.view(B, n * L, -1)
                attn   = flat_mask.view(B, n * L)

                out = model(
                    encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
                    attention_mask=attn,
                    labels=labels,
                )
                if args.label_smoothing > 0:
                    loss = smooth_ce(out.logits, labels, args.label_smoothing)
                else:
                    loss = out.loss
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()

            n_toks       = (labels != -100).sum().item()
            epoch_loss  += loss.item() * args.grad_accum
            epoch_tokens += n_toks

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        # Flush any remaining gradients if loader length is not divisible by grad_accum
        if (len(train_loader)) % args.grad_accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        avg_loss = epoch_loss / max(1, len(train_loader))
        train_losses.append(avg_loss)

        # ── Validation nDCG ────────────────────────────────────────────────────
        print(f"\n  Epoch {epoch} — train loss: {avg_loss:.4f}")
        print("  Running validation nDCG@10 …")
        val_ndcg = evaluate_ndcg(
            model, tokenizer, val_entries, queries, corpus, qrels,
            device, args.window, args.stride, args.top_k, args.max_length,
        )
        val_ndcgs.append(val_ndcg)
        model.train()

        lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else args.lr
        print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} | val nDCG@10={val_ndcg:.4f} | lr={lr_now:.2e}")

        # ── Overfit check ──────────────────────────────────────────────────────
        if len(train_losses) >= 3:
            recent_loss_drop = train_losses[-3] - train_losses[-1]
            recent_ndcg_gain = val_ndcgs[-1] - val_ndcgs[-3]
            if recent_loss_drop > 0.05 and recent_ndcg_gain < 0.001:
                print("  ⚠  Train loss dropping fast but val nDCG stagnant → possible overfit")

        # ── Early stopping + storage-aware checkpoint rotation ─────────────────
        if val_ndcg > best_ndcg + 1e-4:
            best_ndcg      = val_ndcg
            best_epoch     = epoch
            patience_count = 0

            ckpt_path = OUT_DIR / f"ep{epoch:02d}_ndcg{val_ndcg:.4f}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            saved_ckpts.append((val_ndcg, ckpt_path))

            # Keep only the top MAX_SAVED_CKPTS by nDCG; delete the worst
            saved_ckpts.sort(key=lambda x: x[0], reverse=True)
            while len(saved_ckpts) > MAX_SAVED_CKPTS:
                _, old_path = saved_ckpts.pop()   # lowest nDCG
                import shutil
                shutil.rmtree(old_path, ignore_errors=True)
                print(f"  🗑  Deleted old checkpoint {old_path.name} to save space")

            print(f"  ✓ New best nDCG@10={best_ndcg:.4f} — saved to {ckpt_path.name}")
            print(f"     Kept checkpoints: {[p.name for _, p in saved_ckpts]}")
        else:
            patience_count += 1
            print(f"  No improvement ({patience_count}/{args.patience})")
            if patience_count >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} (best epoch={best_epoch})")
                break

    # ── Final report ───────────────────────────────────────────────────────────
    best_saved = saved_ckpts[0][1] if saved_ckpts else OUT_DIR
    print("\n" + "=" * 60)
    print(f"  Best nDCG@10 : {best_ndcg:.4f}  (epoch {best_epoch})")
    print(f"  Best ckpt    : {best_saved}")
    print(f"  Saved ckpts  : {[p.name for _, p in saved_ckpts]}")
    print("=" * 60)

    # ── Training curve ─────────────────────────────────────────────────────────
    epochs_done = list(range(1, len(train_losses) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs_done, train_losses, marker="o", label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()

    ax2.plot(epochs_done[: len(val_ndcgs)], val_ndcgs, marker="s", color="darkorange",
             label=f"Val nDCG@{VAL_EVAL_K}")
    if best_epoch:
        ax2.axvline(best_epoch, linestyle="--", color="grey", alpha=0.6, label=f"Best (ep {best_epoch})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(f"nDCG@{VAL_EVAL_K}")
    ax2.set_title(f"Validation nDCG@{VAL_EVAL_K}")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(CURVE_IMG, dpi=150)
    print(f"  Curve saved  : {CURVE_IMG}")
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune LiT5-Distill on BioASQ")
    p.add_argument("--checkpoint",      default=str(BASE_CKPT))
    p.add_argument("--window",          type=int,   default=WINDOW_SIZE)
    p.add_argument("--stride",          type=int,   default=STRIDE)
    p.add_argument("--top_k",           type=int,   default=TOP_K)
    p.add_argument("--max_length",      type=int,   default=TEXT_MAXLENGTH)
    p.add_argument("--lr",              type=float, default=LR)
    p.add_argument("--weight_decay",    type=float, default=WEIGHT_DECAY)
    p.add_argument("--warmup_ratio",    type=float, default=WARMUP_RATIO)
    p.add_argument("--epochs",          type=int,   default=EPOCHS)
    p.add_argument("--grad_accum",      type=int,   default=GRAD_ACCUM)
    p.add_argument("--patience",        type=int,   default=PATIENCE)
    p.add_argument("--val_split",       type=float, default=VAL_SPLIT)
    p.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTH)
    p.add_argument("--seed",            type=int,   default=SEED)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
