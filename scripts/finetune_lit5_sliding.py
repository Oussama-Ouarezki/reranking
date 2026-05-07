"""
Fine-tune LiT5-Distill on BioASQ using DeepSeek sliding-window permutations.

Teacher  : DeepSeek (sliding-window prompt-2 reranking on BM25 top-50)
Student  : LiT5-Distill-base (FiD seq2seq listwise reranker)
Dataset  : data/bioasq/reranked/deepseek_sliding_reranked_prompt_2.jsonl
           Each line: {qid, bm25_order[50], permutation[50]}
           bm25_order  → model input order
           permutation → teacher target order

Method   : Listwise distillation with multiple INDEPENDENT windows per query
           (default window=20, stride=10). Each window is its own training
           example — no state carries between windows. For BM25 top-50:
             win0 = bm25[0:20], win1 = bm25[10:30], win2 = bm25[20:40], win3 = bm25[30:50]
           Target for each window = ordering of those positions by their rank
           in DeepSeek's full permutation.

           At evaluation, each window of a held-out query is reranked
           independently; the per-doc scores are summed across the windows
           it appears in to produce a single final ranking.

Quick check first:
    python scripts/finetune_lit5_sliding.py --limit 200 --epochs 3

Full run:
    python scripts/finetune_lit5_sliding.py --limit 0 --epochs 10
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
from ir_measures import nDCG, ScoredDoc, Qrel

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RERANKED_FILE = ROOT / "data/bioasq/reranked/deepseek_oracle_hybrid.jsonl"
FULL_CORPUS   = ROOT / "data/bioasq/pubmed_full/full/corpus_full_processed.jsonl"
QUERIES_FILE  = ROOT / "data/bioasq/processed/queries.jsonl"
QRELS_FILE    = ROOT / "data/bioasq/processed/qrels.tsv"
BASE_CKPT     = ROOT / "checkpoints/LiT5-Distill-base"
OUT_DIR       = ROOT / "checkpoints/lit5_finetune_sliding"
CURVE_IMG     = OUT_DIR / "training_curve.png"

# ── LiT5 prompt format ────────────────────────────────────────────────────────
QUERY_PREFIX = "Search Query:"
PASS_PREFIX  = "Passage:"
SUFFIX       = " Relevance Ranking:"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_corpus() -> dict[str, str]:
    corpus: dict[str, str] = {}
    with FULL_CORPUS.open() as f:
        for line in tqdm(f, desc="Loading corpus"):
            d = json.loads(line)
            did = d.get("_id") or d.get("id", "")
            corpus[did] = (d.get("title", "") + " " + d.get("text", "")).strip()
    print(f"  Corpus: {len(corpus):,} docs")
    return corpus


def load_queries() -> dict[str, str]:
    out: dict[str, str] = {}
    with QUERIES_FILE.open() as f:
        for line in f:
            q = json.loads(line)
            out[q["_id"]] = q["text"]
    return out


def load_qrels() -> list[Qrel]:
    qrels = []
    with QRELS_FILE.open() as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qrels.append(Qrel(parts[0], parts[1], int(parts[2])))
    return qrels


def load_reranked(limit: int) -> list[dict]:
    entries = []
    with RERANKED_FILE.open() as f:
        for line in f:
            entries.append(json.loads(line))
            if limit and len(entries) >= limit:
                break
    return entries


# ── Build training examples ───────────────────────────────────────────────────

def build_target(window_docids: list[str], permutation: list[str]) -> str:
    """Sort window positions by DeepSeek rank → '[3] > [1] > [2] > ...'."""
    perm_rank = {d: r for r, d in enumerate(permutation)}
    sorted_pos = sorted(
        range(len(window_docids)),
        key=lambda i: perm_rank.get(window_docids[i], len(permutation)),
    )
    return " > ".join(f"[{p + 1}]" for p in sorted_pos)


def window_starts(n_docs: int, window: int, stride: int) -> list[int]:
    """Independent window start positions covering [0, n_docs)."""
    if n_docs <= window:
        return [0]
    starts = list(range(0, n_docs - window + 1, stride))
    # Guarantee the tail is covered without truncating the last window.
    if starts[-1] + window < n_docs:
        starts.append(n_docs - window)
    return starts


def make_examples(
    entries: list[dict],
    queries: dict[str, str],
    corpus: dict[str, str],
    window: int,
    stride: int,
) -> list[dict]:
    """Multiple independent windows per query.

    Each window is one training example: input = BM25 docs in that slice,
    target = those positions ordered by their rank in DeepSeek's permutation.
    Windows do NOT share state — the model sees each as a standalone task.
    """
    examples = []
    skipped_missing_query = 0
    skipped_too_few_docs  = 0
    for e in entries:
        qid = e["qid"]
        if qid not in queries:
            skipped_missing_query += 1
            continue
        bm25_full = e["bm25_order"]
        perm      = e["permutation"]
        n_docs    = len(bm25_full)
        if n_docs == 0:
            continue
        for start in window_starts(n_docs, window, stride):
            slice_docids = bm25_full[start:start + window]
            texts = [corpus.get(d, "") for d in slice_docids]
            if sum(1 for t in texts if not t) > 0.3 * len(texts):
                skipped_too_few_docs += 1
                continue
            examples.append({
                "qid":           qid,
                "query":         queries[qid],
                "window_start":  start,
                "bm25_docids":   slice_docids,
                "passage_texts": texts,
                "target":        build_target(slice_docids, perm),
            })
    print(f"  Built {len(examples):,} window-examples "
          f"(skipped: missing_query={skipped_missing_query}, sparse_docs={skipped_too_few_docs})")
    return examples


def split_by_query(examples: list[dict], val_ratio: float, seed: int):
    rng = random.Random(seed)
    qids = list({ex["qid"] for ex in examples})
    rng.shuffle(qids)
    n_val = max(1, int(len(qids) * val_ratio))
    val_qids = set(qids[:n_val])
    train = [ex for ex in examples if ex["qid"] not in val_qids]
    val   = [ex for ex in examples if ex["qid"] in val_qids]
    return train, val


# ── Dataset ───────────────────────────────────────────────────────────────────

class FiDDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length, max_target_tokens):
        self.examples = examples
        self.tok = tokenizer
        self.max_length = max_length
        self.max_target = max_target_tokens

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        strings = [
            f"{QUERY_PREFIX} {ex['query']} {PASS_PREFIX} [{i + 1}] {t}{SUFFIX}"
            for i, t in enumerate(ex["passage_texts"])
        ]
        enc = self.tok(strings, max_length=self.max_length, padding="max_length",
                       truncation=True, return_tensors="pt")
        lbl = self.tok(ex["target"], max_length=self.max_target, padding="max_length",
                       truncation=True, return_tensors="pt")
        labels = lbl["input_ids"].squeeze(0)
        labels[labels == self.tok.pad_token_id] = -100
        return {
            "input_ids":      enc["input_ids"],       # [n, L]
            "attention_mask": enc["attention_mask"],  # [n, L]
            "labels":         labels,                 # [T]
        }


def collate(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
    }


# ── FiD forward ───────────────────────────────────────────────────────────────

def fid_encode(model, input_ids, attention_mask):
    """Encode each passage independently then concat → FiD."""
    B, n, L = input_ids.shape
    flat_ids  = input_ids.view(B * n, L)
    flat_mask = attention_mask.view(B * n, L)
    enc = model.encoder(input_ids=flat_ids, attention_mask=flat_mask)
    hidden = enc.last_hidden_state.view(B, n * L, -1)
    attn   = flat_mask.view(B, n * L)
    return hidden, attn


def smooth_ce(logits, labels, eps):
    vocab = logits.size(-1)
    flat_logits = logits.view(-1, vocab)
    flat_labels = labels.view(-1)
    mask = flat_labels != -100
    flat_logits = flat_logits[mask]
    flat_labels = flat_labels[mask]
    log_probs = F.log_softmax(flat_logits, dim=-1)
    nll    = -log_probs.gather(1, flat_labels.unsqueeze(1)).squeeze(1)
    smooth = -log_probs.mean(dim=-1)
    return ((1 - eps) * nll + eps * smooth).mean()


# ── Validation: nDCG@10 via single-window inference ───────────────────────────

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


def evaluate_ndcg(model, tokenizer, val_examples, qrels, device, max_length, max_new_tokens, amp_dtype):
    """Rerank each window independently, sum per-doc window scores, compute nDCG@10."""
    model.eval()
    val_qids = {ex["qid"] for ex in val_examples}
    use_amp = amp_dtype != torch.float32 and device.type == "cuda"

    # qid → {docid: aggregated_score}
    qid_scores: dict[str, dict[str, float]] = {}

    for ex in tqdm(val_examples, desc="  val", leave=False):
        strings = [
            f"{QUERY_PREFIX} {ex['query']} {PASS_PREFIX} [{i + 1}] {t}{SUFFIX}"
            for i, t in enumerate(ex["passage_texts"])
        ]
        enc = tokenizer(strings, max_length=max_length, padding="max_length",
                        truncation=True, return_tensors="pt").to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            hidden, attn = fid_encode(model, enc["input_ids"].unsqueeze(0),
                                      enc["attention_mask"].unsqueeze(0))
            out_ids = model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        perm_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        docids = ex["bm25_docids"]
        n = len(docids)
        order = parse_perm(perm_text, n)

        bucket = qid_scores.setdefault(ex["qid"], {})
        for new_rank, pos in enumerate(order):
            did = docids[pos]
            # Higher = better. Sum across overlapping windows.
            bucket[did] = bucket.get(did, 0.0) + float(n - new_rank)

    scored: list[ScoredDoc] = []
    for qid, bucket in qid_scores.items():
        for did, sc in bucket.items():
            scored.append(ScoredDoc(qid, did, sc))

    qrel_subset = [q for q in qrels if q.query_id in val_qids]
    if not qrel_subset or not scored:
        return 0.0
    res = ir_measures.calc_aggregate([nDCG @ 10], qrel_subset, scored)
    return float(dict(res).get(nDCG @ 10, 0.0))


# ── Train loop ────────────────────────────────────────────────────────────────

def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    print("\nLoading data …")
    corpus  = load_corpus()
    queries = load_queries()
    qrels   = load_qrels()
    entries = load_reranked(args.limit)
    print(f"  Loaded {len(entries)} reranked entries (limit={args.limit or 'ALL'})")

    print("\nBuilding examples …")
    examples = make_examples(entries, queries, corpus, args.window, args.stride)

    train_ex, val_ex = split_by_query(examples, args.val_split, args.seed)
    n_train_q = len({ex["qid"] for ex in train_ex})
    n_val_q   = len({ex["qid"] for ex in val_ex})
    print(f"  Train: {len(train_ex)} windows / {n_train_q} queries  "
          f"Val: {len(val_ex)} windows / {n_val_q} queries")

    print(f"\nLoading model from {args.checkpoint} …")
    tokenizer = T5Tokenizer.from_pretrained(args.checkpoint, legacy=False, use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint).to(device)
    model.gradient_checkpointing_enable()

    train_loader = DataLoader(
        FiDDataset(train_ex, tokenizer, args.max_length, args.max_target_tokens),
        batch_size=1, shuffle=True, collate_fn=collate, num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps  = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    print(f"Total optimizer steps: {total_steps} | warmup: {warmup_steps}")

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # T5 is numerically unstable in fp16 (NaN losses); use bf16 on Ampere+ GPUs.
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"AMP dtype: {amp_dtype}")

    best_ndcg = -1.0
    best_epoch = 0
    patience_count = 0
    train_losses, val_ndcgs = [], []
    saved_ckpts: list[tuple[float, Path]] = []

    print("\n" + "=" * 60)
    print(f"  LiT5 fine-tuning (independent windows) | epochs={args.epochs} | "
          f"window={args.window} | stride={args.stride}")
    print(f"  LR={args.lr} WD={args.weight_decay} GradAccum={args.grad_accum}")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_bf16):
                hidden, attn = fid_encode(model, input_ids, attention_mask)
                out = model(
                    encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
                    attention_mask=attn,
                    labels=labels,
                )
                loss = (smooth_ce(out.logits, labels, args.label_smoothing)
                        if args.label_smoothing > 0 else out.loss)
                loss = loss / args.grad_accum

            if not torch.isfinite(loss):
                print(f"  ⚠ non-finite loss at step {step}; skipping batch")
                optimizer.zero_grad()
                continue

            loss.backward()
            epoch_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        if len(train_loader) % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(1, len(train_loader))
        train_losses.append(avg_loss)

        print(f"\n  Epoch {epoch} | train loss={avg_loss:.4f}")
        val_ndcg = evaluate_ndcg(model, tokenizer, val_ex, qrels, device,
                                 args.max_length, args.max_target_tokens, amp_dtype)
        val_ndcgs.append(val_ndcg)
        print(f"  Epoch {epoch} | val nDCG@10={val_ndcg:.4f}")

        if val_ndcg > best_ndcg + 1e-4:
            best_ndcg = val_ndcg
            best_epoch = epoch
            patience_count = 0
            ckpt_path = OUT_DIR / f"ep{epoch:02d}_ndcg{val_ndcg:.4f}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            saved_ckpts.append((val_ndcg, ckpt_path))
            saved_ckpts.sort(key=lambda x: x[0], reverse=True)
            while len(saved_ckpts) > args.keep_top:
                _, old = saved_ckpts.pop()
                shutil.rmtree(old, ignore_errors=True)
                print(f"  Removed older ckpt: {old.name}")
            print(f"  ✓ best nDCG@10={best_ndcg:.4f} → saved {ckpt_path.name}")
        else:
            patience_count += 1
            print(f"  no improvement ({patience_count}/{args.patience})")
            if patience_count >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} (best={best_epoch})")
                break

    print("\n" + "=" * 60)
    print(f"  Best nDCG@10: {best_ndcg:.4f} (epoch {best_epoch})")
    if saved_ckpts:
        print(f"  Best ckpt   : {saved_ckpts[0][1]}")
    print("=" * 60)

    # Curve plot
    epochs_done = list(range(1, len(train_losses) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs_done, train_losses, marker="o", label="Train Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Training Loss"); ax1.legend()
    ax2.plot(epochs_done[: len(val_ndcgs)], val_ndcgs, marker="s", color="darkorange",
             label="Val nDCG@10")
    if best_epoch:
        ax2.axvline(best_epoch, linestyle="--", color="grey", alpha=0.6, label=f"best ep {best_epoch}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("nDCG@10"); ax2.set_title("Validation nDCG@10"); ax2.legend()
    plt.tight_layout()
    plt.savefig(CURVE_IMG, dpi=150)
    plt.close()
    print(f"  Curve saved : {CURVE_IMG}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",        default=str(BASE_CKPT))
    p.add_argument("--limit",             type=int,   default=2000,
                   help="use first N queries (0 = use all)")
    p.add_argument("--window",            type=int,   default=20)
    p.add_argument("--stride",            type=int,   default=10)
    p.add_argument("--max_length",        type=int,   default=350)
    p.add_argument("--max_target_tokens", type=int,   default=140)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--weight_decay",      type=float, default=0.01)
    p.add_argument("--warmup_ratio",      type=float, default=0.06)
    p.add_argument("--epochs",            type=int,   default=5)
    p.add_argument("--grad_accum",        type=int,   default=8)
    p.add_argument("--patience",          type=int,   default=3)
    p.add_argument("--val_split",         type=float, default=0.2)
    p.add_argument("--label_smoothing",   type=float, default=0.1)
    p.add_argument("--keep_top",          type=int,   default=2)
    p.add_argument("--seed",              type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
