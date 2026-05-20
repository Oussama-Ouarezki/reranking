"""
Fine-tune LiT5-Distill on BioASQ — TOP-20 ONLY, high-accuracy configuration.
... (docstring truncated for brevity) ...
Updates:
  - Implements LiT5's weighted cross-entropy loss with exponential decay (λ=0.95).
  - Per-token weight = λ^(position_index) for position_index=0,1,2,... in the target sequence.
"""

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
from tqdm import tqdm

try:
    import ir_measures
    from ir_measures import nDCG, ScoredDoc, Qrel
    HAS_IR_MEASURES = True
except ImportError:
    HAS_IR_MEASURES = False
    print("⚠  ir_measures not found — validation will use proxy top-1 accuracy instead of nDCG.")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parent.parent
RERANKED_FILE  = ROOT / "data/bioasq/reranked/gold_front_listwise.jsonl"
FULL_CORPUS    = ROOT / "data/bioasq/pubmed_full/full/corpus_full_processed.jsonl"
QUERIES_FILE   = ROOT / "data/bioasq/processed/queries.jsonl"
QRELS_FILE     = ROOT / "data/bioasq/processed/qrels.tsv"
BASE_CKPT      = ROOT / "checkpoints/LiT5-Distill-base"
OUT_DIR        = ROOT / "checkpoints/lit5_top20_oracle"
CURVE_IMG      = OUT_DIR / "training_curve.png"

MAX_SAVED_CKPTS = 6

# ── Hyperparameters ────────────────────────────────────────────────────────────
TOP_K = 20
WINDOW_SIZE = 20

TEXT_MAXLENGTH = 350
MAX_NEW_TOKENS = 140

LR             = 5e-5
WEIGHT_DECAY   = 0.01
WARMUP_RATIO   = 0.10
EPOCHS         = 10
GRAD_ACCUM     = 16
PATIENCE       = 5
VAL_SPLIT      = 0.15
LABEL_SMOOTH   = 0.05
SEED           = 42
VAL_EVAL_K     = 5
DECAY_FACTOR   = 0.95          # LiT5 exponential decay weight (λ)

# Input format strings
QUERY_PREFIX   = "Search Query:"
PASS_PREFIX    = "Passage:"
SUFFIX         = " Relevance Ranking:"


# ── Data loading ──────────────────────────────────────────────────────────────
# ... (all data loading functions remain identical) ...
def load_corpus() -> dict[str, str]:
    corpus: dict[str, str] = {}
    with FULL_CORPUS.open() as f:
        for line in tqdm(f, desc="Loading corpus"):
            d = json.loads(line)
            did   = d.get("_id") or d.get("id", "")
            title = d.get("title", "")
            text  = d.get("text", "")
            corpus[did] = (title + " " + text).strip()
    print(f"  Corpus: {len(corpus):,} documents")
    return corpus

def load_queries() -> dict[str, str]:
    queries: dict[str, str] = {}
    with QUERIES_FILE.open() as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    return queries

def load_qrels() -> list:
    if not HAS_IR_MEASURES:
        return []
    qrels = []
    with QRELS_FILE.open() as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qrels.append(Qrel(parts[0], parts[1], int(parts[2])))
    return qrels

def load_reranked() -> list[dict]:
    entries = []
    with RERANKED_FILE.open() as f:
        for line in f:
            entries.append(json.loads(line))
    print(f"  Reranked entries: {len(entries):,}")
    return entries


# ── Example building ──────────────────────────────────────────────────────────
def build_target(window_docids: list[str], permutation: list[str]) -> str:
    perm_rank = {docid: rank for rank, docid in enumerate(permutation)}
    sorted_pos = sorted(
        range(len(window_docids)),
        key=lambda i: perm_rank.get(window_docids[i], len(permutation)),
    )
    return " > ".join(f"[{p + 1}]" for p in sorted_pos)

def generate_examples(
    entries: list[dict],
    queries: dict[str, str],
    corpus: dict[str, str],
    top_k: int,
) -> list[dict]:
    examples = []
    skipped_missing_query = 0
    skipped_missing_docs  = 0

    for entry in entries:
        qid = entry["qid"]
        if qid not in queries:
            skipped_missing_query += 1
            continue
        query       = queries[qid]
        bm25_order  = entry["bm25_order"][:top_k]
        permutation = entry["permutation"][:top_k]
        if len(bm25_order) < top_k:
            bm25_order = bm25_order + ["__PAD__"] * (top_k - len(bm25_order))
        texts = []
        n_missing = 0
        for did in bm25_order:
            t = corpus.get(did, "")
            if not t:
                n_missing += 1
            texts.append(t)
        if n_missing > 0.30 * top_k:
            skipped_missing_docs += 1
            continue
        target = build_target(bm25_order, permutation)
        examples.append({
            "qid":           qid,
            "query":         query,
            "passage_texts": texts,
            "target":        target,
        })

    print(
        f"  Examples built: {len(examples):,} | "
        f"skipped (no query)={skipped_missing_query} "
        f"skipped (missing docs)={skipped_missing_docs}"
    )
    return examples

def train_val_split(
    examples: list[dict], val_ratio: float, seed: int
) -> tuple[list[dict], list[dict]]:
    rng   = random.Random(seed)
    qids  = list({ex["qid"] for ex in examples})
    rng.shuffle(qids)
    n_val    = max(1, int(len(qids) * val_ratio))
    val_qids = set(qids[:n_val])
    train = [ex for ex in examples if ex["qid"] not in val_qids]
    val   = [ex for ex in examples if ex["qid"] in     val_qids]
    return train, val


# ── Dataset ───────────────────────────────────────────────────────────────────
class LiT5Top20Dataset(Dataset):
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
        }

def collate_fn(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
    }


# ── Sorted sampler ────────────────────────────────────────────────────────────
class SortedSampler(Sampler):
    def __init__(self, examples: list[dict]):
        self._order = sorted(
            range(len(examples)),
            key=lambda i: sum(len(t) for t in examples[i]["passage_texts"]),
            reverse=True,
        )
    def __iter__(self):
        return iter(self._order)
    def __len__(self):
        return len(self._order)


# ── FiD forward pass ──────────────────────────────────────────────────────────
def fid_forward(model, input_ids, attention_mask, labels=None):
    B, n, L = input_ids.shape
    flat_ids  = input_ids.view(B * n, L)
    flat_mask = attention_mask.view(B * n, L)
    encoder_out = model.encoder(input_ids=flat_ids, attention_mask=flat_mask)
    hidden = encoder_out.last_hidden_state.view(B, n * L, -1)
    attn   = flat_mask.view(B, n * L)
    if labels is not None:
        return model(
            encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
            attention_mask=attn,
            labels=labels,
        )
    else:
        return model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
            attention_mask=attn,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
        )


# ── NEW: Weighted label-smoothed cross-entropy with exponential decay ─────────
def weighted_smooth_ce(logits, labels, eps, decay_factor):
    """
    Weighted label-smoothed cross-entropy.
    Weight for token at position t = decay_factor ** t (t=0,1,2,...).
    Tokens with label = -100 (padding) are masked out.
    Returns scalar loss (weighted mean over active tokens).
    """
    vocab = logits.size(-1)
    B, T = labels.shape
    device = logits.device

    # Positional weights: λ^0, λ^1, ..., λ^(T-1)
    pos = torch.arange(T, device=device).float()
    weights = decay_factor ** pos               # [T]
    weights = weights.unsqueeze(0).expand(B, T)  # [B, T]

    # Mask padding
    mask = (labels != -100).float()
    weights = weights * mask

    flat_logits = logits.view(-1, vocab)
    flat_labels = labels.view(-1)
    flat_weights = weights.view(-1)

    # Keep only active tokens
    active = flat_weights > 0
    active_logits = flat_logits[active]
    active_labels = flat_labels[active]
    active_weights = flat_weights[active]

    log_probs = F.log_softmax(active_logits, dim=-1)
    nll = -log_probs.gather(1, active_labels.unsqueeze(1)).squeeze(1)

    if eps > 0:
        smooth = -log_probs.mean(dim=-1)
        loss_per_token = (1 - eps) * nll + eps * smooth
    else:
        loss_per_token = nll

    # Weighted average
    loss = (loss_per_token * active_weights).sum() / active_weights.sum()
    return loss


# ── Permutation parser ────────────────────────────────────────────────────────
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

def top1_accuracy(pred_perm: str, gold_perm: str, n: int) -> float:
    pred = parse_ranking(pred_perm, n)
    gold = parse_ranking(gold_perm, n)
    return float(pred[0] == gold[0]) if pred and gold else 0.0


# ── nDCG@K evaluation ─────────────────────────────────────────────────────────
def evaluate_ndcg(
    model, tokenizer, val_examples, entry_map, qrels,
    device, max_length, top_k, val_eval_k,
) -> float:
    model.eval()
    if HAS_IR_MEASURES:
        metric = nDCG @ val_eval_k
        scored = []
    else:
        accs = []

    with torch.no_grad():
        for ex in tqdm(val_examples, desc="  val eval", leave=False):
            qid = ex["qid"]
            query = ex["query"]
            texts = ex["passage_texts"]
            strings = [
                f"{QUERY_PREFIX} {query} {PASS_PREFIX} [{i + 1}] {text}{SUFFIX}"
                for i, text in enumerate(texts)
            ]
            enc = tokenizer(
                strings, max_length=max_length, padding="max_length",
                truncation=True, return_tensors="pt",
            ).to(device)
            out = fid_forward(
                model,
                enc["input_ids"].unsqueeze(0),
                enc["attention_mask"].unsqueeze(0),
            )
            perm_text = tokenizer.decode(out[0], skip_special_tokens=True)
            perm = parse_ranking(perm_text, len(texts))

            if HAS_IR_MEASURES:
                entry = entry_map.get(qid, {})
                bm25_order = entry.get("bm25_order", [])[:top_k]
                ranked_ids = [bm25_order[i] for i in perm if i < len(bm25_order)]
                n_docs = len(bm25_order)
                for rank, did in enumerate(ranked_ids):
                    scored.append(ScoredDoc(qid, did, float(n_docs - rank)))
            else:
                accs.append(top1_accuracy(perm_text, ex["target"], len(texts)))

    if HAS_IR_MEASURES:
        if not scored:
            return 0.0
        val_qids = {ex["qid"] for ex in val_examples}
        qrel_sub = [q for q in qrels if q.query_id in val_qids]
        results = ir_measures.calc_aggregate([metric], qrel_sub, scored)
        return float(dict(results).get(metric, 0.0))
    else:
        return float(np.mean(accs)) if accs else 0.0


# ── Training loop ─────────────────────────────────────────────────────────────
def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    if device.type == "cuda":
        use_bf16 = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        print(f"AMP    : {'bfloat16' if use_bf16 else 'float16'}")
    else:
        amp_dtype = torch.float32

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\nLoading corpus …")
    corpus  = load_corpus()
    print("Loading queries …")
    queries = load_queries()
    print("Loading qrels …")
    qrels   = load_qrels()
    print("Loading reranked entries …")
    entries = load_reranked()
    entry_map = {e["qid"]: e for e in entries}

    # ── Build examples ─────────────────────────────────────────────────────────
    print(f"\nBuilding examples (top-{args.top_k} docs per query) …")
    examples = generate_examples(entries, queries, corpus, args.top_k)
    train_ex, val_ex = train_val_split(examples, args.val_split, args.seed)
    print(f"  Train: {len(train_ex):,} examples ({len({e['qid'] for e in train_ex})} queries)")
    print(f"  Val  : {len(val_ex):,}  examples ({len({e['qid'] for e in val_ex})} queries)")

    # ── Model ──────────────────────────────────────────────────────────────────
    print(f"\nLoading model from {args.checkpoint} …")
    tokenizer = T5Tokenizer.from_pretrained(args.checkpoint, legacy=False, use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint).to(device)
    model.gradient_checkpointing_enable()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {param_count / 1e6:.1f}M")

    # ── DataLoader ─────────────────────────────────────────────────────────────
    train_ds = LiT5Top20Dataset(train_ex, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        sampler=SortedSampler(train_ex),
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ── Optimiser ──────────────────────────────────────────────────────────────
    encoder_params = list(model.encoder.parameters())
    encoder_ids    = {id(p) for p in encoder_params}
    decoder_params = [p for p in model.parameters() if id(p) not in encoder_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.lr * 0.5},
            {"params": decoder_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    total_steps  = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    print(f"\nOptimiser steps: {total_steps:,}  |  warmup: {warmup_steps}  |  LR={args.lr}")

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_scaler = (device.type == "cuda") and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # ── State ──────────────────────────────────────────────────────────────────
    best_val     = -1.0
    best_epoch   = 0
    patience_cnt = 0
    global_step  = 0
    train_losses = []
    val_scores   = []
    saved_ckpts  = []
    metric_name  = f"nDCG@{args.val_eval_k}" if HAS_IR_MEASURES else "Top-1 Acc"

    print("\n" + "═" * 65)
    print(f"  LiT5 TOP-20 fine-tune | {args.epochs} epochs | top_k={args.top_k}")
    print(f"  text_maxlength={args.max_length} | answer_maxlength={MAX_NEW_TOKENS}")
    print(f"  LR={args.lr} (enc×0.5) | WD={args.weight_decay} | GradAccum={args.grad_accum}")
    print(f"  Label-smooth={args.label_smoothing} | Weight decay λ={args.decay_factor} | Val-metric={metric_name}")
    print("═" * 65)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch}")
        for step, batch in pbar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                B, n, L = input_ids.shape
                flat_ids = input_ids.view(B * n, L)
                flat_mask = attention_mask.view(B * n, L)
                enc_out = model.encoder(input_ids=flat_ids, attention_mask=flat_mask)
                hidden = enc_out.last_hidden_state.view(B, n * L, -1)
                attn = flat_mask.view(B, n * L)

                out = model(
                    encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
                    attention_mask=attn,
                    labels=labels,
                )
                # ── Weighted cross-entropy with exponential decay ──
                loss = weighted_smooth_ce(
                    out.logits, labels, args.label_smoothing, args.decay_factor
                )
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                pbar.set_postfix(
                    loss=f"{loss.item() * args.grad_accum:.4f}",
                    step=global_step,
                )

        if len(train_loader) % args.grad_accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        avg_loss = epoch_loss / max(1, len(train_loader))
        train_losses.append(avg_loss)

        # ── Validation ─────────────────────────────────────────────────────────
        print(f"\n  Epoch {epoch} — avg train loss: {avg_loss:.4f}")
        print(f"  Running validation ({metric_name}) …")
        val_score = evaluate_ndcg(
            model, tokenizer, val_ex, entry_map, qrels,
            device, args.max_length, args.top_k, args.val_eval_k,
        )
        val_scores.append(val_score)
        model.train()

        lr_now = optimizer.param_groups[-1]["lr"]
        print(
            f"  Epoch {epoch:3d} | loss={avg_loss:.4f} "
            f"| {metric_name}={val_score:.4f} | lr={lr_now:.2e}"
        )

        # ── Overfit warning ────────────────────────────────────────────────────
        if len(train_losses) >= 3:
            loss_drop  = train_losses[-3] - train_losses[-1]
            score_gain = (val_scores[-1] - val_scores[-3]) if len(val_scores) >= 3 else 0.0
            if loss_drop > 0.05 and score_gain < 0.002:
                print("  ⚠  Train loss dropping but val score stagnant — watch for overfit")

        # ── Checkpoint rotation ────────────────────────────────────────────────
        if val_score > best_val + 1e-5:
            best_val     = val_score
            best_epoch   = epoch
            patience_cnt = 0
            tag = f"ep{epoch:02d}_{metric_name.replace('@', 'at')}_{val_score:.4f}"
            ckpt_path = OUT_DIR / tag
            ckpt_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            saved_ckpts.append((val_score, ckpt_path))
            saved_ckpts.sort(key=lambda x: x[0], reverse=True)
            while len(saved_ckpts) > MAX_SAVED_CKPTS:
                _, old = saved_ckpts.pop()
                shutil.rmtree(old, ignore_errors=True)
                print(f"  🗑  Removed old checkpoint: {old.name}")
            print(f"  ✓ Best {metric_name}={best_val:.4f} — saved {ckpt_path.name}")
            print(f"     Kept: {[p.name for _, p in saved_ckpts]}")
        else:
            patience_cnt += 1
            print(f"  No improvement ({patience_cnt}/{args.patience})")
            if patience_cnt >= args.patience:
                print(f"\n  Early stop at epoch {epoch} (best epoch={best_epoch})")
                break

    # ── Final report ───────────────────────────────────────────────────────────
    best_path = saved_ckpts[0][1] if saved_ckpts else OUT_DIR
    print("\n" + "═" * 65)
    print(f"  Best {metric_name} : {best_val:.4f}  (epoch {best_epoch})")
    print(f"  Best checkpoint   : {best_path}")
    print("═" * 65)

    # ── Training curves ────────────────────────────────────────────────────────
    epochs_done = list(range(1, len(train_losses) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(epochs_done, train_losses, "o-", label="Train Loss")
    ax1.set(xlabel="Epoch", ylabel="Loss", title="Training Loss")
    ax1.legend()

    ax2.plot(epochs_done[: len(val_scores)], val_scores, "s-",
             color="darkorange", label=metric_name)
    if best_epoch:
        ax2.axvline(best_epoch, ls="--", color="grey", alpha=0.6,
                    label=f"Best (ep {best_epoch})")
    ax2.set(xlabel="Epoch", ylabel=metric_name, title=f"Validation {metric_name}")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(CURVE_IMG, dpi=150)
    print(f"  Curve saved: {CURVE_IMG}")
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune LiT5-Distill on BioASQ (top-20)")
    p.add_argument("--checkpoint",      default=str(BASE_CKPT))
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
    p.add_argument("--val_eval_k",      type=int,   default=VAL_EVAL_K)
    p.add_argument("--seed",            type=int,   default=SEED)
    p.add_argument("--decay_factor",    type=float, default=DECAY_FACTOR,
                   help="Exponential decay factor λ for weighted CE (LiT5 default 0.95)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)