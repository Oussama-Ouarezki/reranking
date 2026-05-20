"""
Fine-tune LiT5-Distill on BioASQ — TOP-20 ONLY, high-accuracy configuration.

Dataset  : data/bioasq/reranked/deepseek_oracle_hybrid.jsonl
           Each entry has bm25_order + permutation (DeepSeek oracle-hybrid ranking).
           Only the TOP-20 BM25 docs are used per query — no sliding window needed
           since window_size == top_k == 20 → exactly ONE example per query.

Method   : Supervised listwise distillation (FiD encoder, T5 decoder).
           The model sees all 20 (query, passage) pairs fused in the encoder,
           and must output "[j] > [k] > ..." sorted by oracle relevance.

Input format follows the LiT5-Distill paper exactly (Tamber et al., 2023):
  "Search Query: <query> Passage: [i] <text> Relevance Ranking:"
  Output: "[2] > [1] > ... > [20]"
  text_maxlength=150 (repo default); answer_maxlength=140 (repo default).

High-accuracy settings:
  - lr=5e-5  (lower, more careful fine-tuning)
  - top_k=20, window=20 → single window per query (no positional inconsistency)
  - Larger grad_accum=16 → effective batch of 16 queries (smoother gradient)
  - label_smoothing=0.05 (light smoothing, preserves ranking signal)
  - val_split=0.15  (more training data, ~85% train / ~15% val queries)
  - epochs=20, patience=5  (more time to converge)
  - Cosine LR with warmup_ratio=0.10; encoder uses 0.5× LR
  - SortedSampler: longest examples first to minimise padding waste
  - nDCG@5 early stopping (changed from @10 per request)
  - bfloat16 preferred (matches repo); fp16 fallback on older GPUs

Usage:
    python finetune_lit5_bioasq_top20.py
    python finetune_lit5_bioasq_top20.py --lr 5e-5 --epochs 20 --checkpoint /path/to/LiT5-Distill-base
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
RERANKED_FILE  = ROOT / "data/bioasq/reranked/deepseek_oracle_hybrid.jsonl"
FULL_CORPUS    =  "data/bioasq/pubmed_full/full/corpus_full.jsonl"
QUERIES_FILE   = ROOT / "data/bioasq/processed/queries.jsonl"
QRELS_FILE     = ROOT / "data/bioasq/processed/qrels.tsv"
BASE_CKPT      = ROOT / "checkpoints/LiT5-Distill-base"
OUT_DIR        = ROOT / "checkpoints/lit5_top20_oracle"
CURVE_IMG      = OUT_DIR / "training_curve.png"

MAX_SAVED_CKPTS = 3

# ── Hyperparameters ────────────────────────────────────────────────────────────
TOP_K = 20     # only top-20 BM25 docs
WINDOW_SIZE = 20  # == TOP_K → one window per query, no sliding

# Repo-accurate defaults (from LiT5-Distill.sh / FiD/LiT5-Distill.py)
TEXT_MAXLENGTH = 150   # --text_maxlength 150 in the official shell script
MAX_NEW_TOKENS = 140   # --answer_maxlength 140 in the official shell script
#  "[1] > [2] > ... > [20]" is ~79 tokens; 140 gives generous headroom.

LR             = 5e-5
WEIGHT_DECAY   = 0.01
WARMUP_RATIO   = 0.10
EPOCHS         = 20
GRAD_ACCUM     = 16
PATIENCE       = 5
VAL_SPLIT      = 0.15
LABEL_SMOOTH   = 0.05
SEED           = 42
VAL_EVAL_K     = 5    # ← changed to nDCG@5 as requested

# Input format strings — must match the LiT5-Distill paper exactly
# "Search Query: {q} Passage: [{i}] {text} Relevance Ranking:"
QUERY_PREFIX   = "Search Query:"
PASS_PREFIX    = "Passage:"
SUFFIX         = " Relevance Ranking:"


# ── Data loading ──────────────────────────────────────────────────────────────

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
    """Return list[Qrel] if ir_measures available, else empty list."""
    if not HAS_IR_MEASURES:
        return []
    qrels = []
    with QRELS_FILE.open() as f:
        next(f)   # skip header
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
    """
    Map DeepSeek permutation to a 1-based position string over window_docids.

    permutation is the oracle ranking (best-first) of doc IDs.
    We want to output: the positions in window_docids ordered by permutation rank,
    e.g. "[3] > [1] > [2] > ..."
    """
    # perm_rank[docid] = rank in permutation (0 = most relevant)
    perm_rank = {docid: rank for rank, docid in enumerate(permutation)}
    # Sort positions in window_docids by their perm_rank (missing → last)
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
    """One example per query using exactly the top-k BM25 docs."""
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
            # Pad with sentinel if fewer than top_k docs retrieved
            bm25_order = bm25_order + ["__PAD__"] * (top_k - len(bm25_order))

        texts = []
        n_missing = 0
        for did in bm25_order:
            t = corpus.get(did, "")
            if not t:
                n_missing += 1
            texts.append(t)

        # Skip if more than 30% of passages are missing
        if n_missing > 0.30 * top_k:
            skipped_missing_docs += 1
            continue

        target = build_target(bm25_order, permutation)
        examples.append({
            "qid":           qid,
            "query":         query,
            "passage_texts": texts,   # list of top_k strings
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
    """Split by query ID so all windows per query stay in one partition."""
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
    """
    Each item is one query + top-k passages.

    The input format follows the LiT5-Distill paper (Tamber et al., 2023):
        "Search Query: <query> Passage: [i] <text> Relevance Ranking:"

    FiD encoding: each of the n passage strings is tokenized independently
    so the encoder can process them in parallel.  The decoder then attends
    over the concatenated representations and outputs the permutation string.
    """

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

        # Build one string per passage (FiD style)
        # Format: "Search Query: {q} Passage: [{i}] {text} Relevance Ranking:"
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
        # Replace padding token id with -100 so it's ignored in the loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      enc["input_ids"],       # [n_passages, max_length]
            "attention_mask": enc["attention_mask"],   # [n_passages, max_length]
            "labels":         labels,                  # [target_len]
        }


def collate_fn(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
    }


# ── Sorted sampler ────────────────────────────────────────────────────────────

class SortedSampler(Sampler):
    """
    Yield indices sorted by total passage text length (descending).
    Putting the longest sequences first packs GPU memory efficiently
    and avoids OOM surprises mid-epoch.
    """

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
    """
    Perform a FiD forward (train) or generate (inference) pass.

    input_ids      : [B, n_passages, seq_len]
    attention_mask : [B, n_passages, seq_len]
    labels         : [B, target_len]  — provided during training, None at inference

    The key FiD trick: encode every passage independently, then concatenate
    the hidden states so the decoder attends over all of them jointly.
    """
    B, n, L = input_ids.shape
    flat_ids  = input_ids.view(B * n, L)
    flat_mask = attention_mask.view(B * n, L)

    # Encode each (query, passage) independently
    encoder_out = model.encoder(input_ids=flat_ids, attention_mask=flat_mask)

    # Concatenate representations: [B, n*L, hidden_dim]
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
            num_beams=1,   # greedy for speed; use num_beams=4 for best accuracy
        )


# ── Label-smoothed cross-entropy ──────────────────────────────────────────────

def smooth_ce(logits: torch.Tensor, labels: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Label-smoothed CE.  Labels with value -100 are masked out (padding).
    eps=0 reduces exactly to standard CE.
    """
    vocab       = logits.size(-1)
    flat_logits = logits.view(-1, vocab)
    flat_labels = labels.view(-1)
    mask        = flat_labels != -100
    flat_logits = flat_logits[mask]
    flat_labels = flat_labels[mask]
    log_probs   = F.log_softmax(flat_logits, dim=-1)
    nll         = -log_probs.gather(1, flat_labels.unsqueeze(1)).squeeze(1)
    smooth      = -log_probs.mean(dim=-1)
    return ((1 - eps) * nll + eps * smooth).mean()


# ── Permutation parser ────────────────────────────────────────────────────────

def parse_ranking(perm_text: str, n: int) -> list[int]:
    """
    Parse "[j] > [k] > ..." into a 0-based index list.
    Fills any missing positions with their original order to guarantee length n.
    """
    nums, seen = [], set()
    for tok in perm_text.replace(",", " ").replace(">", " ").split():
        try:
            v = int(tok.strip("[]()."))
            if 1 <= v <= n and (v - 1) not in seen:
                nums.append(v - 1)
                seen.add(v - 1)
        except ValueError:
            continue
    # Append any missing positions in original order
    for i in range(n):
        if i not in seen:
            nums.append(i)
    return nums


# ── Proxy accuracy (fallback when ir_measures is unavailable) ─────────────────

def top1_accuracy(pred_perm: str, gold_perm: str, n: int) -> float:
    """Did the model place the same doc at rank-1 as the oracle?"""
    pred = parse_ranking(pred_perm, n)
    gold = parse_ranking(gold_perm, n)
    return float(pred[0] == gold[0]) if pred and gold else 0.0


# ── nDCG@K evaluation ─────────────────────────────────────────────────────────

def evaluate_ndcg(
    model,
    tokenizer,
    val_examples: list[dict],
    entry_map: dict[str, dict],
    qrels: list,
    device: torch.device,
    max_length: int,
    top_k: int,
    val_eval_k: int,
) -> float:
    """
    Run single-window (no sliding) inference on the val set and return nDCG@val_eval_k.
    Falls back to top-1 accuracy if ir_measures is not installed.
    """
    model.eval()

    if HAS_IR_MEASURES:
        metric = nDCG @ val_eval_k
        scored: list = []
    else:
        accs: list[float] = []

    with torch.no_grad():
        for ex in tqdm(val_examples, desc="  val eval", leave=False):
            qid   = ex["qid"]
            query = ex["query"]
            texts = ex["passage_texts"]

            # Build per-passage strings (same format as training)
            strings = [
                f"{QUERY_PREFIX} {query} {PASS_PREFIX} [{i + 1}] {text}{SUFFIX}"
                for i, text in enumerate(texts)
            ]
            enc = tokenizer(
                strings,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)

            # fid_forward expects [B, n, L]; add batch dim
            out = fid_forward(
                model,
                enc["input_ids"].unsqueeze(0),
                enc["attention_mask"].unsqueeze(0),
            )
            perm_text = tokenizer.decode(out[0], skip_special_tokens=True)
            perm      = parse_ranking(perm_text, len(texts))

            if HAS_IR_MEASURES:
                entry      = entry_map.get(qid, {})
                bm25_order = entry.get("bm25_order", [])[:top_k]
                ranked_ids = [bm25_order[i] for i in perm if i < len(bm25_order)]
                n_docs     = len(bm25_order)
                for rank, did in enumerate(ranked_ids):
                    scored.append(ScoredDoc(qid, did, float(n_docs - rank)))
            else:
                accs.append(top1_accuracy(perm_text, ex["target"], len(texts)))

    if HAS_IR_MEASURES:
        if not scored:
            return 0.0
        val_qids = {ex["qid"] for ex in val_examples}
        qrel_sub = [q for q in qrels if q.query_id in val_qids]
        results  = ir_measures.calc_aggregate([metric], qrel_sub, scored)
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

    # Prefer bfloat16 (repo default); fall back to fp16 on older GPUs
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
    # legacy=False avoids the SentencePiece deprecation warning
    tokenizer = T5Tokenizer.from_pretrained(
        args.checkpoint, legacy=False, use_fast=True
    )
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint).to(device)
    model.gradient_checkpointing_enable()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {param_count / 1e6:.1f}M")

    # ── DataLoader ─────────────────────────────────────────────────────────────
    train_ds = LiT5Top20Dataset(train_ex, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=1,               # one query per step (FiD is already heavy)
        sampler=SortedSampler(train_ex),
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ── Optimiser ──────────────────────────────────────────────────────────────
    # Give the encoder a lower LR to avoid forgetting pretrained representations
    encoder_params = list(model.encoder.parameters())
    encoder_ids    = {id(p) for p in encoder_params}
    decoder_params = [p for p in model.parameters() if id(p) not in encoder_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.lr * 0.5},  # encoder: half LR
            {"params": decoder_params, "lr": args.lr},         # decoder: full LR
        ],
        weight_decay=args.weight_decay,
    )

    total_steps  = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    print(f"\nOptimiser steps: {total_steps:,}  |  warmup: {warmup_steps}  |  LR={args.lr}")

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # GradScaler: enabled only for fp16 (bf16 doesn't need it; cpu always off)
    use_scaler = (device.type == "cuda") and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # ── State ──────────────────────────────────────────────────────────────────
    best_val      = -1.0
    best_epoch    = 0
    patience_cnt  = 0
    global_step   = 0
    train_losses: list[float] = []
    val_scores:   list[float] = []
    saved_ckpts:  list[tuple[float, Path]] = []

    metric_name = f"nDCG@{args.val_eval_k}" if HAS_IR_MEASURES else "Top-1 Acc"

    print("\n" + "═" * 65)
    print(f"  LiT5 TOP-20 fine-tune | {args.epochs} epochs | top_k={args.top_k}")
    print(f"  text_maxlength={args.max_length} | answer_maxlength={MAX_NEW_TOKENS}")
    print(f"  LR={args.lr} (enc×0.5) | WD={args.weight_decay} | GradAccum={args.grad_accum}")
    print(f"  Label-smooth={args.label_smoothing} | Val-metric={metric_name}")
    print("═" * 65)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch}")
        for step, batch in pbar:
            input_ids      = batch["input_ids"].to(device)       # [B, n, L]
            attention_mask = batch["attention_mask"].to(device)   # [B, n, L]
            labels         = batch["labels"].to(device)           # [B, T]

            with torch.amp.autocast("cuda",
                                    dtype=amp_dtype,
                                    enabled=(device.type == "cuda")):
                B, n, L   = input_ids.shape
                flat_ids  = input_ids.view(B * n, L)
                flat_mask = attention_mask.view(B * n, L)
                enc_out   = model.encoder(input_ids=flat_ids, attention_mask=flat_mask)
                hidden    = enc_out.last_hidden_state.view(B, n * L, -1)
                attn      = flat_mask.view(B, n * L)

                out  = model(
                    encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
                    attention_mask=attn,
                    labels=labels,
                )
                loss = (
                    smooth_ce(out.logits, labels, args.label_smoothing)
                    if args.label_smoothing > 0
                    else out.loss
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

        # Flush any remaining gradient accumulation at epoch end
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
            model, tokenizer,
            val_ex, entry_map, qrels,
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

            tag       = f"ep{epoch:02d}_{metric_name.replace('@', 'at')}_{val_score:.4f}"
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
    p = argparse.ArgumentParser(
        description="Fine-tune LiT5-Distill on BioASQ (top-20)"
    )
    p.add_argument("--checkpoint",      default=str(BASE_CKPT))
    p.add_argument("--top_k",           type=int,   default=TOP_K,
                   help="Number of BM25 docs per query (default: 20)")
    p.add_argument("--max_length",      type=int,   default=TEXT_MAXLENGTH,
                   help="Per-passage token budget (repo default: 150)")
    p.add_argument("--lr",              type=float, default=LR)
    p.add_argument("--weight_decay",    type=float, default=WEIGHT_DECAY)
    p.add_argument("--warmup_ratio",    type=float, default=WARMUP_RATIO)
    p.add_argument("--epochs",          type=int,   default=EPOCHS)
    p.add_argument("--grad_accum",      type=int,   default=GRAD_ACCUM)
    p.add_argument("--patience",        type=int,   default=PATIENCE)
    p.add_argument("--val_split",       type=float, default=VAL_SPLIT)
    p.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTH)
    p.add_argument("--val_eval_k",      type=int,   default=VAL_EVAL_K,
                   help="Validation nDCG cut-off (default: 5)")
    p.add_argument("--seed",            type=int,   default=SEED)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)