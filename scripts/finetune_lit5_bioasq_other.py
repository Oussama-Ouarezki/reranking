"""
Fine-tune LiT5-Distill on BioASQ — light adaptation, internal train/val split.

Goal
────
The base LiT5-Distill checkpoint is already strong. This script performs a
*light adaptation* pass to improve permutation generation quality on BioASQ
domain text. It does NOT attempt to teach ranking from scratch.

Key design choices
──────────────────
• Full distillation JSONL split internally into train / val (no test set needed).
• Proper listwise cross-entropy loss over passage-index token positions
  (Tamber et al., 2023) — sharper gradient than token-level CE.
• Conservative hyperparameters: low LR, few epochs, early stopping on val loss.
• Encoder LR is halved relative to decoder (encoder already well-adapted).
• Per-epoch shuffling so the model never memorises example order.
• No label smoothing — not meaningful for the listwise formulation.
• Validation reports both listwise CE loss and nDCG@5 (pseudo-relevance from
  the distillation permutation: rank position used as graded relevance).
  Early stopping is on val loss; nDCG@5 is reported for information only.

Input format (LiT5-Distill paper):
  "Search Query: <query> Passage: [i] <text> Relevance Ranking:"
  Output: "[2] > [1] > ... > [20]"

Usage
─────
    python finetune_lit5_bioasq_top20.py
    python finetune_lit5_bioasq_top20.py --lr 3e-5 --epochs 6
    python finetune_lit5_bioasq_top20.py --val_split 0.1
"""

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

RERANKED_FILE = ROOT / "data/bioasq/reranked/deepseek_sliding_reranked_prompt_2.jsonl"
TRAIN_CORPUS  = ROOT / "data/bioasq/pubmed_full/full/corpus_full_processed.jsonl"
TRAIN_QUERIES = ROOT / "data/bioasq/processed/queries.jsonl"

BASE_CKPT = ROOT / "checkpoints/LiT5-Distill-base"
OUT_DIR   = ROOT / "checkpoints/lit5_top20_adapted"
CURVE_IMG = OUT_DIR / "training_curve.png"

MAX_SAVED_CKPTS = 5

# ── Hyperparameters ────────────────────────────────────────────────────────────
# Conservative — base model is already strong, we want gentle nudging only.
TOP_K          = 20
TEXT_MAXLENGTH = 350
MAX_NEW_TOKENS = 140

LR           = 2e-5    # low — avoid overwriting pretrained knowledge
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
EPOCHS       = 6       # enough for adaptation, not enough to overfit
GRAD_ACCUM   = 8       # effective batch = 8 (lighter than full fine-tuning)
PATIENCE     = 3       # stop early if val loss stops improving
VAL_SPLIT    = 0.15    # 15% of queries held out for validation
SEED         = 42

QUERY_PREFIX = "Search Query:"
PASS_PREFIX  = "Passage:"
SUFFIX       = " Relevance Ranking:"


# ── Passage-index token IDs ───────────────────────────────────────────────────

def build_passage_token_ids(tokenizer, top_k: int) -> list[int]:
    """
    Token ID for "[1]", "[2]", ... "[top_k]" in T5's vocabulary.
    Used by the listwise loss to restrict logits to passage-index positions.
    """
    ids = []
    for i in range(1, top_k + 1):
        toks = tokenizer.encode(f"[{i}]", add_special_tokens=False)
        ids.append(toks[-1])   # take last token (strips leading space token)
    return ids


# ── Listwise cross-entropy loss ───────────────────────────────────────────────

def listwise_ce_loss(
    logits:            torch.Tensor,   # [B, seq_len, vocab]
    labels:            torch.Tensor,   # [B, seq_len]  (-100 = ignore)
    passage_token_ids: list[int],
    device:            torch.device,
) -> torch.Tensor:
    """
    For every position in the label sequence that is a passage-index token
    (one of "[1]"..."[K]"), compute softmax over the K passage-index logits
    and take cross-entropy against the gold passage index.

    Positions that are ">", EOS, or padding are ignored — they carry no
    ranking information and would only dilute the gradient.

    Falls back to standard token-level CE if no passage-index tokens are
    found (should not happen with well-formed data).
    """
    K          = len(passage_token_ids)
    pid_tensor = torch.tensor(passage_token_ids, device=device)   # [K]
    pid_set    = set(passage_token_ids)

    total_loss = torch.tensor(0.0, device=device)
    n_slots    = 0
    B, seq_len, vocab = logits.shape

    for b in range(B):
        for t in range(seq_len):
            target_id = labels[b, t].item()
            if target_id == -100 or target_id not in pid_set:
                continue

            gold_k = (pid_tensor == target_id).nonzero(as_tuple=True)[0]
            if gold_k.numel() == 0:
                continue
            gold_k = gold_k[0]

            slot_logits = logits[b, t, pid_tensor]          # [K]
            total_loss  = total_loss + F.cross_entropy(
                slot_logits.unsqueeze(0), gold_k.unsqueeze(0)
            )
            n_slots += 1

    if n_slots == 0:
        mask        = labels != -100
        flat_logits = logits.view(-1, vocab)[mask.view(-1)]
        flat_labels = labels.view(-1)[mask.view(-1)]
        return F.cross_entropy(flat_logits, flat_labels)

    return total_loss / n_slots


# ── Permutation parser ───────────────────────────────────────────────────────

def parse_ranking(perm_text: str, n: int) -> list[int]:
    """
    Parse a generated permutation string like "[3] > [1] > [2]" into a list
    of 0-based passage indices.  Missing indices are appended at the end.
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
    for i in range(n):
        if i not in seen:
            nums.append(i)
    return nums


def ndcg_at_k(predicted_ranking: list[int], gold_ranking: list[int], k: int = 5) -> float:
    """
    Compute nDCG@k using pseudo-relevance grades derived from the gold
    distillation permutation.

    Grade assignment: the passage ranked 1st in gold gets grade K, 2nd gets
    K-1, …, last gets 1.  This gives a smooth graded signal rather than
    binary relevance.

    predicted_ranking / gold_ranking : 0-based passage indices in ranked order.
    """
    n     = len(gold_ranking)
    grade = {pid: (n - rank) for rank, pid in enumerate(gold_ranking)}

    def dcg(ranking):
        return sum(
            grade.get(pid, 0) / math.log2(rank + 2)   # rank+2 because rank is 0-based
            for rank, pid in enumerate(ranking[:k])
        )

    ideal = dcg(gold_ranking)
    if ideal == 0.0:
        return 0.0
    return dcg(predicted_ranking) / ideal


# ── FiD encoder pass ──────────────────────────────────────────────────────────

def fid_encode(model, input_ids, attention_mask):
    """
    Fusion-in-Decoder encoder: encode each passage independently, then
    concatenate hidden states so the decoder attends to all passages.

    input_ids      : [B, n_passages, L]
    attention_mask : [B, n_passages, L]
    Returns:
        hidden : [B, n_passages * L, d_model]
        attn   : [B, n_passages * L]
    """
    B, n, L   = input_ids.shape
    flat_ids  = input_ids.view(B * n, L)
    flat_mask = attention_mask.view(B * n, L)
    enc_out   = model.encoder(input_ids=flat_ids, attention_mask=flat_mask)
    hidden    = enc_out.last_hidden_state.view(B, n * L, -1)
    attn      = flat_mask.view(B, n * L)
    return hidden, attn


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_jsonl_corpus(path: Path, desc: str = "corpus") -> dict[str, str]:
    corpus: dict[str, str] = {}
    with path.open() as f:
        for line in tqdm(f, desc=f"Loading {desc}"):
            d   = json.loads(line)
            did = d.get("_id") or d.get("id", "")
            corpus[did] = (d.get("title", "") + " " + d.get("text", "")).strip()
    print(f"  {desc}: {len(corpus):,} docs")
    return corpus


def load_jsonl_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            q   = json.loads(line)
            qid = q.get("_id") or q.get("id", "")
            queries[qid] = q.get("text", "")
    return queries


# ── Example building ──────────────────────────────────────────────────────────

def build_target_from_permutation(window_docids: list[str], permutation: list[str]) -> str:
    perm_rank  = {docid: rank for rank, docid in enumerate(permutation)}
    sorted_pos = sorted(
        range(len(window_docids)),
        key=lambda i: perm_rank.get(window_docids[i], len(permutation)),
    )
    return " > ".join(f"[{p + 1}]" for p in sorted_pos)


def build_examples(
    entries: list[dict],
    queries: dict[str, str],
    corpus:  dict[str, str],
    top_k:   int,
) -> list[dict]:
    examples              = []
    skipped_missing_query = 0
    skipped_missing_docs  = 0

    for entry in entries:
        qid = entry["qid"]
        if qid not in queries:
            skipped_missing_query += 1
            continue

        bm25_order  = entry["bm25_order"][:top_k]
        permutation = entry["permutation"][:top_k]

        # Pad to top_k if necessary
        if len(bm25_order) < top_k:
            bm25_order = bm25_order + ["__PAD__"] * (top_k - len(bm25_order))

        texts     = []
        n_missing = 0
        for did in bm25_order:
            t = corpus.get(did, "")
            if not t:
                n_missing += 1
            texts.append(t)

        if n_missing > 0.30 * top_k:
            skipped_missing_docs += 1
            continue

        # gold_ranking: 0-based passage indices ordered best-first by the
        # distillation permutation.  Used for nDCG@5 during validation.
        perm_rank    = {docid: rank for rank, docid in enumerate(permutation)}
        gold_ranking = sorted(
            range(len(bm25_order)),
            key=lambda i: perm_rank.get(bm25_order[i], len(permutation)),
        )

        examples.append({
            "qid":           qid,
            "query":         queries[qid],
            "passage_texts": texts,
            "target":        build_target_from_permutation(bm25_order, permutation),
            "gold_ranking":  gold_ranking,   # 0-based indices, best-first
        })

    print(
        f"  Examples built: {len(examples):,} | "
        f"skipped (no query)={skipped_missing_query} "
        f"(missing docs)={skipped_missing_docs}"
    )
    return examples


def train_val_split(
    examples: list[dict], val_ratio: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """
    Split by query ID so the same query never appears in both train and val.
    This prevents the model from memorising query-specific patterns.
    """
    rng      = random.Random(seed)
    qids     = list({ex["qid"] for ex in examples})
    rng.shuffle(qids)
    n_val    = max(1, int(len(qids) * val_ratio))
    val_qids = set(qids[:n_val])
    train    = [ex for ex in examples if ex["qid"] not in val_qids]
    val      = [ex for ex in examples if ex["qid"] in     val_qids]
    return train, val


# ── Dataset ───────────────────────────────────────────────────────────────────

class LiT5Dataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer, max_length: int):
        self.examples   = examples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        strings = [
            f"{QUERY_PREFIX} {ex['query']} {PASS_PREFIX} [{i+1}] {t}{SUFFIX}"
            for i, t in enumerate(ex["passage_texts"])
        ]
        enc = self.tokenizer(
            strings,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        lbl    = self.tokenizer(
            ex["target"],
            max_length=MAX_NEW_TOKENS,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = lbl["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      enc["input_ids"],       # [n_passages, L]
            "attention_mask": enc["attention_mask"],   # [n_passages, L]
            "labels":         labels,                  # [seq_len]
        }


def collate_fn(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
    }


# ── Validation ────────────────────────────────────────────────────────────────

def evaluate_val(
    model,
    val_examples:      list[dict],
    tokenizer,
    passage_token_ids: list[int],
    device:            torch.device,
    max_length:        int,
    amp_dtype,
    val_eval_k:        int = 5,
) -> tuple[float, float]:
    """
    Compute listwise CE loss and nDCG@k on the validation split.

    The encoder is run once per example and reused for both the loss forward
    pass and the generate call, so we pay the encoding cost only once.

    nDCG uses pseudo-relevance grades derived from the distillation permutation
    (rank position → grade), since we have no external qrels for the internal
    val split.

    Returns (val_loss, ndcg_at_k).
    """
    model.eval()
    total_loss  = 0.0
    ndcg_scores = []
    n           = 0

    with torch.no_grad():
        for ex in tqdm(val_examples, desc="  val", leave=False):
            n_passages = len(ex["passage_texts"])
            strings    = [
                f"{QUERY_PREFIX} {ex['query']} {PASS_PREFIX} [{i+1}] {t}{SUFFIX}"
                for i, t in enumerate(ex["passage_texts"])
            ]
            enc = tokenizer(
                strings,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)

            lbl    = tokenizer(
                ex["target"],
                max_length=MAX_NEW_TOKENS,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)
            labels = lbl["input_ids"].squeeze(0)
            labels[labels == tokenizer.pad_token_id] = -100

            ids_b  = enc["input_ids"].unsqueeze(0)
            mask_b = enc["attention_mask"].unsqueeze(0)
            lbl_b  = labels.unsqueeze(0)

            with torch.amp.autocast("cuda", dtype=amp_dtype,
                                    enabled=(device.type == "cuda")):
                # Single encoder pass — reused for loss AND generate
                hidden, attn = fid_encode(model, ids_b, mask_b)
                enc_obj      = BaseModelOutput(last_hidden_state=hidden)

                # Loss
                out  = model(
                    encoder_outputs=enc_obj,
                    attention_mask=attn,
                    labels=lbl_b,
                )
                loss = listwise_ce_loss(out.logits, lbl_b, passage_token_ids, device)

                # Generate permutation (encoder already cached)
                gen_ids = model.generate(
                    encoder_outputs=enc_obj,
                    attention_mask=attn,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    num_beams=1,
                )

            total_loss += loss.item()
            n          += 1

            perm_text        = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            pred_ranking     = parse_ranking(perm_text, n_passages)
            gold_ranking     = ex.get("gold_ranking", list(range(n_passages)))
            ndcg_scores.append(ndcg_at_k(pred_ranking, gold_ranking, k=val_eval_k))

    val_loss   = total_loss / max(1, n)
    mean_ndcg  = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
    return val_loss, mean_ndcg


# ── Training curves ───────────────────────────────────────────────────────────

def save_curves(
    train_losses: list[float],
    val_losses:   list[float],
    best_epoch:   int,
    out_path:     Path,
) -> None:
    epochs = list(range(1, len(train_losses) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("LiT5 Light Adaptation — Training Curves", fontsize=13, fontweight="bold")

    vline_kw = dict(ls="--", color="grey", alpha=0.6, label=f"Best ep {best_epoch}")

    def _decorate(ax, ylabel, title):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax = axes[0]
    ax.plot(epochs, train_losses, "o-", color="#2196F3", lw=2, ms=5, label="Train loss")
    if best_epoch:
        ax.axvline(best_epoch, **vline_kw)
    _decorate(ax, "Listwise CE", "Training Loss")

    ax = axes[1]
    if val_losses:
        ax.plot(epochs[:len(val_losses)], val_losses, "s-",
                color="#E91E63", lw=2, ms=5, label="Val loss")
        if best_epoch:
            ax.axvline(best_epoch, **vline_kw)
        best_i = int(np.argmin(val_losses))
        ax.scatter(
            [epochs[best_i]], [val_losses[best_i]],
            color="gold", edgecolors="#E91E63", s=120, zorder=5,
            label=f"Best {val_losses[best_i]:.4f}",
        )
    _decorate(ax, "Listwise CE", "Validation Loss")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"  Curve saved -> {out_path}")
    plt.close()


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
        use_bf16  = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        print(f"AMP    : {'bfloat16' if use_bf16 else 'float16'}")
    else:
        amp_dtype = torch.float32

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nLoading corpus ...")
    corpus = load_jsonl_corpus(TRAIN_CORPUS, "corpus")
    print("Loading queries ...")
    queries = load_jsonl_queries(TRAIN_QUERIES)
    print("Loading distillation JSONL ...")
    with RERANKED_FILE.open() as f:
        entries = [json.loads(line) for line in f]
    print(f"  {len(entries):,} entries")

    print(f"\nBuilding examples (top-{args.top_k}) ...")
    all_examples = build_examples(entries, queries, corpus, args.top_k)

    print(f"\nSplitting: {1 - args.val_split:.0%} train / {args.val_split:.0%} val (by query ID) ...")
    train_ex, val_ex = train_val_split(all_examples, args.val_split, args.seed)
    print(f"  Train : {len(train_ex):,}")
    print(f"  Val   : {len(val_ex):,}")

    # ── Model ──────────────────────────────────────────────────────────────────
    print(f"\nLoading model from {args.checkpoint} ...")
    tokenizer = T5Tokenizer.from_pretrained(args.checkpoint, legacy=False, use_fast=True)
    model     = T5ForConditionalGeneration.from_pretrained(args.checkpoint).to(device)
    model.gradient_checkpointing_enable()
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params / 1e6:.1f}M")

    passage_token_ids = build_passage_token_ids(tokenizer, args.top_k)
    print(f"  Passage token ids [1]...[5]: {passage_token_ids[:5]}")

    # ── Optimiser — separate LRs for encoder vs decoder ───────────────────────
    enc_params = list(model.encoder.parameters())
    enc_ids    = {id(p) for p in enc_params}
    dec_params = [p for p in model.parameters() if id(p) not in enc_ids]

    # Encoder LR halved — already well-adapted; decoder needs more nudging
    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": args.lr * 0.5},
            {"params": dec_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    total_steps  = math.ceil(len(train_ex) / args.grad_accum) * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    print(f"\nTotal steps: {total_steps:,}  warmup: {warmup_steps}  LR={args.lr}")

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler  = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_scaler = (device.type == "cuda") and (amp_dtype == torch.float16)
    scaler     = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # ── State ──────────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_epoch    = 0
    patience_cnt  = 0
    global_step   = 0
    train_losses: list[float] = []
    val_losses:   list[float] = []
    saved_ckpts:  list[tuple[float, Path]] = []

    train_ds = LiT5Dataset(train_ex, tokenizer, args.max_length)

    print("\n" + "=" * 65)
    print(f"  LiT5 light adaptation | top_k={args.top_k} | epochs={args.epochs}")
    print(f"  LR={args.lr} (enc x0.5) | grad_accum={args.grad_accum}")
    print(f"  loss=listwise CE | val=internal {args.val_split:.0%} split")
    print(f"  early stopping patience={args.patience} (on val loss)")
    print("=" * 65)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        # Shuffle training order every epoch with a deterministic seed
        indices = list(range(len(train_ex)))
        random.Random(args.seed + epoch).shuffle(indices)

        pbar = tqdm(indices, desc=f"Epoch {epoch}/{args.epochs}")
        for step_in_epoch, idx in enumerate(pbar):
            batch = collate_fn([train_ds[idx]])
            ids_  = batch["input_ids"].to(device)
            mask_ = batch["attention_mask"].to(device)
            lbl_  = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=amp_dtype,
                                    enabled=(device.type == "cuda")):
                hidden, attn = fid_encode(model, ids_, mask_)
                out  = model(
                    encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
                    attention_mask=attn,
                    labels=lbl_,
                )
                loss = listwise_ce_loss(out.logits, lbl_, passage_token_ids, device)
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.grad_accum

            if (step_in_epoch + 1) % args.grad_accum == 0:
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

        # Flush remaining accumulation at epoch boundary
        if len(indices) % args.grad_accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        avg_train_loss = epoch_loss / max(1, len(indices))
        train_losses.append(avg_train_loss)

        # ── Validation ────────────────────────────────────────────────────────
        v_loss = evaluate_val(
            model, val_ex, tokenizer, passage_token_ids,
            device, args.max_length, amp_dtype,
        )
        val_losses.append(v_loss)
        model.train()

        lr_now = optimizer.param_groups[-1]["lr"]
        print(
            f"\n  Epoch {epoch:3d} | train_loss={avg_train_loss:.4f} "
            f"| val_loss={v_loss:.4f} | lr={lr_now:.2e}"
        )

        # Overfit warning
        if len(train_losses) >= 3 and len(val_losses) >= 3:
            if (train_losses[-3] - train_losses[-1] > 0.05
                    and val_losses[-1] - val_losses[-3] > 0.02):
                print("  warning: train loss falling but val loss rising — possible overfit")

        # ── Checkpoint on best val loss ───────────────────────────────────────
        if v_loss < best_val_loss - 1e-5:
            best_val_loss = v_loss
            best_epoch    = epoch
            patience_cnt  = 0

            tag       = f"ep{epoch:02d}_vloss_{v_loss:.4f}"
            ckpt_path = OUT_DIR / tag
            ckpt_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            saved_ckpts.append((v_loss, ckpt_path))
            # Keep only the best MAX_SAVED_CKPTS (lowest val loss first)
            saved_ckpts.sort(key=lambda x: x[0])
            while len(saved_ckpts) > MAX_SAVED_CKPTS:
                _, old = saved_ckpts.pop()   # remove worst (highest loss)
                shutil.rmtree(old, ignore_errors=True)
                print(f"  Removed checkpoint: {old.name}")

            print(f"  Best val_loss={best_val_loss:.4f} -> saved {ckpt_path.name}")
        else:
            patience_cnt += 1
            print(f"  No improvement ({patience_cnt}/{args.patience})")
            if patience_cnt >= args.patience:
                print(f"\n  Early stop triggered — best epoch={best_epoch}")
                break

        save_curves(train_losses, val_losses, best_epoch, CURVE_IMG)

    # ── Summary ───────────────────────────────────────────────────────────────
    best_path = saved_ckpts[0][1] if saved_ckpts else OUT_DIR
    print("\n" + "=" * 65)
    print(f"  Best val_loss : {best_val_loss:.4f}  (epoch {best_epoch})")
    print(f"  Best ckpt     : {best_path}")
    print("=" * 65)
    save_curves(train_losses, val_losses, best_epoch, CURVE_IMG)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Light adaptation of LiT5-Distill on BioASQ (top-20, internal val split)"
    )
    p.add_argument("--checkpoint",   default=str(BASE_CKPT))
    p.add_argument("--top_k",        type=int,   default=TOP_K)
    p.add_argument("--max_length",   type=int,   default=TEXT_MAXLENGTH)
    p.add_argument("--lr",           type=float, default=LR)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--warmup_ratio", type=float, default=WARMUP_RATIO)
    p.add_argument("--epochs",       type=int,   default=EPOCHS)
    p.add_argument("--grad_accum",   type=int,   default=GRAD_ACCUM)
    p.add_argument("--patience",     type=int,   default=PATIENCE)
    p.add_argument("--val_split",    type=float, default=VAL_SPLIT)
    p.add_argument("--seed",         type=int,   default=SEED)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())