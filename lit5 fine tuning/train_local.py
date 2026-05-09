"""LoRA fine-tuning of LiT5-Distill-base on BioASQ — LOCAL.

Flow per run:
  1. For each epoch: train → eval on 10 fixed test queries → save adapter.
     Zero-shot baseline eval is skipped (saves ~2 min); compare against the
     unfine-tuned LiT5 numbers in your existing zero-shot results file.
  2. Loss + nDCG@1 curve plot at the end.

Train data: windowed_train_prompt2_2400.jsonl (600 newest queries × 4 windows,
            built from the prompt-2 DeepSeek teacher — no gold-pushing).
Eval data: bm25_top100_test.jsonl (deterministic first 10 by qid).

Run from project root:
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python "lit5 fine tuning/train_local.py"

Outputs (checkpoints/lit5_bioasq_lora_local/):
  - epoch_{1..N}/   LoRA adapter + train_state.pt per epoch
  - results.tsv     epoch \\t train_loss
  - training_curve.png

Resume: rerun — picks up from the latest epoch_N/ in OUT.
"""

import json
import math
import os
import random
import traceback
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
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from tqdm import tqdm
import ir_measures
from ir_measures import nDCG, RR, Recall, P, ScoredDoc, Qrel

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")


# ── Paths (local) ───────────────────────────────────────────────────────────

PROJECT_ROOT = Path("/home/oussama/Desktop/reranking_project")
FT_DIR = PROJECT_ROOT / "lit5 fine tuning"

BASE_CKPT  = PROJECT_ROOT / "checkpoints" / "LiT5-Distill-base"
TRAIN_FILE = FT_DIR / "windowed_train.jsonl"
TEST_CACHE = FT_DIR / "bm25_top100_test.jsonl"
QRELS_FILE = PROJECT_ROOT / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"

OUT = PROJECT_ROOT / "checkpoints" / "lit5_bioasq_lora_local"
OUT.mkdir(parents=True, exist_ok=True)

for p in (BASE_CKPT, TRAIN_FILE, TEST_CACHE, QRELS_FILE):
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")

# ── Hyperparameters (frozen) ─────────────────────────────────────────────────

# Zero-shot is at nDCG@1=0.9375 — already near-saturated. Earlier r=16,q,k,v,o
# at LR=1e-5 *degraded* the model (0.9375 → 0.84-0.87). New strategy: shrink
# the adapter's capacity instead of just shrinking the LR.
#   - r=8, targets=[q,v] → ~25× fewer trainable params than r=16+FFN megalist.
#     Less capacity ⇒ less ability to drift, even at a moderate LR.
#   - α/r = 2 (standard) — q,v with rank 8 needs the usual scaling.
#   - LR=5e-6 — half the kaggle 1e-5 that degraded; higher than the over-cautious
#     1e-6 because the adapter is now small enough that gradients matter again.
#   - 2 epochs, dropout 0.05 — fewer drift opportunities, lighter regularisation.
LR = 5e-6
EPOCHS = 6
BATCH_SIZE = 4
GRAD_ACCUM = 2 
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
LABEL_SMOOTHING = 0.1

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q", "v"]

WINDOW_SIZE = 20
STRIDE = 10
MAX_LENGTH = 350        # full passage context (per request)
MAX_TARGET = 100        # 20-item perm is ~80 tokens; 140 was wasted decoding
SEED = 42

# 12 GB local GPU + FFN-LoRA + MAX_LENGTH=350 doesn't fit without checkpointing
# (OOMs in cross-attention softmax). Keep ON locally; flip to False only if
# you move to a 16 GB+ card.
USE_GRAD_CKPT = True

QUERY_PREFIX = "Search Query:"
PASS_PREFIX = "Passage:"
SUFFIX = " Relevance Ranking:"

# Same fixed eval subset as the Kaggle script — sorted-by-qid, first N.
EVAL_FIXED_N = 10
# During training-time eval, only rerank the top-K BM25 candidates per query
# instead of all 100. Cuts sliding-window passes from 9/query to 4/query.
EVAL_TOP_K = 50

METRICS = [
    nDCG @ 1, nDCG @ 5, nDCG @ 10, nDCG @ 20,
    RR,
    Recall @ 1, Recall @ 5, Recall @ 10, Recall @ 20,
    P @ 1, P @ 5, P @ 10,
]
SELECTION_METRIC = nDCG @ 1

# ── HuggingFace Hub push ────────────────────────────────────────────────────
# Each epoch's adapter (and the rolling best_adapter) is uploaded to a single
# repo as a subfolder: epoch_1/, epoch_2/, ..., best_adapter/.
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "ouarezkioussama/lit5-bioasq-lora-local"
HF_PRIVATE = True
PUSH_TO_HUB = False     # local run keeps adapters on disk; flip to True to push

# ── HuggingFace Hub helpers ─────────────────────────────────────────────────

_hf_api_obj = None
_hf_repo_ready = False

def _hf_api():
    global _hf_api_obj, _hf_repo_ready
    if _hf_api_obj is None:
        from huggingface_hub import HfApi
        _hf_api_obj = HfApi(token=HF_TOKEN)
    if not _hf_repo_ready:
        _hf_api_obj.create_repo(repo_id=HF_REPO, private=HF_PRIVATE,
                            exist_ok=True, repo_type="model")
        _hf_repo_ready = True
    return _hf_api_obj


def push_dir(local_dir, subfolder, commit_msg):
    """Upload `local_dir` to HF as `<HF_REPO>/<subfolder>/`. Best-effort: log
    on failure rather than killing training."""
    if not PUSH_TO_HUB:
        return
    try:
        api = _hf_api()
        api.upload_folder(
            folder_path=str(local_dir),
            repo_id=HF_REPO,
            path_in_repo=subfolder,
            commit_message=commit_msg,
            ignore_patterns=["*.pt"],   # skip optimizer state — large, training-only
        )
        print(f"  ↑ pushed → https://huggingface.co/{HF_REPO}/tree/main/{subfolder}")
    except Exception as e:
        print(f"  [HF push failed for {subfolder}: {e}]")
        traceback.print_exc()


# ── Data ─────────────────────────────────────────────────────────────────────

def load_train():
    return [json.loads(l) for l in TRAIN_FILE.open()]


def load_test_cache():
    return [json.loads(l) for l in TEST_CACHE.open()]


def load_qrels():
    qrels = []
    with QRELS_FILE.open() as f:
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


def build_target(input_docids, target_docids):
    pos = {d: i for i, d in enumerate(input_docids)}
    return " > ".join(f"[{pos[d] + 1}]" for d in target_docids if d in pos)


class WindowDataset(Dataset):
    def __init__(self, rows, tokenizer):
        super().__init__()
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


# ── FiD encode + smoothed CE ────────────────────────────────────────────────

def fid_encode(model, input_ids, attention_mask):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    B, n, L = input_ids.shape
    flat_ids = input_ids.view(B * n, L)
    flat_mask = attention_mask.view(B * n, L)
    enc = base.encoder(input_ids=flat_ids, attention_mask=flat_mask)
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


# ── Sliding-window inference ────────────────────────────────────────────────

def parse_perm(text, n):
    out, seen = [], set()
    for tok in text.replace(",", " ").replace(">", " ").split():
        try:
            v = int(tok.strip("[]()."))
            if 1 <= v <= n and (v - 1) not in seen:
                out.append(v - 1); seen.add(v - 1)
        except ValueError:
            continue
    for i in range(n):
        if i not in seen:
            out.append(i)
    return out


@torch.no_grad()
def rerank_window(model, tokenizer, query, window, device, amp_dtype):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    strings = [
        f"{QUERY_PREFIX} {query} {PASS_PREFIX} [{i + 1}] {t}{SUFFIX}"
        for i, (_, t) in enumerate(window)
    ]
    enc = tokenizer(strings, max_length=MAX_LENGTH, padding="max_length",
                    truncation=True, return_tensors="pt").to(device)
    use_amp = amp_dtype != torch.float32 and device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        hidden, attn = fid_encode(model, enc["input_ids"].unsqueeze(0),
                                  enc["attention_mask"].unsqueeze(0))
        out_ids = base.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
            attention_mask=attn,
            max_new_tokens=MAX_TARGET,
            do_sample=False,
        )
    perm_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    order = parse_perm(perm_text, len(window))
    return [window[i] for i in order]


def sliding_window_rerank(model, tokenizer, query, candidates, device, amp_dtype):
    ranked = list(candidates)
    if len(ranked) <= WINDOW_SIZE:
        return rerank_window(model, tokenizer, query, ranked, device, amp_dtype)
    start = len(ranked) - WINDOW_SIZE
    while True:
        end = min(start + WINDOW_SIZE, len(ranked))
        ranked[start:end] = rerank_window(model, tokenizer, query, ranked[start:end],
                                          device, amp_dtype)
        if start == 0:
            break
        start = max(0, start - STRIDE)
    return ranked


def evaluate(model, tokenizer, test_rows, qrels, device, amp_dtype):
    model.eval()
    # Re-enable decoder KV cache — gradient_checkpointing_enable() turns it off
    # model-wide; without it, every generated token re-attends to the full
    # ~7000-token encoder context (3-5× slower).
    base_for_cfg = model.get_base_model() if hasattr(model, "get_base_model") else model
    prev_use_cache = getattr(base_for_cfg.config, "use_cache", True)
    base_for_cfg.config.use_cache = True
    test_qids = {r["qid"] for r in test_rows}
    run = []
    for r in tqdm(test_rows, desc="  eval", leave=False):
        cands = [(c["docid"], c["text"]) for c in r["candidates"][:EVAL_TOP_K]]
        reranked = sliding_window_rerank(model, tokenizer, r["query"], cands, device, amp_dtype)
        for rank, (did, _) in enumerate(reranked, start=1):
            run.append(ScoredDoc(r["qid"], did, score=len(cands) - rank + 1))
    qrel_subset = [q for q in qrels if q.query_id in test_qids]
    res = ir_measures.calc_aggregate(METRICS, qrel_subset, run)
    base_for_cfg.config.use_cache = prev_use_cache
    return {str(m): float(v) for m, v in res.items()}


# ── Train ────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if not use_bf16 and device.type == "cuda":
        print("WARNING: bf16 not supported on this GPU (likely T4). "
              "Falling back to fp32 — slower & higher memory. "
              "Use a P100/A100/L4 GPU for bf16.")
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"AMP dtype: {amp_dtype}")

    print("\nLoading data ...")
    train_rows = load_train()
    test_rows_full = load_test_cache()
    qrels = load_qrels()
    eval_rows = sorted(test_rows_full, key=lambda r: r["qid"])[:EVAL_FIXED_N]
    print(f"  Train examples (windows): {len(train_rows)}")
    print(f"  Steps/epoch (batches):    {len(train_rows) // BATCH_SIZE}")
    print(f"  Optimizer updates/epoch:  {len(train_rows) // (BATCH_SIZE * GRAD_ACCUM)}")
    print(f"  Eval queries (fixed):     {len(eval_rows)} of {len(test_rows_full)}")
    print(f"  Eval qids: {[r['qid'] for r in eval_rows]}")

    # ── Resume detection ─────────────────────────────────────────────────────
    # If OUT already contains epoch_{N} adapters, pick up from the latest one.
    existing = sorted(
        (int(p.name.split("_")[1]), p) for p in OUT.glob("epoch_*") if p.is_dir()
    )
    resume_dir, start_epoch = None, 1
    if existing:
        last_epoch, resume_dir = existing[-1]
        start_epoch = last_epoch + 1
        print(f"\n[resume] Found checkpoint epoch_{last_epoch} at {resume_dir}")
        if start_epoch > EPOCHS:
            print(f"[resume] All {EPOCHS} epochs already finished. Nothing to do.")
            return

    print(f"\nLoading base checkpoint: {BASE_CKPT}")
    tokenizer = T5Tokenizer.from_pretrained(str(BASE_CKPT), legacy=False, use_fast=True)
    base = T5ForConditionalGeneration.from_pretrained(str(BASE_CKPT))
    if resume_dir is not None:
        print(f"[resume] Loading LoRA adapter from {resume_dir}")
        model = PeftModel.from_pretrained(base, str(resume_dir), is_trainable=True).to(device)
    else:
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGETS, bias="none",
        )
        model = get_peft_model(base, lora_cfg).to(device)
    model.print_trainable_parameters()
    if USE_GRAD_CKPT:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        print("  gradient_checkpointing: ON")
    else:
        print("  gradient_checkpointing: OFF (faster, more VRAM)")

    loader = DataLoader(WindowDataset(train_rows, tokenizer),
                        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate,
                        num_workers=2)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = math.ceil(len(loader) / GRAD_ACCUM) * EPOCHS
    warmup = max(1, int(total_steps * WARMUP_RATIO))

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print("\n" + "=" * 70)
    print(f"  LoRA  LR={LR}  epochs={EPOCHS}  windows={len(train_rows)}  "
          f"start_epoch={start_epoch}")
    print(f"  r={LORA_R} alpha={LORA_ALPHA} dropout={LORA_DROPOUT} targets={LORA_TARGETS}")
    print("=" * 70)

    results_tsv = OUT / "results.tsv"
    losses = []
    sel0 = 0.0

    if resume_dir is not None:
        state_path = resume_dir / "train_state.pt"
        if state_path.exists():
            print(f"[resume] Loading optimizer/scheduler state from {state_path}")
            # weights_only=False — PyTorch 2.6 flipped the default to True,
            # which rejects numpy RNG state in our checkpoint. Safe here
            # because we wrote the file ourselves.
            ckpt = torch.load(state_path, map_location=device, weights_only=False)
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
        else:
            print("[resume] No train_state.pt found — optimizer state will be fresh.")
        if results_tsv.exists():
            for line in results_tsv.read_text().splitlines()[1:]:
                parts = line.split("\t")
                try:
                    losses.append(float(parts[1]))
                except (ValueError, IndexError):
                    pass
        baseline_meta = OUT / "baseline_metrics.json"
        if baseline_meta.exists():
            with baseline_meta.open() as f:
                sel0 = json.load(f).get("nDCG@1", 0.0)
    else:
        with results_tsv.open("w") as f:
            cols = "\t".join(str(m) for m in METRICS)
            f.write(f"epoch\ttrain_loss\t{cols}\n")
        # Zero-shot baseline eval skipped on purpose — go straight into
        # epoch 1 to save ~2 min. Compare epoch metrics against the project's
        # standalone zero-shot LiT5 numbers if you need a delta.

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        running = 0.0
        optimizer.zero_grad()
        pbar = tqdm(loader, desc=f"epoch {epoch}/{EPOCHS}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                hidden, attn = fid_encode(model, input_ids, attn_mask)
                base_m = model.get_base_model()
                out = base_m(encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
                             attention_mask=attn, labels=labels)
                loss = smooth_ce(out.logits, labels, LABEL_SMOOTHING)
                loss = loss / GRAD_ACCUM
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue
            loss.backward()
            running += loss.item() * GRAD_ACCUM
            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
            pbar.set_postfix(loss=f"{running / (step + 1):.4f}")
        if len(loader) % GRAD_ACCUM != 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        avg = running / max(1, len(loader))
        losses.append(avg)
        print(f"\n  Epoch {epoch}  loss={avg:.4f}")

        # Eval on the fixed N queries (no baseline → no zero-shot delta).
        m = evaluate(model, tokenizer, eval_rows, qrels, device, amp_dtype)
        print(f"    nDCG  @1={m.get('nDCG@1', 0):.4f}  "
              f"@5={m.get('nDCG@5', 0):.4f}  "
              f"@10={m.get('nDCG@10', 0):.4f}  "
              f"@20={m.get('nDCG@20', 0):.4f}")
        print(f"    MRR={m.get('RR', 0):.4f}  "
              f"Recall@10={m.get('R@10', 0):.4f}")

        with results_tsv.open("a") as f:
            cols = "\t".join(f"{m.get(str(k), 0.0):.6f}" for k in METRICS)
            f.write(f"{epoch}\t{avg:.6f}\t{cols}\n")

        # Save adapter for this epoch.
        ep_dir = OUT / f"epoch_{epoch}"
        if ep_dir.exists():
            shutil.rmtree(ep_dir)
        ep_dir.mkdir(parents=True)
        model.save_pretrained(str(ep_dir))
        tokenizer.save_pretrained(str(ep_dir))
        torch.save({
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (torch.cuda.get_rng_state_all()
                         if device.type == "cuda" else None),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }, ep_dir / "train_state.pt")
        with (ep_dir / "metrics.json").open("w") as f:
            json.dump({"epoch": epoch, "train_loss": avg, **m}, f, indent=2)

        push_dir(ep_dir, f"epoch_{epoch}", f"epoch {epoch} | loss={avg:.4f}")

    print("\n" + "=" * 70)
    print(f"  Done. {EPOCHS} epochs trained. Adapters in {OUT}/epoch_*/")
    print(f"  Eval was on a fixed {EVAL_FIXED_N} queries (zero-shot Δ in TSV).")
    print("=" * 70)

    # Loss + nDCG@1 curves (rebuild from results.tsv for resume-safety).
    plot_epochs, plot_loss, plot_sel = [], [], []
    metric_idx = 2 + METRICS.index(SELECTION_METRIC)
    for line in results_tsv.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < metric_idx + 1:
            continue
        plot_epochs.append(int(parts[0]))
        plot_loss.append(float(parts[1]))
        plot_sel.append(float(parts[metric_idx]))
    if plot_epochs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        train_pts = [(e, l) for e, l in zip(plot_epochs, plot_loss) if e >= 1]
        if train_pts:
            ax1.plot([e for e, _ in train_pts], [l for _, l in train_pts],
                     marker="o", color="steelblue")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss"); ax1.set_title("Training loss")
        ax2.plot(plot_epochs, plot_sel, marker="s", color="darkorange")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel(f"nDCG@1 (fixed {EVAL_FIXED_N} queries)")
        ax2.set_title("Validation nDCG@1")
        plt.tight_layout()
        plt.savefig(OUT / "training_curve.png", dpi=150)
        plt.close()
        print(f"  Curve     : {OUT / 'training_curve.png'}")
    print(f"  Results   : {results_tsv}")


if __name__ == "__main__":
    main()
