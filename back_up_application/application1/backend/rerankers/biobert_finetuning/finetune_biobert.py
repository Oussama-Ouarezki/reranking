"""Fine-tune BioBERT cross-encoder on DeepSeek listwise distillation data.

Teacher signal
--------------
`data/bioasq/reranked/deepseek_sliding_reranked_prompt_2.jsonl` contains, for
each query, a BM25 top-50 list and a DeepSeek-reranked permutation. We treat
the permutation as the gold ordering and distill it into BioBERT with a
pairwise margin-ranking loss.

For every query we sample N pairs (d_high, d_low) where d_high comes from the
top-T positions of the permutation and d_low from the bottom-B positions. The
model (single-logit cross-encoder) must score d_high > d_low.

Starting checkpoint : NeuML/biomedbert-base-reranker
Output              : checkpoints/biobert-bioasq-deepseek/

Usage
-----
python application/backend/rerankers/biobert_finetuning/finetune_biobert.py \
    --reranked_path data/bioasq/reranked/deepseek_sliding_reranked_prompt_2.jsonl \
    --output_dir   checkpoints/biobert-bioasq-deepseek
"""

import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DEFAULT_CHECKPOINT = "NeuML/biomedbert-base-reranker"
DEFAULT_QUERIES = os.path.join(ROOT, "data/bioasq/processed/queries.jsonl")
DEFAULT_INDEX = os.path.join(ROOT, "data/bm25_indexing_full/corpus_full/lucene_index")


# ── corpus access (Pyserini Lucene) ────────────────────────────────────────────

def make_doc_lookup(index_path: str):
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
    os.environ["PATH"] = "/usr/lib/jvm/java-21-openjdk-amd64/bin:" + os.environ.get("PATH", "")
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(index_path)
    cache: dict[str, str] = {}

    def lookup(docid: str) -> str | None:
        if docid in cache:
            return cache[docid]
        doc = searcher.doc(docid)
        if doc is None:
            cache[docid] = ""
            return ""
        try:
            text = json.loads(doc.raw())["contents"]
        except Exception:
            text = doc.raw() or ""
        cache[docid] = text
        return text

    return lookup


# ── pair generation ────────────────────────────────────────────────────────────

def build_pairs(
    reranked_path: str,
    queries: dict[str, str],
    top_k: int,
    bottom_k: int,
    pairs_per_query: int,
    seed: int = 42,
) -> list[tuple[str, str, str]]:
    """Return [(qid, doc_high_id, doc_low_id), ...]."""
    rng = random.Random(seed)
    out: list[tuple[str, str, str]] = []
    with open(reranked_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = row["qid"]
            if qid not in queries:
                continue
            perm = row["permutation"]
            if len(perm) < top_k + bottom_k:
                continue
            highs = perm[:top_k]
            lows = perm[-bottom_k:]
            for _ in range(pairs_per_query):
                h = rng.choice(highs)
                l = rng.choice(lows)
                out.append((qid, h, l))
    rng.shuffle(out)
    return out


# ── dataset ────────────────────────────────────────────────────────────────────

class PairwiseDataset(Dataset):
    def __init__(self, pairs, queries, doc_lookup):
        self.pairs = pairs
        self.queries = queries
        self.lookup = doc_lookup

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        qid, hi, lo = self.pairs[idx]
        return (
            self.queries[qid],
            self.lookup(hi) or "",
            self.lookup(lo) or "",
        )


def build_collator(tokenizer, max_length: int):
    def collate(batch):
        queries = [b[0] for b in batch]
        highs = [b[1] for b in batch]
        lows = [b[2] for b in batch]

        enc_hi = tokenizer(
            list(zip(queries, highs)),
            padding=True, truncation=True, max_length=max_length, return_tensors="pt",
        )
        enc_lo = tokenizer(
            list(zip(queries, lows)),
            padding=True, truncation=True, max_length=max_length, return_tensors="pt",
        )
        return enc_hi, enc_lo
    return collate


# ── plotting ───────────────────────────────────────────────────────────────────

def plot_curves(history, out_path):
    sns.set_theme(style="darkgrid")
    plt.style.use("ggplot")

    steps = [h["step"] for h in history if "train_loss" in h]
    losses = [h["train_loss"] for h in history if "train_loss" in h]
    val_steps = [h["step"] for h in history if "val_loss" in h]
    val_losses = [h["val_loss"] for h in history if "val_loss" in h]
    val_acc = [h["val_acc"] for h in history if "val_acc" in h]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(steps, losses, label="Train loss", linewidth=1.2, alpha=0.8)
    if val_losses:
        axes[0].plot(val_steps, val_losses, marker="o", linewidth=2, label="Val loss")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Margin-ranking loss")
    axes[0].set_title("BioBERT — Loss"); axes[0].legend()

    if val_acc:
        axes[1].plot(val_steps, val_acc, marker="o", linewidth=2, color="steelblue",
                     label="Val pairwise accuracy")
        axes[1].set_xlabel("Step"); axes[1].set_ylabel("Pairwise accuracy")
        axes[1].set_ylim(0, 1); axes[1].set_title("BioBERT — Validation accuracy")
        axes[1].legend()

    plt.suptitle("BioBERT BioASQ Fine-tuning (DeepSeek distillation)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curve saved → {out_path}")


# ── evaluation step ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    losses, correct, total = [], 0, 0
    for enc_hi, enc_lo in loader:
        enc_hi = {k: v.to(device) for k, v in enc_hi.items()}
        enc_lo = {k: v.to(device) for k, v in enc_lo.items()}
        s_hi = model(**enc_hi).logits.view(-1).float()
        s_lo = model(**enc_lo).logits.view(-1).float()
        target = torch.ones_like(s_hi)
        loss = loss_fn(s_hi, s_lo, target)
        losses.append(loss.item())
        correct += (s_hi > s_lo).sum().item()
        total += s_hi.size(0)
    model.train()
    return sum(losses) / max(len(losses), 1), correct / max(total, 1)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--reranked_path", required=True)
    parser.add_argument("--queries_path", default=DEFAULT_QUERIES)
    parser.add_argument("--index_path", default=DEFAULT_INDEX)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--top_k", default=5, type=int,
                        help="Sample positives from the top-K of the permutation")
    parser.add_argument("--bottom_k", default=15, type=int,
                        help="Sample negatives from the bottom-K of the permutation")
    parser.add_argument("--pairs_per_query", default=8, type=int)
    parser.add_argument("--val_ratio", default=0.05, type=float)

    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--margin", default=0.2, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--logging_steps", default=50, type=int)
    parser.add_argument("--eval_every", default=500, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading queries from {args.queries_path}")
    queries: dict[str, str] = {}
    with open(args.queries_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            queries[r["_id"]] = r["text"]

    print("Opening Lucene index for document text lookup …")
    doc_lookup = make_doc_lookup(args.index_path)

    print(f"Building pairs from {args.reranked_path}")
    pairs = build_pairs(
        args.reranked_path, queries,
        top_k=args.top_k, bottom_k=args.bottom_k,
        pairs_per_query=args.pairs_per_query, seed=args.seed,
    )
    n_val = max(1, int(len(pairs) * args.val_ratio))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    print(f"  Train pairs : {len(train_pairs):,}")
    print(f"  Val pairs   : {len(val_pairs):,}")

    print(f"Loading model from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model).to(device)
    model.train()

    collate = build_collator(tokenizer, args.max_length)
    train_loader = DataLoader(
        PairwiseDataset(train_pairs, queries, doc_lookup),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0,
    )
    val_loader = DataLoader(
        PairwiseDataset(val_pairs, queries, doc_lookup),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0,
    )

    total_steps = len(train_loader) * args.epochs
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optim, int(total_steps * args.warmup_ratio), total_steps,
    )
    loss_fn = torch.nn.MarginRankingLoss(margin=args.margin)

    history = []
    best_val_acc = -1.0
    step = 0
    running = 0.0; running_n = 0

    for epoch in range(args.epochs):
        for enc_hi, enc_lo in train_loader:
            step += 1
            enc_hi = {k: v.to(device) for k, v in enc_hi.items()}
            enc_lo = {k: v.to(device) for k, v in enc_lo.items()}

            s_hi = model(**enc_hi).logits.view(-1).float()
            s_lo = model(**enc_lo).logits.view(-1).float()
            target = torch.ones_like(s_hi)
            loss = loss_fn(s_hi, s_lo, target)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()

            running += loss.item(); running_n += 1

            if step % args.logging_steps == 0:
                avg = running / running_n
                print(f"epoch {epoch+1} step {step}/{total_steps}  train_loss={avg:.4f}")
                history.append({"step": step, "train_loss": avg})
                running = 0.0; running_n = 0

            if step % args.eval_every == 0:
                v_loss, v_acc = evaluate(model, val_loader, device, loss_fn)
                print(f"  ↳ val_loss={v_loss:.4f}  val_pair_acc={v_acc:.4f}")
                history.append({"step": step, "val_loss": v_loss, "val_acc": v_acc})
                if v_acc > best_val_acc:
                    best_val_acc = v_acc
                    model.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)
                    print(f"  ↳ new best (val_pair_acc={v_acc:.4f}) — saved")

    # final eval & save fallback
    v_loss, v_acc = evaluate(model, val_loader, device, loss_fn)
    print(f"FINAL  val_loss={v_loss:.4f}  val_pair_acc={v_acc:.4f}")
    history.append({"step": step, "val_loss": v_loss, "val_acc": v_acc})
    if v_acc > best_val_acc:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print("Saved final model.")

    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    plot_curves(history, os.path.join(args.output_dir, "training_curve.png"))
    print(f"Done. Model in {args.output_dir}")


if __name__ == "__main__":
    main()
