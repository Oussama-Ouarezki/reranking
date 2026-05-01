"""Fine-tune duoT5 on BioASQ pairwise data.

Starting checkpoint : checkpoints/duot5-base-msmarco  (MS-MARCO pretrained)
Training data       : data/bioasq/finetune/pairwise_duot5.tsv
                      columns: query \t doc_a \t doc_b \t label
                               label=0  →  doc_a wins  → target "true"
                               label=1  →  doc_b wins  → target "false"
Output              : checkpoints/duot5-bioasq-finetuned/
                      + training_curve.png  (loss & accuracy vs step)

The pairwise dataset already contains both (pos,neg)→"true" and
(neg,pos)→"false" orderings, so label distribution is perfectly 50/50.

Usage
-----
python scripts/finetune_duot5_bioasq.py \
    --triples_path data/bioasq/finetune/pairwise_duot5.tsv \
    --output_model_path checkpoints/duot5-bioasq-finetuned
"""

import argparse
import csv
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOC_MAX_WORDS = 200   # matches inference truncation in duot5.py
MAX_INPUT_LEN = 1024  # matches MAX_LENGTH in duot5.py


# ── dataset ────────────────────────────────────────────────────────────────────

def _truncate(text: str, max_words: int = DOC_MAX_WORDS) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


class PairwiseDataset(Dataset):
    def __init__(self, samples: list[tuple[str, str, str, str]]):
        # samples: (query, doc_a, doc_b, label_str)  label_str ∈ {"true", "false"}
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        q, da, db, label = self.samples[idx]
        return {
            "text": (
                f"Query: {q} "
                f"Document0: {_truncate(da)} "
                f"Document1: {_truncate(db)} "
                f"Relevant:"
            ),
            "labels": label,
        }


# ── data helpers ───────────────────────────────────────────────────────────────

def load_samples(path: str, max_samples: int | None = None) -> list[tuple[str, str, str, str]]:
    samples = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # label=0 → doc_a wins → Document0 more relevant → "true"
            # label=1 → doc_b wins → Document0 NOT more relevant → "false"
            label_str = "true" if row["label"].strip() == "0" else "false"
            samples.append((row["query"], row["doc_a"], row["doc_b"], label_str))
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def stratified_split(
    samples: list, val_ratio: float, seed: int = 42
) -> tuple[list, list]:
    """Split preserving true/false ratio in both partitions."""
    rng = random.Random(seed)
    pos = [s for s in samples if s[3] == "true"]
    neg = [s for s in samples if s[3] == "false"]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_val_pos = max(1, int(len(pos) * val_ratio))
    n_val_neg = max(1, int(len(neg) * val_ratio))
    val   = pos[:n_val_pos]   + neg[:n_val_neg]
    train = pos[n_val_pos:]   + neg[n_val_neg:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ── collator ───────────────────────────────────────────────────────────────────

def build_collator(tokenizer, device):
    def collate(batch):
        enc = tokenizer(
            [ex["text"] for ex in batch],
            padding=True, truncation=True, max_length=MAX_INPUT_LEN, return_tensors="pt",
        )
        enc["labels"] = tokenizer(
            [ex["labels"] for ex in batch], return_tensors="pt"
        )["input_ids"]
        return {k: v.to(device) for k, v in enc.items()}
    return collate


# ── metrics ────────────────────────────────────────────────────────────────────

def make_compute_metrics(tokenizer):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        decoded_preds  = tokenizer.batch_decode(preds,   skip_special_tokens=True)
        labels         = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels,  skip_special_tokens=True)
        correct = sum(
            p.strip() == l.strip()
            for p, l in zip(decoded_preds, decoded_labels)
        )
        return {"accuracy": correct / max(len(decoded_preds), 1)}
    return compute_metrics


# ── plotting ───────────────────────────────────────────────────────────────────

def plot_training_curves(log_history: list[dict], out_path: str) -> None:
    sns.set_theme(style="darkgrid")
    plt.style.use("ggplot")

    train_steps, train_loss = [], []
    eval_steps,  eval_loss  = [], []
    eval_acc_steps, eval_acc = [], []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry["step"])
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_loss.append(entry["eval_loss"])
        if "eval_accuracy" in entry:
            eval_acc_steps.append(entry["step"])
            eval_acc.append(entry["eval_accuracy"])

    has_acc = bool(eval_acc)
    fig, axes = plt.subplots(1, 2 if has_acc else 1, figsize=(14 if has_acc else 7, 5))
    if not has_acc:
        axes = [axes]

    # — loss panel —
    ax = axes[0]
    ax.plot(train_steps, train_loss, label="Train loss",      linewidth=1.5, alpha=0.8)
    ax.plot(eval_steps,  eval_loss,  label="Validation loss", linewidth=2,
            marker="o", markersize=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("duoT5 — Loss")
    ax.legend()

    # — accuracy panel —
    if has_acc:
        ax2 = axes[1]
        ax2.plot(eval_acc_steps, eval_acc, color="steelblue", linewidth=2,
                 marker="o", markersize=4, label="Validation accuracy")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("duoT5 — Validation Accuracy")
        ax2.set_ylim(0, 1)
        ax2.legend()

    plt.suptitle("duoT5 BioASQ Fine-tuning", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curve saved → {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default=None,
                        help="Starting checkpoint (default: checkpoints/duot5-base-msmarco)")
    parser.add_argument("--triples_path", required=True,
                        help="pairwise_duot5.tsv path")
    parser.add_argument("--output_model_path", required=True,
                        help="Where to save the fine-tuned model")
    parser.add_argument("--val_split", default=0.1, type=float,
                        help="Fraction of data held out for validation (default 0.1)")
    parser.add_argument("--max_samples", default=None, type=int,
                        help="Cap total samples before the val split (quick tests)")
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--per_device_train_batch_size", default=4, type=int,
                        help="Lower default than monoT5 — inputs are ~2x longer")
    parser.add_argument("--gradient_accumulation_steps", default=32, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--warmup_steps", default=500, type=int)
    parser.add_argument("--save_every_n_steps", default=0, type=int,
                        help="Save/eval every N steps. 0 = once per epoch.")
    parser.add_argument("--logging_steps", default=50, type=int)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    checkpoint = args.base_model or os.path.join(ROOT, "checkpoints/duot5-base-msmarco")
    print(f"Loading model from: {checkpoint}")
    model     = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    print(f"Loading data from: {args.triples_path}")
    all_samples = load_samples(args.triples_path, args.max_samples)
    train_samples, val_samples = stratified_split(all_samples, args.val_split)

    n_tr_t = sum(1 for *_, l in train_samples if l == "true")
    n_va_t = sum(1 for *_, l in val_samples   if l == "true")
    print(f"  Train : {len(train_samples):>7,}  (true={n_tr_t:,}  false={len(train_samples)-n_tr_t:,})")
    print(f"  Val   : {len(val_samples):>7,}  (true={n_va_t:,}  false={len(val_samples)-n_va_t:,})")

    train_dataset = PairwiseDataset(train_samples)
    val_dataset   = PairwiseDataset(val_samples)
    collator      = build_collator(tokenizer, device)

    strategy   = "steps" if args.save_every_n_steps else "epoch"
    step_count = args.save_every_n_steps or 1

    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_model_path,
        do_train=True,
        do_eval=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=5e-5,
        warmup_steps=args.warmup_steps,
        save_strategy=strategy,
        save_steps=step_count,
        eval_strategy=strategy,
        eval_steps=step_count,
        logging_steps=args.logging_steps,
        predict_with_generate=True,
        adafactor=True,
        seed=42,
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=make_compute_metrics(tokenizer),
    )

    print("Starting training …")
    trainer.train()

    print(f"Saving fine-tuned model to: {args.output_model_path}")
    trainer.save_model(args.output_model_path)
    tokenizer.save_pretrained(args.output_model_path)
    trainer.save_state()

    curve_path = os.path.join(args.output_model_path, "training_curve.png")
    plot_training_curves(trainer.state.log_history, curve_path)
    print("Done.")


if __name__ == "__main__":
    main()
