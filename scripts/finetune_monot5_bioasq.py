"""Fine-tune monoT5 on BioASQ pointwise data.

Starting checkpoint : checkpoints/monot5-base-msmarco-100k  (MS-MARCO pretrained)
Training data       : data/bioasq/finetune/pointwise_monot5.tsv
                      columns: query \t passage \t label  (1=relevant, 0=not)
Output              : checkpoints/monot5-bioasq-finetuned/
                      + training_curve.png  (loss & accuracy vs step)

Input format fed to the model (same as monoT5 inference):
    Query: {query} Document: {passage} Relevant:
Target:
    "true"  if label == 1
    "false" if label == 0

Usage
-----
python scripts/finetune_monot5_bioasq.py \
    --triples_path data/bioasq/finetune/pointwise_monot5.tsv \
    --output_model_path checkpoints/monot5-bioasq-finetuned
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


# ── dataset ────────────────────────────────────────────────────────────────────

class PointwiseDataset(Dataset):
    def __init__(self, samples: list[tuple[str, str, str]]):
        # samples: (query, passage, label_str)  label_str ∈ {"true", "false"}
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # FIX: return tuple instead of dict — robust across Transformers versions
        q, p, label = self.samples[idx]
        return (
            f"Query: {q} Document: {p} Relevant:",
            label,
        )


# ── data helpers ───────────────────────────────────────────────────────────────

def load_samples(path: str, max_samples: int | None = None) -> list[tuple[str, str, str]]:
    samples = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            label_str = "true" if row["label"].strip() == "1" else "false"
            samples.append((row["query"], row["passage"], label_str))
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def stratified_split(
    samples: list, val_ratio: float, seed: int = 42
) -> tuple[list, list]:
    """Split preserving true/false ratio in both partitions."""
    rng = random.Random(seed)
    pos = [s for s in samples if s[2] == "true"]
    neg = [s for s in samples if s[2] == "false"]
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
        # FIX: handle both tuple and dict batch items across Transformers versions
        if isinstance(batch[0], dict):
            texts  = [ex["text"]   for ex in batch]
            labels = [ex["labels"] for ex in batch]
        else:
            texts  = [ex[0] for ex in batch]
            labels = [ex[1] for ex in batch]

        enc = tokenizer(
            texts,
            padding=True, truncation=True, max_length=512, return_tensors="pt",
        )
        enc["labels"] = tokenizer(labels, return_tensors="pt")["input_ids"]
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
    ax.set_title("monoT5 — Loss")
    ax.legend()

    # — accuracy panel —
    if has_acc:
        ax2 = axes[1]
        ax2.plot(eval_acc_steps, eval_acc, color="steelblue", linewidth=2,
                 marker="o", markersize=4, label="Validation accuracy")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("monoT5 — Validation Accuracy")
        ax2.set_ylim(0, 1)
        ax2.legend()

    plt.suptitle("monoT5 BioASQ Fine-tuning", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curve saved → {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default=None,
                        help="Starting checkpoint (default: checkpoints/monot5-base-msmarco-100k)")
    parser.add_argument("--triples_path", required=True,
                        help="pointwise_monot5.tsv path")
    parser.add_argument("--output_model_path", required=True,
                        help="Where to save the fine-tuned model")
    parser.add_argument("--val_split", default=0.1, type=float,
                        help="Fraction of data held out for validation (default 0.1)")
    parser.add_argument("--max_samples", default=None, type=int,
                        help="Cap total samples before the val split (quick tests)")
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--per_device_train_batch_size", default=8, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--warmup_steps", default=500, type=int)
    parser.add_argument("--save_every_n_steps", default=0, type=int,
                        help="Save/eval every N steps. 0 = once per epoch.")
    parser.add_argument("--logging_steps", default=50, type=int)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    checkpoint = args.base_model or os.path.join(ROOT, "checkpoints/monot5-base-msmarco-100k")
    print(f"Loading model from: {checkpoint}")
    model     = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    model.gradient_checkpointing_enable()   # ← add this line
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    print(f"Loading data from: {args.triples_path}")
    all_samples = load_samples(args.triples_path, args.max_samples)
    train_samples, val_samples = stratified_split(all_samples, args.val_split)

    print(f"  Train : {len(train_samples):>7,}  "
          f"(pos={sum(1 for *_,l in train_samples if l=='true'):,}  "
          f"neg={sum(1 for *_,l in train_samples if l=='false'):,})")
    print(f"  Val   : {len(val_samples):>7,}  "
          f"(pos={sum(1 for *_,l in val_samples if l=='true'):,}  "
          f"neg={sum(1 for *_,l in val_samples if l=='false'):,})")

    train_dataset = PointwiseDataset(train_samples)
    val_dataset   = PointwiseDataset(val_samples)
    collator      = build_collator(tokenizer, device)

    # save/eval strategy must match for load_best_model_at_end
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
        optim="adafactor",       # FIX: replaces deprecated adafactor=True
        gradient_checkpointing=True,
        seed=42,
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,  # FIX: replaces deprecated tokenizer=
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