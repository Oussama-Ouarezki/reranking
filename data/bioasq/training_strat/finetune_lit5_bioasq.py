"""
Fine-tune LiT5-Distill-base on BioASQ using FiD-style encoding.

Usage:
  python scripts/finetune_lit5_bioasq.py --train lit5_dataset1_train.jsonl --output checkpoint1
  python scripts/finetune_lit5_bioasq.py --train lit5_dataset2_train.jsonl --output checkpoint2

Validation is always lit5_validation.jsonl. Best checkpoint (lowest val loss) is saved.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH       = Path("checkpoints/LiT5-Distill-base")
DATA_DIR         = Path("data/bioasq/training_strat")
VAL_FILE         = "lit5_validation.jsonl"

TEXT_MAXLENGTH   = 64
ANSWER_MAXLENGTH = 64
N_PASSAGES       = 20
BATCH_SIZE       = 1
GRAD_ACCUM       = 4
EPOCHS           = 3
LR               = 5e-5
WARMUP_RATIO     = 0.1
USE_BF16         = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
QUERY_PREFIX     = "Search Query:"
PASSAGE_PREFIX   = "Passage:"
SUFFIX           = " Relevance Ranking:"


# ── Dataset & Collator ────────────────────────────────────────────────────────
class LiT5Dataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def load_records(path):
    records = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class FiDCollator:
    def __init__(self, tokenizer, text_maxlength, answer_maxlength, n_passages):
        self.tokenizer        = tokenizer
        self.text_maxlength   = text_maxlength
        self.answer_maxlength = answer_maxlength
        self.n_passages       = n_passages

    def __call__(self, batch):
        all_input_ids, all_attn_masks, all_labels = [], [], []

        for rec in batch:
            query  = rec["question"]
            ctxs   = rec["ctxs"][:self.n_passages]
            target = rec["target"]

            while len(ctxs) < self.n_passages:
                ctxs.append({"text": ""})

            passage_strings = [
                f"{QUERY_PREFIX} {query} {PASSAGE_PREFIX} [{i+1}] {ctx['text']}{SUFFIX}"
                for i, ctx in enumerate(ctxs)
            ]

            enc = self.tokenizer(
                passage_strings,
                max_length=self.text_maxlength,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            all_input_ids.append(enc.input_ids.view(1, -1))
            all_attn_masks.append(enc.attention_mask.view(1, -1))

            tgt    = self.tokenizer(target, max_length=self.answer_maxlength,
                                    padding="max_length", truncation=True, return_tensors="pt")
            labels = tgt.input_ids.squeeze(0)
            labels[labels == self.tokenizer.pad_token_id] = -100
            all_labels.append(labels)

        return {
            "input_ids":      torch.cat(all_input_ids,  dim=0),
            "attention_mask": torch.cat(all_attn_masks, dim=0),
            "labels":         torch.stack(all_labels,   dim=0),
        }


# ── Validation ────────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_steps = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss  += outputs.loss.item()
            total_steps += 1
    model.train()
    return total_loss / total_steps if total_steps > 0 else float("inf")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",  required=True, help="Training JSONL filename in DATA_DIR")
    parser.add_argument("--output", required=True, help="Output checkpoint folder name in DATA_DIR")
    args = parser.parse_args()

    output_dir = DATA_DIR / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Train:  {args.train}")
    print(f"Val:    {VAL_FILE}")
    print(f"Output: {output_dir}")

    # Load model
    print("Loading model …")
    tokenizer = T5Tokenizer.from_pretrained(str(MODEL_PATH), legacy=False, use_fast=True)
    model     = T5ForConditionalGeneration.from_pretrained(str(MODEL_PATH))
    if USE_BF16:
        model = model.bfloat16()
        print("Using bf16")
    model.to(device)
    model.gradient_checkpointing_enable()
    torch.cuda.empty_cache()

    # Load data
    collator   = FiDCollator(tokenizer, TEXT_MAXLENGTH, ANSWER_MAXLENGTH, N_PASSAGES)
    train_recs = load_records(DATA_DIR / args.train)
    val_recs   = load_records(DATA_DIR / VAL_FILE)
    print(f"Train: {len(train_recs)} records  |  Val: {len(val_recs)} records")

    train_loader = DataLoader(LiT5Dataset(train_recs), batch_size=BATCH_SIZE,
                              shuffle=True,  collate_fn=collator)
    val_loader   = DataLoader(LiT5Dataset(val_recs),   batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collator)

    # Optimizer
    optimizer    = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps  = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_loss = float("inf")
    optimizer.zero_grad()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for step, batch in enumerate(pbar):
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            (outputs.loss / GRAD_ACCUM).backward()
            epoch_loss += outputs.loss.item()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=f"{outputs.loss.item():.4f}")

        # flush remaining gradients
        if len(train_loader) % GRAD_ACCUM != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train = epoch_loss / len(train_loader)
        avg_val   = evaluate(model, val_loader, device)
        print(f"  Epoch {epoch+1}  train loss: {avg_train:.4f}  val loss: {avg_val:.4f}", end="")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            print("  ✓ saved best")
        else:
            print()

    print(f"\nDone. Best val loss: {best_val_loss:.4f} → {output_dir}")


if __name__ == "__main__":
    main()
