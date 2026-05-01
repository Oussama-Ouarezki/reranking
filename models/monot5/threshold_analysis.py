"""monoT5 threshold analysis — BioASQ.

For every query:
  1. BM25 retrieves top-50 candidates
  2. monoT5 scores all 50  →  P(true) probability per doc
  3. For each threshold t in {0.1, 0.2, …, 0.9}:
       accepted  = docs with score >= t
       precision = |gold ∩ accepted| / |accepted|    (0 if nothing accepted)
       recall    = |gold ∩ accepted| / |gold in qrels|

Mean precision and recall across all queries → grouped bar chart.

Usage (from project root):
  python models/monot5/threshold_analysis.py
  python models/monot5/threshold_analysis.py --n 30        # first 30 queries
  python models/monot5/threshold_analysis.py --batch-size 16
"""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]
QUERIES_PATH = ROOT / "data/bioasq/raw/Task13BGoldenEnriched/queries_full.jsonl"
QRELS_PATH   = ROOT / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"
CORPUS_PATH  = ROOT / "data/bioasq/pubmed_full/full/corpus_full.jsonl"
LUCENE_INDEX = ROOT / "data/bm25_indexing_full/corpus_full/lucene_index"
MONOT5_CKPT  = ROOT / "checkpoints/monot5-base-msmarco-100k"
OUT_DIR      = ROOT / "models/monot5"

BM25_TOP_K   = 50
THRESHOLDS   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Java must be set before any pyserini import
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
os.environ["PATH"] = "/usr/lib/jvm/java-21-openjdk-amd64/bin:" + os.environ.get("PATH", "")


# ── data loaders ──────────────────────────────────────────────────────────────

def load_queries(path):
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def load_qrels(path):
    """Returns {qid: set of relevant docids}."""
    qrels = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if int(row["score"]) > 0:
                qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])
    return qrels


def load_corpus(path):
    """Returns {docid: 'title text'} string for monoT5 input."""
    corpus = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            corpus[doc["_id"]] = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
    return corpus


# ── monoT5 scorer ─────────────────────────────────────────────────────────────

def build_scorer(checkpoint, batch_size):
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading monoT5 on {device} …")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    model = T5ForConditionalGeneration.from_pretrained(
        str(checkpoint),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    true_id  = tokenizer.convert_tokens_to_ids("▁true")
    false_id = tokenizer.convert_tokens_to_ids("▁false")

    def score_passages(query, passages):
        all_scores = []
        for i in range(0, len(passages), batch_size):
            batch = passages[i : i + batch_size]
            prompts = [f"Query: {query} Document: {p} Relevant:" for p in batch]
            enc = tokenizer(prompts, padding=True, truncation=True,
                            max_length=512, return_tensors="pt").to(device)
            dec = torch.zeros((len(batch), 1), dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(input_ids=enc["input_ids"],
                               attention_mask=enc["attention_mask"],
                               decoder_input_ids=dec).logits   # (B, 1, vocab)
            tf = logits[:, 0, [true_id, false_id]]             # (B, 2)
            probs = torch.softmax(tf, dim=-1)
            all_scores.extend(probs[:, 0].cpu().tolist())
        return all_scores

    return score_passages


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",          type=int, default=None,
                        help="Limit to first N queries (default: all)")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    print("Loading queries …")
    all_queries = load_queries(QUERIES_PATH)
    qrels       = load_qrels(QRELS_PATH)
    queries     = [q for q in all_queries if q["_id"] in qrels]
    if args.n:
        queries = queries[: args.n]
    print(f"  {len(queries)} queries with relevance judgments")

    print("Loading corpus …")
    corpus = load_corpus(CORPUS_PATH)
    print(f"  {len(corpus):,} documents loaded")

    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher(str(LUCENE_INDEX))
    searcher.set_bm25(0.7, 0.9)

    score_passages = build_scorer(MONOT5_CKPT, args.batch_size)

    # precision[thr], recall[thr], accepted[thr] accumulate per-query values
    precision_sums = {t: 0.0 for t in THRESHOLDS}
    recall_sums    = {t: 0.0 for t in THRESHOLDS}
    accepted_sums  = {t: 0   for t in THRESHOLDS}
    n_queries_with_gold = 0   # queries where gold docs exist

    for idx, q in enumerate(queries, 1):
        qid  = q["_id"]
        text = q["text"]
        gold = qrels[qid]     # set of relevant docids from qrels

        hits = searcher.search(text, k=BM25_TOP_K)
        if not hits:
            continue

        docids   = [h.docid for h in hits]
        passages = [corpus.get(did, "") for did in docids]
        scores   = score_passages(text, passages)

        n_gold_in_qrels = len(gold)
        if n_gold_in_qrels == 0:
            continue
        n_queries_with_gold += 1

        gold_in_pool = sum(1 for did in docids if did in gold)
        print(f"  [{idx:3}/{len(queries)}] {text[:60]:<60} "
              f"gold in top-{BM25_TOP_K}: {gold_in_pool}/{n_gold_in_qrels}",
              flush=True)

        for thr in THRESHOLDS:
            accepted     = [(did, did in gold) for did, s in zip(docids, scores) if s >= thr]
            n_accepted   = len(accepted)
            n_gold_hit   = sum(1 for _, is_g in accepted if is_g)

            p = n_gold_hit / n_accepted       if n_accepted        else 0.0
            r = n_gold_hit / n_gold_in_qrels

            precision_sums[thr] += p
            recall_sums[thr]    += r
            accepted_sums[thr]  += n_accepted

    # mean across queries
    mean_p = {t: precision_sums[t] / n_queries_with_gold for t in THRESHOLDS}
    mean_r = {t: recall_sums[t]    / n_queries_with_gold for t in THRESHOLDS}
    mean_a = {t: accepted_sums[t]  / n_queries_with_gold for t in THRESHOLDS}

    # ── print table ───────────────────────────────────────────────────────────
    print(f"\n{'Threshold':>10}  {'Mean Precision':>14}  {'Mean Recall':>12}  {'Mean Accepted':>14}")
    print("─" * 58)
    for t in THRESHOLDS:
        print(f"{t:>10.1f}  {mean_p[t]:>14.4f}  {mean_r[t]:>12.4f}  {mean_a[t]:>14.2f}")

    # ── save CSV ──────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "threshold_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "mean_precision", "mean_recall", "mean_accepted"])
        for t in THRESHOLDS:
            w.writerow([t, round(mean_p[t], 4), round(mean_r[t], 4), round(mean_a[t], 2)])
    print(f"\nCSV → {csv_path}")

    # ── bar chart ─────────────────────────────────────────────────────────────
    sns.set_theme(style="darkgrid")
    plt.style.use("ggplot")

    x      = np.arange(len(THRESHOLDS))
    width  = 0.35
    labels = [str(t) for t in THRESHOLDS]
    p_vals = [mean_p[t] for t in THRESHOLDS]
    r_vals = [mean_r[t] for t in THRESHOLDS]

    a_vals = [mean_a[t] for t in THRESHOLDS]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars_p = ax.bar(x - width / 2, p_vals, width, label="Mean Precision", color="#4c72b0")
    bars_r = ax.bar(x + width / 2, r_vals, width, label="Mean Recall",    color="#dd8452")

    # value labels on top of each bar
    for bar in bars_p:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_r:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    # secondary axis: mean number of accepted documents
    ax2 = ax.twinx()
    ax2.plot(x, a_vals, "D--", color="#2ca02c", linewidth=1.8, markersize=6,
             label="Mean accepted docs")
    for xi, val in zip(x, a_vals):
        ax2.text(xi, val + 0.3, f"{val:.1f}", ha="center", va="bottom",
                 fontsize=8, color="#2ca02c")
    ax2.set_ylabel("Mean accepted documents", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.set_ylim(0, BM25_TOP_K * 1.25)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("monoT5 score threshold")
    ax.set_ylabel("Score (mean across queries)")
    ax.set_title(f"monoT5 Precision & Recall & Accepted docs per threshold  "
                 f"(BM25 top-{BM25_TOP_K} pool, {n_queries_with_gold} queries)")
    ax.set_ylim(0, 1.12)

    # merge legends from both axes
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plot_path = OUT_DIR / "threshold_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot → {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
