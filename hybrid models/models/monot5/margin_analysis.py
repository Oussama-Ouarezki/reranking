"""monoT5 dynamic margin analysis — BioASQ.

For each margin in {0.001, 0.002, …, 0.010}: score all BM25 top-50 docs with
monoT5, look at the top-20 ranked docs, and count how many adjacent pairs
have a score gap < margin (flagged as "uncertain").

Chart 1 — Mean uncertain docs per margin (bar chart).
Chart 2 — Mean duoT5 pairwise operations per query, n*(n-1) ordered pairs (bar chart).
Helps pick DYNAMIC_MARGIN for MonoDynamicDuoLiT5Cascade.

Scores are cached in margin_scores_cache.json — re-runs skip monoT5 scoring.

Usage (from project root):
  python models/monot5/margin_analysis.py
  python models/monot5/margin_analysis.py --n 50    # first 50 queries
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]
QUERIES_PATH = ROOT / "data/bioasq/raw/Task13BGoldenEnriched/queries_full.jsonl"
QRELS_PATH   = ROOT / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"
CORPUS_PATH  = ROOT / "data/bioasq/pubmed_full/full/corpus_full.jsonl"
LUCENE_INDEX = ROOT / "data/bm25_indexing_full/corpus_full/lucene_index"
MONOT5_CKPT  = ROOT / "checkpoints/monot5-base-msmarco-100k"
OUT_DIR      = ROOT / "models/monot5"
CACHE_PATH   = OUT_DIR / "margin_scores_cache.json"

BM25_TOP_K  = 50
SCAN_TOP    = 20   # only look at top-N docs (mirrors DYNAMIC_SCAN_TOP in cascade)
MARGINS     = [round(m / 1000, 3) for m in range(1, 11)]  # 0.001 … 0.010

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
os.environ["PATH"] = "/usr/lib/jvm/java-21-openjdk-amd64/bin:" + os.environ.get("PATH", "")


# ── loaders ───────────────────────────────────────────────────────────────────

def load_queries(path: Path) -> list[dict]:
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def load_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if int(row["score"]) > 0:
                qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])
    return qrels


def load_corpus(path: Path) -> dict[str, str]:
    corpus: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            corpus[doc["_id"]] = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
    return corpus


# ── score cache ───────────────────────────────────────────────────────────────

def load_cache(path: Path) -> dict[str, dict[str, float]]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(path: Path, cache: dict[str, dict[str, float]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f)


# ── monoT5 scorer ─────────────────────────────────────────────────────────────

def build_scorer(checkpoint: Path, batch_size: int):
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

    def score_passages(query: str, passages: list[str]) -> list[float]:
        all_scores: list[float] = []
        for i in range(0, len(passages), batch_size):
            batch = passages[i : i + batch_size]
            prompts = [f"Query: {query} Document: {p} Relevant:" for p in batch]
            enc = tokenizer(prompts, padding=True, truncation=True,
                            max_length=512, return_tensors="pt").to(device)
            dec = torch.zeros((len(batch), 1), dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(input_ids=enc["input_ids"],
                               attention_mask=enc["attention_mask"],
                               decoder_input_ids=dec).logits
            tf = logits[:, 0, [true_id, false_id]]
            probs = torch.softmax(tf, dim=-1)
            all_scores.extend(probs[:, 0].cpu().tolist())
        return all_scores

    return score_passages


# ── count uncertain docs for one query ───────────────────────────────────────

def count_uncertain(scores: list[float], margin: float) -> int:
    """Count docs flagged as uncertain within the sorted scores."""
    uncertain: set[int] = set()
    for i in range(len(scores) - 1):
        gap = scores[i] - scores[i + 1]  # always >= 0 (sorted desc)
        if gap < margin:
            uncertain.add(i)
            uncertain.add(i + 1)
    return len(uncertain)


# ── helpers ───────────────────────────────────────────────────────────────────

def _bar_chart(ax: Any, labels: list[str], values: list[float],
               color: str, ylabel: str, title: str, ylim_pad: float = 1.2) -> None:
    bars = ax.bar(labels, values, color=color, width=0.6)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{bar.get_height():.1f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_xlabel("Score gap margin")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(values) * ylim_pad + 1)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",          type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    print("Loading queries …")
    all_queries = load_queries(QUERIES_PATH)
    qrels       = load_qrels(QRELS_PATH)
    queries     = [q for q in all_queries if q["_id"] in qrels]
    if args.n:
        queries = queries[: args.n]
    print(f"  {len(queries)} queries")

    print("Loading corpus …")
    corpus = load_corpus(CORPUS_PATH)

    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher(str(LUCENE_INDEX))
    searcher.set_bm25(0.7, 0.9)

    # ── load score cache (skip monoT5 for already-scored queries) ─────────────
    cache = load_cache(CACHE_PATH)
    n_cached = sum(1 for q in queries if q["_id"] in cache)
    print(f"  Score cache: {n_cached}/{len(queries)} queries already cached "
          f"({CACHE_PATH.name})")

    # build the scorer only if there are uncached queries
    uncached_queries = [q for q in queries if q["_id"] not in cache]
    score_passages = build_scorer(MONOT5_CKPT, args.batch_size) if uncached_queries else None

    # per-query tracking
    uncertain_counts: dict[float, list[int]]   = {m: [] for m in MARGINS}
    pairwise_counts:  dict[float, list[int]]   = {m: [] for m in MARGINS}

    for idx, q in enumerate(queries, 1):
        qid  = q["_id"]
        text = q["text"]

        hits = searcher.search(text, k=BM25_TOP_K)
        if not hits:
            continue
        docids = [h.docid for h in hits]

        if qid in cache:
            # restore scores in BM25 order from cache
            scores = [cache[qid].get(did, 0.0) for did in docids]
            tag = "[cache]"
        else:
            passages = [corpus.get(did, "") for did in docids]
            scores = score_passages(text, passages)  # type: ignore[misc]
            cache[qid] = {did: s for did, s in zip(docids, scores)}
            save_cache(CACHE_PATH, cache)
            tag = "[scored]"

        sorted_scores = sorted(scores, reverse=True)
        top_scores    = sorted_scores[:SCAN_TOP]

        for margin in MARGINS:
            n = count_uncertain(top_scores, margin)
            uncertain_counts[margin].append(n)
            pairwise_counts[margin].append(n * (n - 1))  # duoT5 ordered pairs

        print(f"  [{idx:3}/{len(queries)}] {tag} {text[:60]:<60}", flush=True)

    # ── compute means ─────────────────────────────────────────────────────────
    means      = {m: sum(uncertain_counts[m]) / len(uncertain_counts[m])
                  for m in MARGINS if uncertain_counts[m]}
    mean_pairs = {m: sum(pairwise_counts[m]) / len(pairwise_counts[m])
                  for m in MARGINS if pairwise_counts[m]}

    # ── print table ───────────────────────────────────────────────────────────
    print(f"\n{'Margin':>8}  {'Mean uncertain':>14}  {'Mean pairs':>10}  {'Max':>5}  {'Min':>5}")
    print("─" * 52)
    for m in MARGINS:
        vals = uncertain_counts[m]
        if vals:
            print(f"{m:>8.3f}  {means[m]:>14.2f}  {mean_pairs[m]:>10.1f}  "
                  f"{max(vals):>5}  {min(vals):>5}")

    # ── save CSV ──────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "margin_analysis.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["margin", "mean_uncertain", "mean_pairwise_ops",
                    "max_uncertain", "min_uncertain", "n_queries"])
        for m in MARGINS:
            vals = uncertain_counts[m]
            if vals:
                w.writerow([m, round(means[m], 2), round(mean_pairs[m], 1),
                             max(vals), min(vals), len(vals)])
    print(f"\nCSV → {csv_path}")

    # ── charts ────────────────────────────────────────────────────────────────
    sns.set_theme(style="darkgrid")
    plt.style.use("ggplot")

    labels      = [str(m) for m in MARGINS]
    unc_values  = [means[m]      for m in MARGINS]
    pair_values = [mean_pairs[m] for m in MARGINS]

    # Chart 1 — mean uncertain docs
    fig1: Any
    ax1: Any
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    _bar_chart(ax1, labels, unc_values, "#4c72b0",
               f"Mean uncertain docs (top {SCAN_TOP})",
               f"monoT5 — mean uncertain docs per margin  "
               f"(BM25 top-{BM25_TOP_K}, scan top-{SCAN_TOP}, {len(queries)} queries)")
    ax1.axhline(20, color="red",    linestyle="--", linewidth=1, label="duoT5 cap (20)")
    ax1.axhline(10, color="orange", linestyle=":",  linewidth=1, label="suggested max (10)")
    ax1.legend()
    fig1.tight_layout()
    plot1_path = OUT_DIR / "margin_plot.png"
    fig1.savefig(plot1_path, dpi=150)
    print(f"Plot 1 → {plot1_path}")

    # Chart 2 — mean duoT5 pairwise operations  n*(n-1)
    fig2: Any
    ax2: Any
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    _bar_chart(ax2, labels, pair_values, "#dd8452",
               "Mean duoT5 pairwise operations  n·(n−1)",
               f"monoT5 — mean duoT5 operations per margin  "
               f"(BM25 top-{BM25_TOP_K}, scan top-{SCAN_TOP}, {len(queries)} queries)")
    # reference: duoT5 cap is 20 docs → 20*19 = 380 ordered pairs
    ax2.axhline(380, color="red",    linestyle="--", linewidth=1, label="cap: 20 docs → 380 pairs")
    ax2.axhline(90,  color="orange", linestyle=":",  linewidth=1, label="sweet spot: 10 docs → 90 pairs")
    ax2.legend()
    fig2.tight_layout()
    plot2_path = OUT_DIR / "margin_pairs_plot.png"
    fig2.savefig(plot2_path, dpi=150)
    print(f"Plot 2 → {plot2_path}")

    plt.show()


if __name__ == "__main__":
    main()
