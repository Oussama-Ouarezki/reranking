"""Centralized paths and runtime config."""

from pathlib import Path
import os

# Project root: .../reranking_project/
ROOT = Path(__file__).resolve().parents[2]

# Data paths
CORPUS_PATH = ROOT / "data/bioasq/pubmed_full/full/corpus_full.jsonl"
QUERIES_PATH = ROOT / "data/bioasq/raw/Task13BGoldenEnriched/queries_full.jsonl"
QRELS_PATH = ROOT / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"
#QUERIES_PATH = ROOT / "data/bioasq/processed/queries.jsonl"
#QRELS_PATH = ROOT / "data/bioasq/processed/qrels.tsv"
LUCENE_INDEX = ROOT / "data/bm25_indexing_full/corpus_full/lucene_index"

# BM25 params (BioASQ standard)
BM25_K1 = 0.7
BM25_B = 0.9

# Reranker checkpoints
CHECKPOINTS = {
    "monot5": ROOT / "checkpoints/monot5-base-msmarco-100k",
    "duot5": ROOT / "checkpoints/duot5-base-msmarco",
    "lit5": ROOT / "checkpoints/lit5_top20_oracle/ep08_nDCGat5_0.7864",
    "monot5_bioasq": ROOT / "checkpoints/monot5-bioasq-finetuned",
    "duot5_bioasq": ROOT / "checkpoints/duot5-bioasq-finetuned",
    "lit5_finetuned": ROOT / "checkpoints/lit5_top20_oracle/ep07_nDCGat5_0.7619",
    "lit5_bioasq_lora": ROOT / "checkpoints/lit5_bioasq_lora_local/epoch_2",
    "lit5_bioasq_lora_e1": ROOT / "checkpoints/lit5_bioasq_lora_local/epoch_1",
    "lit5_bioasq_lora_e3": ROOT / "checkpoints/lit5_bioasq_lora_local/epoch_3",
    "lit5_bioasq_lora_kaggle": ROOT / "checkpoints/lit5_bioasq_lora_kaggle/epoch_3",
    "lit5_bioasq_lora_kaggle_e1": ROOT / "checkpoints/lit5_bioasq_lora_kaggle/epoch_1",
    "lit5_bioasq_lora_kaggle_e2": ROOT / "checkpoints/lit5_bioasq_lora_kaggle/epoch_2",
    "lit5_bioasq_lora_kaggle_e3": ROOT / "checkpoints/lit5_bioasq_lora_kaggle/epoch_3",
    "lit5_bioasq_lora_kaggle_e4": ROOT / "checkpoints/lit5_bioasq_lora_kaggle/epoch_4",
}

# Java for Pyserini (must be set before importing pyserini)
JAVA_HOME = "/usr/lib/jvm/java-21-openjdk-amd64"

# Ollama
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

# Bulk eval cache
CACHE_DIR = ROOT / "application/cache"
EVAL_CACHE = CACHE_DIR / "eval_results.json"

# Retrieval defaults
BM25_RETRIEVE_K = 50  # always retrieve 50 then rerank → truncate to top_k
DEFAULT_TOP_K = 10
