"""
Download all reranker checkpoints from HuggingFace.

Models included:
  monoT5  - castorini/monot5-base-msmarco-10k   (best for fine-tuning)
  monoT5  - castorini/monot5-base-msmarco        (100k steps, best for MS MARCO eval)
  duoT5   - castorini/duot5-base-msmarco         (pairwise, used after monoT5)
  LiT5    - castorini/LiT5-Distill-base          (listwise distillation)

Usage:
    python download_checkpoints.py                      # download all
    python download_checkpoints.py --model monot5_10k  # single model
    python download_checkpoints.py --model monot5_10k monot5_100k lit5_distill

Available --model keys:
    monot5_10k, monot5_100k, duot5, lit5_distill
"""

import os
import argparse
from huggingface_hub import snapshot_download

CHECKPOINTS = {
    "monot5_10k": {
        "hf_id":     "castorini/monot5-base-msmarco-10k",
        "local_dir": "checkpoints/monot5-base-msmarco-10k",
        "note":      "Best starting point for fine-tuning on medical data",
    },
    "monot5_100k": {
        "hf_id":     "castorini/monot5-base-msmarco",
        "local_dir": "checkpoints/monot5-base-msmarco-100k",
        "note":      "Strongest on MS MARCO — use for thesis MRR@10 evaluation",
    },
    "duot5": {
        "hf_id":     "castorini/duot5-base-msmarco",
        "local_dir": "checkpoints/duot5-base-msmarco",
        "note":      "Pairwise reranker — typically cascaded after monoT5",
    },
    "lit5_distill": {
        "hf_id":     "castorini/LiT5-Distill-base",
        "local_dir": "checkpoints/LiT5-Distill-base",
        "note":      "Listwise reranker via distillation (nDCG@10: 71.7 on TREC DL)",
    },
}


def download(key: str):
    cfg = CHECKPOINTS[key]
    os.makedirs(cfg["local_dir"], exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Model  : {key}")
    print(f"  HF ID  : {cfg['hf_id']}")
    print(f"  Note   : {cfg['note']}")
    print(f"  Dest   : {cfg['local_dir']}")
    print(f"{'='*60}")
    snapshot_download(
        repo_id=cfg["hf_id"],
        local_dir=cfg["local_dir"],
        ignore_patterns=["*.msgpack", "*.h5"],  # skip TF/Flax weights — saves ~500MB each
    )
    print(f"  ✓ {key} complete\n")


def print_summary(targets):
    print("\n--- Download Summary ---")
    for key in targets:
        cfg = CHECKPOINTS[key]
        size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, files in os.walk(cfg["local_dir"])
            for f in files
        ) / 1e9
        print(f"  {cfg['local_dir']:<45} {size:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download reranker checkpoints")
    parser.add_argument(
        "--model",
        nargs="+",
        choices=list(CHECKPOINTS.keys()) + ["all"],
        default=["all"],
        help="Which checkpoints to download. Default: all",
    )
    args = parser.parse_args()

    targets = list(CHECKPOINTS.keys()) if "all" in args.model else args.model

    print(f"\nDownloading {len(targets)} checkpoint(s): {', '.join(targets)}")

    failed = []
    for key in targets:
        try:
            download(key)
        except Exception as e:
            print(f"  ✗ FAILED: {key} — {e}")
            failed.append(key)

    print_summary([t for t in targets if t not in failed])

    if failed:
        print(f"\n⚠ Failed: {failed}")
        print("  Tip: set HF_TOKEN env var to avoid rate limits:")
        print("       export HF_TOKEN=your_token_here")
    else:
        print("\n✓ All downloads complete.")