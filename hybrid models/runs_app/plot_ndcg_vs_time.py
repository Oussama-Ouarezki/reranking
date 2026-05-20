"""Plot nDCG@1, nDCG@5, nDCG@10 vs inference time — latest run per model.

Produces 4 separate PNG files (all saved to runs_app/):
  ndcg_vs_time_all.png   — one graph with all 3 cutoffs (marker shape = cutoff)
  ndcg_vs_time_1.png     — nDCG@1 only
  ndcg_vs_time_5.png     — nDCG@5 only
  ndcg_vs_time_10.png    — nDCG@10 only

Each model gets a unique color. BM25 is excluded.

Usage (from project root):
  python runs_app/plot_ndcg_vs_time.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

ROOT     = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "application/cache/runs"
OUT_DIR  = ROOT / "runs_app"

EXCLUDE = {"bm25"}

MODEL_MARKERS  = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', 'p',
                  '<', '>', 'H', '8', 'd']  # all filled — cycles if > 15 models
CUTOFF_MARKERS = {'1': 'o', '5': 's', '10': '^'}
CUTOFFS        = [("1", "nDCG@1"), ("5", "nDCG@5"), ("10", "nDCG@10")]


def load_latest_runs(runs_dir: Path) -> list[dict]:
    latest: dict[str, dict] = {}
    for model_dir in sorted(runs_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name in EXCLUDE:
            continue
        best = None
        for jf in model_dir.glob("*.json"):
            try:
                data = json.loads(jf.read_text())
            except Exception:
                continue
            if best is None or data.get("started_at", 0) > best.get("started_at", 0):
                best = data
                best.setdefault("model", model_dir.name)
        if best:
            latest[best["model"]] = best
    return sorted(latest.values(), key=lambda r: r["model"])


def annotate(ax, x, y, label, color):
    ax.annotate(label, (x, y), textcoords="offset points",
                xytext=(6, 4), fontsize=7, color=color)


def save_fig(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close(fig)


def plot_combined(runs, palette):
    """All 3 cutoffs on one graph. Marker shape = cutoff, color = model."""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("nDCG@1 / @5 / @10 vs Inference Time — all models",
                 fontsize=13)

    model_handles, model_labels = [], []

    for i, run in enumerate(runs):
        model   = run.get("model", "?")
        elapsed = run.get("elapsed_s")
        color   = palette[i]
        if elapsed is None:
            continue

        annotated = False
        for cutoff_k, _ in CUTOFFS:
            ndcg = run.get("aggregate", {}).get("ndcg_at", {}).get(cutoff_k)
            if ndcg is None:
                continue
            sc = ax.scatter(elapsed, ndcg,
                            color=color, marker=CUTOFF_MARKERS[cutoff_k],
                            s=110, zorder=5, linewidths=0.5, edgecolors="white")
            if not annotated:
                model_handles.append(sc)
                model_labels.append(model)
                annotate(ax, elapsed, ndcg, model, color)
                annotated = True

    # model legend (colors)
    model_leg = ax.legend(model_handles, model_labels,
                          loc="upper left", fontsize=7, title="Model",
                          ncol=2, framealpha=0.8)
    ax.add_artist(model_leg)

    # cutoff legend (shapes)
    cutoff_handles = [
        mlines.Line2D([], [], color="gray", marker=CUTOFF_MARKERS[k],
                      linestyle="None", markersize=9, label=lbl)
        for k, lbl in CUTOFFS
    ]
    ax.legend(handles=cutoff_handles, title="Cutoff (shape)",
              loc="lower right", fontsize=9)

    ax.set_xlabel("Inference time (s)", fontsize=11)
    ax.set_ylabel("nDCG", fontsize=11)
    fig.tight_layout()
    return fig


def plot_single(runs, palette, cutoff_k, cutoff_label):
    """One nDCG cutoff per figure. Color + marker shape both = model."""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(f"{cutoff_label} vs Inference Time — latest run per model",
                 fontsize=13)

    handles, labels = [], []

    for i, run in enumerate(runs):
        model   = run.get("model", "?")
        elapsed = run.get("elapsed_s")
        ndcg    = run.get("aggregate", {}).get("ndcg_at", {}).get(cutoff_k)
        if elapsed is None or ndcg is None:
            continue
        color  = palette[i]
        marker = MODEL_MARKERS[i % len(MODEL_MARKERS)]
        sc = ax.scatter(elapsed, ndcg, color=color, marker=marker,
                        s=120, zorder=5, linewidths=0.5, edgecolors="white")
        annotate(ax, elapsed, ndcg, model, color)
        handles.append(sc)
        labels.append(model)

    ax.legend(handles, labels, loc="lower right", fontsize=7,
              title="Model", ncol=2, framealpha=0.8)
    ax.set_xlabel("Inference time (s)", fontsize=11)
    ax.set_ylabel(cutoff_label, fontsize=11)
    fig.tight_layout()
    return fig


def main() -> None:
    runs    = load_latest_runs(RUNS_DIR)
    palette = sns.color_palette("tab20", len(runs))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    save_fig(plot_combined(runs, palette),   OUT_DIR / "ndcg_vs_time_all.png")
    for cutoff_k, cutoff_label in CUTOFFS:
        save_fig(plot_single(runs, palette, cutoff_k, cutoff_label),
                 OUT_DIR / f"ndcg_vs_time_{cutoff_k}.png")


if __name__ == "__main__":
    main()
