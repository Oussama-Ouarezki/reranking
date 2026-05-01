"""Retrieval-only bulk evaluation via WebSocket.

Generation has moved to ``routers/generation.py``. This module is purely about
retrieval: given a model (BM25 or a reranker), score every query against the
qrels and save the top-``SAVE_TOPN`` passages per query for downstream
generation runs.

Each model's results are saved to ``application/cache/runs/<model>/<ts>.json``;
``GET /api/eval/runs`` lists summaries and ``GET /api/eval/runs/{run_id}``
returns the full payload.
"""

import asyncio
import json
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, WebSocket, WebSocketDisconnect

from .. import config, deps
from ..rerankers import registry as rerank_registry
from ..evaluation.ranking import per_query_metrics, aggregate_metrics

router = APIRouter()

# Top-N passages saved per query (caps generation k).
SAVE_TOPN = 20
RUNS_DIR = config.CACHE_DIR / "runs"
SAMPLE_SEED = 42


def _sample_queries(queries: list[dict], n: int | None) -> list[dict]:
    """Deterministic sample of N queries (seed=42). n=None or n>=len → no sampling."""
    if n is None or n <= 0 or n >= len(queries):
        return queries
    rng = random.Random(SAMPLE_SEED)
    idxs = sorted(rng.sample(range(len(queries)), n))
    return [queries[i] for i in idxs]


# ---------- run-file helpers --------------------------------------------------


def _runs_dir(model: str) -> Path:
    p = RUNS_DIR / model
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_run(payload: dict) -> str:
    """Write one model's run to disk. Returns the run_id used."""
    model = payload["model"]
    ts = datetime.fromtimestamp(payload["ended_at"], tz=timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_id = f"{model}_{ts}"
    payload = {**payload, "run_id": run_id}
    out = _runs_dir(model) / f"{ts}.json"
    with open(out, "w") as f:
        json.dump(payload, f)
    return run_id


def _list_runs() -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    out: list[dict] = []
    for model_dir in sorted(RUNS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        for f in sorted(model_dir.glob("*.json"), reverse=True):
            try:
                with open(f) as fh:
                    d = json.load(fh)
            except Exception:
                continue
            out.append({
                "run_id": d.get("run_id") or f.stem,
                "model": d.get("model", model_dir.name),
                "started_at": d.get("started_at"),
                "ended_at": d.get("ended_at"),
                "config": d.get("config"),
                "elapsed_s": d.get("elapsed_s"),
                "n_queries": len(d.get("per_query", {})),
                "comment": d.get("comment", ""),
            })
    out.sort(key=lambda r: r.get("ended_at") or 0, reverse=True)
    return out


def _load_run(run_id: str) -> dict | None:
    if not RUNS_DIR.exists():
        return None
    for model_dir in RUNS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        for f in model_dir.glob("*.json"):
            try:
                with open(f) as fh:
                    d = json.load(fh)
            except Exception:
                continue
            if d.get("run_id") == run_id or f.stem == run_id:
                return d
    return None


def _cache_latest(payload: dict) -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.EVAL_CACHE, "w") as f:
        json.dump(payload, f)


# ---------- HTTP endpoints ----------------------------------------------------


@router.get("/eval/cache")
def get_cache():
    if not config.EVAL_CACHE.exists():
        return {"cached": False}
    with open(config.EVAL_CACHE) as f:
        return {"cached": True, "data": json.load(f)}


@router.get("/eval/runs")
def list_runs():
    return {"runs": _list_runs()}


@router.get("/eval/runs/{run_id}")
def get_run(run_id: str):
    d = _load_run(run_id)
    if d is None:
        raise HTTPException(status_code=404, detail="run not found")
    return d


@router.patch("/eval/runs/{run_id}")
def update_run(run_id: str, body: dict = Body(...)):
    """Update mutable fields on a retrieval run. Currently only ``comment``."""
    if "comment" not in body:
        raise HTTPException(status_code=400, detail="only 'comment' is patchable")
    comment = str(body["comment"])
    if not RUNS_DIR.exists():
        raise HTTPException(status_code=404, detail="run not found")
    for model_dir in RUNS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        for f in model_dir.glob("*.json"):
            try:
                with open(f) as fh:
                    d = json.load(fh)
            except Exception:
                continue
            if d.get("run_id") == run_id or f.stem == run_id:
                d["comment"] = comment
                with open(f, "w") as fh:
                    json.dump(d, fh)
                return {"ok": True, "comment": comment}
    raise HTTPException(status_code=404, detail="run not found")


METRIC_KEYS = ("ndcg_at", "mrr_at", "p_at", "r_at", "map_at")
DEFAULT_DIFF_KS = (1, 5, 10, 20)


def _per_query_qtype_means(run: dict, qids: set[str]) -> dict[str, dict[str, dict[int, float]]]:
    """Aggregate per-query metrics by qtype over the given qid subset.

    Returns: {qtype: {metric: {k: mean_value}}}.
    """
    bucket: dict[str, dict[str, dict[int, list[float]]]] = {}
    for qid, entry in (run.get("per_query") or {}).items():
        if qid not in qids:
            continue
        m = entry.get("metrics")
        qtype = entry.get("qtype")
        if not m or not qtype:
            continue
        per_metric = bucket.setdefault(qtype, {k: {} for k in METRIC_KEYS})
        for mk in METRIC_KEYS:
            for k, v in (m.get(mk) or {}).items():
                # JSON keys are strings; coerce.
                k_int = int(k)
                per_metric[mk].setdefault(k_int, []).append(float(v))

    out: dict[str, dict[str, dict[int, float]]] = {}
    for qtype, mmap in bucket.items():
        out[qtype] = {}
        for mk, kmap in mmap.items():
            out[qtype][mk] = {k: round(sum(vs) / len(vs), 4) for k, vs in kmap.items() if vs}
    return out


def _global_means(run: dict, qids: set[str]) -> dict[str, dict[int, float]]:
    bucket: dict[str, dict[int, list[float]]] = {mk: {} for mk in METRIC_KEYS}
    for qid, entry in (run.get("per_query") or {}).items():
        if qid not in qids:
            continue
        m = entry.get("metrics")
        if not m:
            continue
        for mk in METRIC_KEYS:
            for k, v in (m.get(mk) or {}).items():
                k_int = int(k)
                bucket[mk].setdefault(k_int, []).append(float(v))
    return {
        mk: {k: round(sum(vs) / len(vs), 4) for k, vs in kmap.items() if vs}
        for mk, kmap in bucket.items()
    }


def _diff(a: dict, b: dict) -> dict:
    """Cell-wise (a - b) over the same {metric: {k: value}} shape."""
    out: dict[str, dict[int, float]] = {}
    for mk in a:
        kmap_a = a[mk]
        kmap_b = b.get(mk, {})
        out[mk] = {
            k: round(kmap_a[k] - kmap_b.get(k, 0.0), 4)
            for k in kmap_a
            if k in kmap_b
        }
    return out


@router.get("/eval/runs/{run_id}/diff")
def diff_run(run_id: str, baseline: str):
    """Return per-qtype + global metric deltas of run_id vs baseline.

    Both runs must be loadable. Comparison is made over the *intersection* of
    qids that appear in both runs (so comparisons are fair even when one run
    used a different sample size).
    """
    a = _load_run(run_id)
    if a is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    b = _load_run(baseline)
    if b is None:
        raise HTTPException(status_code=404, detail=f"baseline not found: {baseline}")

    qids = set((a.get("per_query") or {}).keys()) & set((b.get("per_query") or {}).keys())
    if not qids:
        raise HTTPException(status_code=400, detail="no overlapping qids between runs")

    a_global = _global_means(a, qids)
    b_global = _global_means(b, qids)
    a_qtype = _per_query_qtype_means(a, qids)
    b_qtype = _per_query_qtype_means(b, qids)

    by_qtype = {
        qtype: {
            "a": a_qtype.get(qtype, {}),
            "b": b_qtype.get(qtype, {}),
            "delta": _diff(a_qtype.get(qtype, {}), b_qtype.get(qtype, {})),
        }
        for qtype in set(a_qtype) | set(b_qtype)
    }

    return {
        "run_id": run_id,
        "baseline": baseline,
        "n_overlapping": len(qids),
        "global": {"a": a_global, "b": b_global, "delta": _diff(a_global, b_global)},
        "by_qtype": by_qtype,
    }


@router.delete("/eval/runs/{run_id}")
def delete_run(run_id: str):
    if not RUNS_DIR.exists():
        raise HTTPException(status_code=404, detail="run not found")
    for model_dir in RUNS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        for f in model_dir.glob("*.json"):
            try:
                with open(f) as fh:
                    d = json.load(fh)
            except Exception:
                continue
            if d.get("run_id") == run_id or f.stem == run_id:
                f.unlink()
                return {"ok": True}
    raise HTTPException(status_code=404, detail="run not found")


# ---------- per-query worker --------------------------------------------------


def _run_one_query(
    model_name: str,
    query: dict,
    bm25,
    corpus,
    qrels: dict,
):
    """Returns metrics, full_ranked, top_docids, qtype."""
    qid = query["_id"]
    qtext = query["text"]
    qtype = query.get("type")

    hits = bm25.search(qtext, k=config.BM25_RETRIEVE_K)

    if model_name != "bm25" and hits:
        reranker = rerank_registry.get(model_name)
        candidates = []
        for h in hits:
            text = corpus.get_text(h["docid"])
            if text:
                candidates.append((h["docid"], text))
        ranked = reranker.rerank(qtext, candidates)
        score_map = dict(ranked)
        order = [docid for docid, _ in ranked]
        hits = [
            {"docid": docid, "score": score_map[docid], "rank": i + 1}
            for i, docid in enumerate(order)
        ]

    full_ranked = [(h["docid"], float(h["score"])) for h in hits]
    metrics = per_query_metrics(qid, full_ranked, qrels) if qid in qrels else None
    top_docids = [str(h["docid"]) for h in hits[:SAVE_TOPN]]

    return metrics, full_ranked, top_docids, qtype


# ---------- WebSocket ---------------------------------------------------------


@router.websocket("/eval/run")
async def eval_run(ws: WebSocket):
    await ws.accept()
    try:
        cfg_msg = await ws.receive_json()
    except WebSocketDisconnect:
        return

    models: list[str] = cfg_msg.get("models", ["bm25"])
    n_questions: int | None = cfg_msg.get("n_questions")
    comment: str = str(cfg_msg.get("comment", "") or "")

    bm25 = deps.get_bm25()
    corpus = deps.get_corpus()
    qrels = deps.get_qrels()
    all_queries = deps.get_queries()
    queries = _sample_queries(all_queries, n_questions)
    sampled_qids = [q["_id"] for q in queries]
    total_queries = len(queries)

    started_at = time.time()
    summary_per_model: dict = {}
    saved_run_ids: list[str] = []

    try:
        for model_name in models:
            if model_name not in rerank_registry.eval_models():
                await ws.send_json({"type": "error", "message": f"unknown model {model_name}"})
                continue

            full_run: list[tuple[str, str, float]] = []
            per_query_model: dict[str, dict] = {}
            t0 = time.time()

            for i, q in enumerate(queries, start=1):
                await asyncio.sleep(0)
                qid = q["_id"]
                try:
                    metrics, full_ranked, top_docids, qtype = await asyncio.to_thread(
                        _run_one_query,
                        model_name, q, bm25, corpus, qrels,
                    )
                except Exception as exc:
                    await ws.send_json({
                        "type": "error",
                        "model": model_name,
                        "qid": qid,
                        "message": str(exc),
                        "trace": traceback.format_exc(),
                    })
                    continue

                for docid, score in full_ranked:
                    full_run.append((qid, str(docid), score))

                entry: dict = {"top_docids": top_docids}
                if metrics is not None:
                    entry["metrics"] = metrics
                if qtype is not None:
                    entry["qtype"] = qtype
                per_query_model[qid] = entry

                if i % 5 == 0 or i == total_queries:
                    await ws.send_json({
                        "type": "progress",
                        "model": model_name,
                        "current": i,
                        "total": total_queries,
                    })

            agg = aggregate_metrics(full_run, qrels)
            elapsed = round(time.time() - t0, 1)
            ended_at = time.time()

            run_payload = {
                "model": model_name,
                "started_at": t0,
                "ended_at": ended_at,
                "elapsed_s": elapsed,
                "config": {
                    "save_topn": SAVE_TOPN,
                    "bm25_retrieve_k": config.BM25_RETRIEVE_K,
                    "n_questions": total_queries,
                    "sampled_qids": sampled_qids if n_questions else None,
                    "seed": SAMPLE_SEED if n_questions else None,
                },
                "comment": comment,
                "aggregate": agg,
                "per_query": per_query_model,
            }
            run_id = _save_run(run_payload)
            saved_run_ids.append(run_id)

            summary_per_model[model_name] = {
                "aggregate": agg,
                "elapsed_s": elapsed,
                "run_id": run_id,
            }

            await ws.send_json({
                "type": "model_done",
                "model": model_name,
                "aggregate": agg,
                "elapsed_s": elapsed,
                "run_id": run_id,
            })

        results = {
            "started_at": started_at,
            "ended_at": time.time(),
            "config": {
                "models": models,
                "save_topn": SAVE_TOPN,
                "bm25_retrieve_k": config.BM25_RETRIEVE_K,
                "n_questions": total_queries,
            },
            "per_model": summary_per_model,
            "saved_run_ids": saved_run_ids,
        }
        _cache_latest(results)
        await ws.send_json({"type": "done", "results": results})
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await ws.send_json({
            "type": "error",
            "message": str(exc),
            "trace": traceback.format_exc(),
        })
    finally:
        try:
            await ws.close()
        except Exception:
            pass
