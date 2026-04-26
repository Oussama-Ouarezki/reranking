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
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from .. import config, deps
from ..rerankers import registry as rerank_registry
from ..evaluation.ranking import per_query_metrics, aggregate_metrics

router = APIRouter()

# Top-N passages saved per query (caps generation k).
SAVE_TOPN = 20
RUNS_DIR = config.CACHE_DIR / "runs"


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

    bm25 = deps.get_bm25()
    corpus = deps.get_corpus()
    qrels = deps.get_qrels()
    queries = deps.get_queries()
    total_queries = len(queries)

    started_at = time.time()
    summary_per_model: dict = {}
    saved_run_ids: list[str] = []

    try:
        for model_name in models:
            if model_name not in {"bm25", "monot5", "duot5", "lit5", "mono_duo"}:
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
                "config": {"save_topn": SAVE_TOPN},
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
            "config": {"models": models, "save_topn": SAVE_TOPN},
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
