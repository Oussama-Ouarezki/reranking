"""Bulk evaluation endpoint via WebSocket.

Client connects to ws://.../api/eval/run, sends a single JSON config message,
then receives a stream of progress events ending with {type: "done", results}.

Each model's results are saved to ``application/cache/runs/<model>/<ts>.json``
so the dashboard can list and replay them. The legacy ``/api/eval/cache``
still returns the most recent run for backward compat.
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
from ..evaluation import qa_metrics
from ..generation import rag

router = APIRouter()

DEFAULT_K_VALUES = [1, 5, 10, 20]
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
    """Return summaries of every saved run, newest first."""
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
                "has_qa": d.get("qa_aggregate") is not None,
            })
    out.sort(key=lambda r: r.get("ended_at") or 0, reverse=True)
    return out


def _load_run(run_id: str) -> dict | None:
    """Walk the runs tree for a matching run_id."""
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
    """Return the last cached eval run if any."""
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
    k_values: list[int],
    generate: bool,
    bm25,
    corpus,
    qrels: dict,
):
    """Returns metrics, full_ranked, answers_by_k, qa_scores_by_k, qtype."""
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

    answers_by_k: dict[int, str] = {}
    qa_scores_by_k: dict[int, float] = {}
    if generate:
        for k in k_values:
            top_hits = hits[:k]
            docs = []
            for h in top_hits:
                d = corpus.get(h["docid"]) or {}
                docs.append({
                    "rank": h["rank"],
                    "docid": h["docid"],
                    "title": d.get("title", ""),
                    "text": d.get("text", ""),
                })
            try:
                ans = rag.generate_answer(qtext, docs, qtype or "summary")
            except Exception as exc:
                ans = f"[gen error: {exc}]"
            answers_by_k[k] = ans
            if qtype and not ans.startswith("[gen error"):
                try:
                    s = qa_metrics.score_answer(qtype, ans, query)
                except Exception:
                    s = None
                if s is not None:
                    qa_scores_by_k[k] = float(s)

    return metrics, full_ranked, answers_by_k, qa_scores_by_k, qtype


def _aggregate_qa_by_k(
    rows: list[tuple[str, dict[int, float]]],
    k_values: list[int],
) -> dict | None:
    """Aggregate per-query (qtype, {k: score}) rows into per-(type,k) means."""
    if not rows:
        return None
    sums: dict[str, dict[int, float]] = {}
    counts: dict[str, dict[int, int]] = {}
    type_counts: dict[str, int] = {}
    for qtype, by_k in rows:
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
        for k, v in by_k.items():
            sums.setdefault(qtype, {}).setdefault(k, 0.0)
            counts.setdefault(qtype, {}).setdefault(k, 0)
            sums[qtype][k] += v
            counts[qtype][k] += 1
    by_type_k: dict[str, dict[int, float]] = {}
    for qtype, sk in sums.items():
        by_type_k[qtype] = {}
        for k, total in sk.items():
            n = counts[qtype][k] or 1
            by_type_k[qtype][k] = round(total / n, 4)
    return {
        "by_type_k": by_type_k,
        "n_per_type": type_counts,
        "k_values": k_values,
    }


# ---------- WebSocket ---------------------------------------------------------


@router.websocket("/eval/run")
async def eval_run(ws: WebSocket):
    await ws.accept()
    try:
        cfg_msg = await ws.receive_json()
    except WebSocketDisconnect:
        return

    models: list[str] = cfg_msg.get("models", ["bm25"])
    top_k: int = int(cfg_msg.get("top_k", 10))  # legacy ranking cutoff (unused for gen now)
    generate: bool = bool(cfg_msg.get("generate", False))
    k_values: list[int] = cfg_msg.get("k_values") or DEFAULT_K_VALUES

    bm25 = deps.get_bm25()
    corpus = deps.get_corpus()
    qrels = deps.get_qrels()
    queries = deps.get_queries()
    total_queries = len(queries)

    started_at = time.time()
    legacy_per_model: dict = {}
    legacy_per_query: dict = {q["_id"]: {} for q in queries}
    legacy_answers: dict = {}
    saved_run_ids: list[str] = []

    try:
        for model_name in models:
            if model_name not in {"bm25", "monot5", "duot5", "lit5", "mono_duo"}:
                await ws.send_json({"type": "error", "message": f"unknown model {model_name}"})
                continue

            full_run: list[tuple[str, str, float]] = []
            qa_rows: list[tuple[str, dict[int, float]]] = []
            per_query_model: dict[str, dict] = {}
            answers_model: dict[str, dict[int, str]] = {}
            t0 = time.time()

            for i, q in enumerate(queries, start=1):
                await asyncio.sleep(0)
                qid = q["_id"]
                try:
                    metrics, full_ranked, answers_by_k, qa_scores_by_k, qtype = await asyncio.to_thread(
                        _run_one_query,
                        model_name, q, k_values, generate, bm25, corpus, qrels,
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
                    full_run.append((qid, docid, score))

                entry: dict = {}
                if metrics is not None:
                    entry["metrics"] = metrics
                if qa_scores_by_k:
                    entry["qa_scores_by_k"] = qa_scores_by_k
                if qtype is not None:
                    entry["qtype"] = qtype
                if entry:
                    per_query_model[qid] = entry
                    legacy_per_query[qid][model_name] = entry

                if answers_by_k:
                    answers_model[qid] = answers_by_k
                    legacy_answers.setdefault(qid, {})[model_name] = answers_by_k.get(top_k) or next(
                        iter(answers_by_k.values()), None
                    )
                if qa_scores_by_k and qtype:
                    qa_rows.append((qtype, qa_scores_by_k))

                if i % 5 == 0 or i == total_queries:
                    await ws.send_json({
                        "type": "progress",
                        "model": model_name,
                        "current": i,
                        "total": total_queries,
                    })

            agg = aggregate_metrics(full_run, qrels)
            qa_aggregate = _aggregate_qa_by_k(qa_rows, k_values) if qa_rows else None
            elapsed = round(time.time() - t0, 1)
            ended_at = time.time()

            run_payload = {
                "model": model_name,
                "started_at": t0,
                "ended_at": ended_at,
                "elapsed_s": elapsed,
                "config": {
                    "top_k": top_k,
                    "generate": generate,
                    "k_values": k_values,
                },
                "aggregate": agg,
                "qa_aggregate": qa_aggregate,
                "per_query": per_query_model,
                "answers": answers_model if generate else None,
            }
            run_id = _save_run(run_payload)
            saved_run_ids.append(run_id)

            legacy_per_model[model_name] = {
                "aggregate": agg,
                "qa_aggregate": qa_aggregate,
                "elapsed_s": elapsed,
                "run_id": run_id,
            }

            await ws.send_json({
                "type": "model_done",
                "model": model_name,
                "aggregate": agg,
                "qa_aggregate": qa_aggregate,
                "elapsed_s": elapsed,
                "run_id": run_id,
            })

        results = {
            "started_at": started_at,
            "ended_at": time.time(),
            "config": {
                "models": models,
                "top_k": top_k,
                "generate": generate,
                "k_values": k_values,
            },
            "per_model": legacy_per_model,
            "per_query": legacy_per_query,
            "answers": legacy_answers if generate else None,
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
