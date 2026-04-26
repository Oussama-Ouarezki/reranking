"""Generation runs.

A generation run takes a *retrieval* run (saved by ``routers/eval.py``) plus
a single ``k`` value, and for every query in the retrieval run:

1. Fetches the top-``k`` saved doc-ids from the retrieval run.
2. Hydrates them with full title + text from the corpus (no truncation).
3. Calls the LLM to produce a typed answer (yesno/factoid/list/summary).
4. Scores the answer against the gold answers in ``queries_full.jsonl`` via
   ``evaluation.qa_metrics``.
5. Recomputes retrieval metrics @ ``k`` from the saved doc-ids + qrels (cheap).
6. Aggregates per-qtype QA scores and per-qtype Spearman correlation matrices
   between (retrieval metrics @ k) and the QA score.

If the user picks several ``k_values`` in one Run click, each k produces its
own generation-run file under ``application/cache/gen_runs/<retrieval_run_id>/k<k>_<ts>.json``.
"""

import asyncio
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from .. import config, deps
from ..evaluation.correlation import spearman_matrix
from ..evaluation.ranking import per_query_metrics
from ..evaluation import qa_metrics
from ..generation import rag
from .eval import _load_run as _load_retrieval_run

router = APIRouter()

GEN_RUNS_DIR = config.CACHE_DIR / "gen_runs"
QTYPES = ("factoid", "yesno", "list", "summary")
QTYPE_METRIC_LABEL = {
    "factoid": "MRR",
    "yesno": "Acc",
    "list": "F1",
    "summary": "Judge",
}
# Bump Ollama context to fit ~20 PubMed abstracts (~6-10k tokens) plus headroom.
NUM_CTX = 32768


# ---------- run-file helpers --------------------------------------------------


def _retrieval_dir(retrieval_run_id: str) -> Path:
    p = GEN_RUNS_DIR / retrieval_run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_gen_run(payload: dict) -> str:
    retrieval_run_id = payload["retrieval_run_id"]
    k = int(payload["k"])
    ts = datetime.fromtimestamp(payload["ended_at"], tz=timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_id = f"gen_{retrieval_run_id}_k{k}_{ts}"
    payload = {**payload, "run_id": run_id}
    out = _retrieval_dir(retrieval_run_id) / f"k{k}_{ts}.json"
    with open(out, "w") as f:
        json.dump(payload, f)
    return run_id


def _list_gen_runs(retrieval_run_id: str | None = None) -> list[dict]:
    if not GEN_RUNS_DIR.exists():
        return []
    out: list[dict] = []
    targets = (
        [GEN_RUNS_DIR / retrieval_run_id]
        if retrieval_run_id
        else [d for d in GEN_RUNS_DIR.iterdir() if d.is_dir()]
    )
    for retrieval_dir in targets:
        if not retrieval_dir.exists() or not retrieval_dir.is_dir():
            continue
        for f in sorted(retrieval_dir.glob("*.json"), reverse=True):
            try:
                with open(f) as fh:
                    d = json.load(fh)
            except Exception:
                continue
            out.append({
                "run_id": d.get("run_id") or f.stem,
                "retrieval_run_id": d.get("retrieval_run_id", retrieval_dir.name),
                "k": d.get("k"),
                "started_at": d.get("started_at"),
                "ended_at": d.get("ended_at"),
                "elapsed_s": d.get("elapsed_s"),
                "n_queries": len(d.get("per_query", {})),
                "by_qtype": (d.get("aggregate") or {}).get("by_qtype"),
                "n_per_qtype": (d.get("aggregate") or {}).get("n_per_qtype"),
            })
    out.sort(key=lambda r: r.get("ended_at") or 0, reverse=True)
    return out


def _load_gen_run(run_id: str) -> dict | None:
    if not GEN_RUNS_DIR.exists():
        return None
    for retrieval_dir in GEN_RUNS_DIR.iterdir():
        if not retrieval_dir.is_dir():
            continue
        for f in retrieval_dir.glob("*.json"):
            try:
                with open(f) as fh:
                    d = json.load(fh)
            except Exception:
                continue
            if d.get("run_id") == run_id or f.stem == run_id:
                return d
    return None


# ---------- HTTP endpoints ----------------------------------------------------


@router.get("/generation/runs")
def list_gen_runs(retrieval_run_id: str | None = None):
    return {"runs": _list_gen_runs(retrieval_run_id)}


@router.get("/generation/runs/{run_id}")
def get_gen_run(run_id: str):
    d = _load_gen_run(run_id)
    if d is None:
        raise HTTPException(status_code=404, detail="generation run not found")
    return d


@router.delete("/generation/runs/{run_id}")
def delete_gen_run(run_id: str):
    if not GEN_RUNS_DIR.exists():
        raise HTTPException(status_code=404, detail="generation run not found")
    for retrieval_dir in GEN_RUNS_DIR.iterdir():
        if not retrieval_dir.is_dir():
            continue
        for f in retrieval_dir.glob("*.json"):
            try:
                with open(f) as fh:
                    d = json.load(fh)
            except Exception:
                continue
            if d.get("run_id") == run_id or f.stem == run_id:
                f.unlink()
                return {"ok": True}
    raise HTTPException(status_code=404, detail="generation run not found")


# ---------- per-query worker --------------------------------------------------


def _generate_one(
    qid: str,
    query: dict,
    top_docids: list[str],
    k: int,
    corpus,
    qrels: dict,
):
    """Returns (per-query payload dict) or None if qtype missing."""
    qtype = query.get("type")
    qtext = query.get("text", "")
    if not qtype or not top_docids:
        return None

    # Hydrate top-k docs with full title + text. No truncation.
    docs = []
    chosen = top_docids[:k]
    for rank, docid in enumerate(chosen, start=1):
        d = corpus.get(docid) or {}
        docs.append({
            "rank": rank,
            "docid": docid,
            "title": d.get("title", ""),
            "text": d.get("text", ""),
        })

    try:
        answer = rag.generate_answer(qtext, docs, qtype, num_ctx=NUM_CTX)
    except Exception as exc:
        return {
            "qid": qid,
            "qtype": qtype,
            "question": qtext,
            "answer": f"[gen error: {exc}]",
            "qa_score": None,
            "retrieval_metrics": None,
            "top_docids": chosen,
        }

    qa_score = None
    if not answer.startswith("[gen error"):
        try:
            qa_score = qa_metrics.score_answer(qtype, answer, query)
        except Exception:
            qa_score = None

    # Recompute retrieval metrics @ k from saved docids + qrels.
    # Synthetic descending scores so the saved order is preserved.
    n = len(top_docids)
    ranked = [(d, float(n - i)) for i, d in enumerate(top_docids)]
    rmetrics = (
        per_query_metrics(qid, ranked, qrels, ks=[k]) if qid in qrels else None
    )
    flat_rmetrics: dict[str, float] | None = None
    if rmetrics is not None:
        flat_rmetrics = {
            "ndcg": rmetrics["ndcg_at"][k],
            "p": rmetrics["p_at"][k],
            "r": rmetrics["r_at"][k],
            "mrr": rmetrics["mrr_at"][k],
            "map": rmetrics["map_at"][k],
        }

    return {
        "qid": qid,
        "qtype": qtype,
        "question": qtext,
        "answer": answer,
        "qa_score": float(qa_score) if qa_score is not None else None,
        "retrieval_metrics": flat_rmetrics,
        "top_docids": chosen,
    }


def _aggregate(per_query: dict[str, dict]) -> dict:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for entry in per_query.values():
        if entry.get("qa_score") is None:
            continue
        t = entry["qtype"]
        sums[t] = sums.get(t, 0.0) + float(entry["qa_score"])
        counts[t] = counts.get(t, 0) + 1
    by_qtype = {t: round(sums[t] / counts[t], 4) for t in sums if counts[t] > 0}
    n_per_qtype = dict(counts)
    return {"by_qtype": by_qtype, "n_per_qtype": n_per_qtype}


def _correlations(per_query: dict[str, dict]) -> dict:
    """Per-qtype Spearman matrix over (ndcg, p, r, mrr, map, qa)."""
    var_names = ["nDCG", "P", "R", "MRR", "MAP", "qa"]
    out: dict[str, dict] = {}
    for qtype in QTYPES:
        rows: list[list[float]] = []
        for entry in per_query.values():
            if entry["qtype"] != qtype:
                continue
            rm = entry.get("retrieval_metrics")
            qa = entry.get("qa_score")
            if rm is None or qa is None:
                continue
            rows.append([rm["ndcg"], rm["p"], rm["r"], rm["mrr"], rm["map"], float(qa)])
        if rows:
            out[qtype] = spearman_matrix(rows, var_names)
    return out


# ---------- WebSocket --------------------------------------------------------


@router.websocket("/generation/run")
async def generation_run(ws: WebSocket):
    await ws.accept()
    try:
        cfg_msg = await ws.receive_json()
    except WebSocketDisconnect:
        return

    retrieval_run_id: str = cfg_msg.get("retrieval_run_id")
    k_values: list[int] = cfg_msg.get("k_values") or [10]
    qtype_filter: list[str] | None = cfg_msg.get("qtypes") or None  # None == all

    if not retrieval_run_id:
        await ws.send_json({"type": "error", "message": "retrieval_run_id required"})
        await ws.close()
        return

    retrieval = _load_retrieval_run(retrieval_run_id)
    if retrieval is None:
        await ws.send_json({"type": "error", "message": f"retrieval run not found: {retrieval_run_id}"})
        await ws.close()
        return

    queries_by_id = {q["_id"]: q for q in deps.get_queries()}
    qrels = deps.get_qrels()
    corpus = deps.get_corpus()

    retrieval_per_query = retrieval.get("per_query", {})
    eligible_qids: list[str] = []
    for qid, entry in retrieval_per_query.items():
        if qid not in queries_by_id:
            continue
        q = queries_by_id[qid]
        if not q.get("type"):
            continue
        if qtype_filter and q["type"] not in qtype_filter:
            continue
        if not entry.get("top_docids"):
            continue
        eligible_qids.append(qid)

    total = len(eligible_qids)
    if total == 0:
        await ws.send_json({"type": "error", "message": "no eligible queries (need top_docids + qtype)"})
        await ws.close()
        return

    started_at = time.time()
    saved_run_ids: list[str] = []

    try:
        for k in k_values:
            t0 = time.time()
            per_query: dict[str, dict] = {}

            for i, qid in enumerate(eligible_qids, start=1):
                await asyncio.sleep(0)
                entry = retrieval_per_query[qid]
                top_docids = list(entry.get("top_docids") or [])
                q = queries_by_id[qid]

                try:
                    payload = await asyncio.to_thread(
                        _generate_one,
                        qid, q, top_docids, k, corpus, qrels,
                    )
                except Exception as exc:
                    await ws.send_json({
                        "type": "error",
                        "k": k,
                        "qid": qid,
                        "message": str(exc),
                        "trace": traceback.format_exc(),
                    })
                    continue

                if payload is not None:
                    per_query[qid] = payload

                if i % 5 == 0 or i == total:
                    await ws.send_json({
                        "type": "progress",
                        "k": k,
                        "current": i,
                        "total": total,
                    })

            agg = _aggregate(per_query)
            corr = _correlations(per_query)
            elapsed = round(time.time() - t0, 1)
            ended_at = time.time()

            run_payload = {
                "retrieval_run_id": retrieval_run_id,
                "retrieval_model": retrieval.get("model"),
                "k": k,
                "started_at": t0,
                "ended_at": ended_at,
                "elapsed_s": elapsed,
                "config": {"qtypes": qtype_filter, "num_ctx": NUM_CTX},
                "per_query": per_query,
                "aggregate": agg,
                "correlations": corr,
            }
            run_id = _save_gen_run(run_payload)
            saved_run_ids.append(run_id)

            await ws.send_json({
                "type": "k_done",
                "k": k,
                "elapsed_s": elapsed,
                "run_id": run_id,
                "aggregate": agg,
            })

        await ws.send_json({
            "type": "done",
            "started_at": started_at,
            "ended_at": time.time(),
            "saved_run_ids": saved_run_ids,
        })
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
