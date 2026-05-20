import { Fragment, useEffect, useRef, useState } from "react";

import {
  deleteGenRun,
  fetchGenRun,
  fetchGenRuns,
  fetchGenSummary,
  fetchRuns,
  openGenSocket,
  patchGenRunComment,
  patchGenRunStarred,
  patchRunComment,
  type GenerationConfig,
} from "../lib/api";
import { modelColor } from "../lib/modelColors";
import type {
  ExtraMetrics,
  GenerationEvent,
  GenerationPerQuery,
  GenerationRunDetail,
  GenerationRunSummary,
  GenerationSummary,
  GenerationSummaryCell,
  GenerationSummaryRow,
  ModelName,
  QuestionType,
  RunSummary,
  SpearmanMatrix,
} from "../lib/types";

const QTYPES: QuestionType[] = ["factoid", "yesno", "list", "summary"];
const QTYPE_METRIC_LABEL: Record<QuestionType, string> = {
  factoid: "MRR",
  yesno: "Acc",
  list: "F1",
  summary: "Judge",
};

const EXTRA_METRIC_LABEL: Record<string, string> = {
  rouge_l: "ROUGE-L",
  bert_score: "BERTScore",
  strict_acc: "Strict Acc",
  map: "MAP",
};
const MODEL_LABEL: Record<string, string> = {
  bm25: "BM25",
  monot5: "BM25 → monoT5",
  duot5: "BM25 → duoT5",
  duot5_rolling: "BM25 → duoT5 (rolling 20, stride 10)",
  mono_duo: "BM25 → monoT5 → duoT5",
  monot5_lit5: "BM25 → monoT5 (≥0.7) → LiT5",
  mono_uncertain_duo_lit5: "BM25 → monoT5 → duoT5 (pos 15-25) → LiT5",
  mono_dynamic_duo_lit5: "BM25 → monoT5 → duoT5 (dynamic) → LiT5",
  mono_gated_duo: "BM25 → monoT5 → duoT5 (gated τ=0.001)",
  mono_proximity_duo: "BM25 → monoT5 → duoT5 (proximity 0.001)",
  mono_proximity_duo_lit5: "BM25 → monoT5 → duoT5 (prox 0.001) → LiT5",
  lit5_duo: "BM25 → LiT5 → duoT5 (top-10)",
  mono_proximity_duo_0005: "BM25 → monoT5 → duoT5 (proximity 0.0005)",
  mono_proximity_duo_005_top30: "BM25 → monoT5 → duoT5 (prox 0.005, top-30)",
  mono_mau_duo_low_cost: "BM25 → monoT5 → duoT5 (MAU τ=0.0001, ~21% duo)",
  mono_mau_duo_pareto: "BM25 → monoT5 → duoT5 (MAU τ=0.001, Pareto knee)",
  mono_gated_lit5_top20: "BM25 → monoT5 → LiT5 top-20 (gated τ=0.001)",
  mono_gated_lit5_top40: "BM25 → monoT5 → LiT5 top-40 (gated τ=0.001)",
  mono_gated_lit5_top50: "BM25 → monoT5 → LiT5 top-50 (gated τ=0.001)",
  lit5: "BM25 → LiT5",
  bge_v2_m3: "BM25 → BGE-v2-m3",
  qwen3_reranker_4b: "BM25 → Qwen3-Reranker-4B",
  qwen3_reranker_0_6b: "BM25 → Qwen3-0.6B (pure)",
  rank_zephyr: "BM25 → RankZephyr (Q4)",
  mono_entropy_gated_duo: "BM25 → monoT5 → duoT5 (entropy H@20, τ=0.95)",
  lit5_finetuned: "BM25 → LiT5 (fine-tuned)",
  lit5_bioasq_lora: "BM25 → LiT5 (LoRA, epoch 2)",
  lit5_bioasq_lora_e1: "BM25 → LiT5 (LoRA q,v, epoch 1)",
  lit5_bioasq_lora_e3: "BM25 → LiT5 (LoRA q,v, epoch 3)",
  lit5_bioasq_lora_kaggle: "BM25 → LiT5 (LoRA q,k,v,o, Kaggle)",
  lit5_bioasq_lora_kaggle_e1: "BM25 → LiT5 (LoRA Kaggle, epoch 1)",
  lit5_bioasq_lora_kaggle_e2: "BM25 → LiT5 (LoRA Kaggle, epoch 2)",
  lit5_bioasq_lora_kaggle_e3: "BM25 → LiT5 (LoRA Kaggle, epoch 3)",
  lit5_bioasq_lora_kaggle_e4: "BM25 → LiT5 (LoRA Kaggle, epoch 4)",
  mono_entropy_h50_lit5: "BM25 → monoT5 → LiT5 (entropy H@50, τ=0.839)",
  mono_entropy_h50_duo: "BM25 → monoT5 → duoT5 (entropy H@50, τ=0.832)",
  qwen4b_linear_fusion: "BM25 → Qwen3-4B + BM25 (linear, α=0.825)",
  qwen4b_linear_fusion_dynamic: "BM25 → Qwen3-4B + BM25 (linear, dynamic α)",
  qwen4b_linear_fusion_dynamic_10: "BM25 → Qwen3-4B + BM25 (linear, dynamic α @10)",
  qwen4b_linear_fusion_dynamic_gated: "BM25 → Qwen3-4B + BM25 (linear, dyn α, gated H@20)",
  qwen06b_lf: "BM25 → Qwen3-0.6B + BM25 (LF, dyn α)",
  qwen06b_lf_999: "BM25 → Qwen3-0.6B + BM25 (LF, α=0.999)",
  qwen06b_lf_999_lit5: "BM25 → Qwen3-0.6B + BM25 (LF, α=0.999) → LiT5 top-20",
  qwen06b_lf_999_duot5_unc_lit5: "BM25 → Qwen3-0.6B + BM25 (LF, α=0.999) → duoT5(15-25) → LiT5 top-20",
  qwen06b_lf_lit5: "BM25 → Qwen3-0.6B + BM25 (LF, dyn α) → LiT5 top-20",
  qwen06b_lf_duot5_unc_lit5: "BM25 → Qwen3-0.6B + BM25 (LF) → duoT5(15-25) → LiT5 top-20",
  biobert: "BM25 → BioBERT",
  bm25_biobert: "BM25 → BioBERT (cascade)",
  deepseek: "DeepSeek (script)",
};
const labelFor = (m?: ModelName | string) =>
  (m && (MODEL_LABEL[m as ModelName] ?? m)) || "?";
const PRESET_KS = [1, 3, 5, 10, 20];
const SUMMARY_METRICS = [
  { key: "qa_overall", label: "QA score (overall)" },
  { key: "qa_factoid", label: "QA factoid (MRR)" },
  { key: "factoid_strict_acc", label: "QA factoid (Strict Acc)" },
  { key: "qa_yesno", label: "QA yesno (Acc)" },
  { key: "qa_list", label: "QA list (F1)" },
  { key: "list_map", label: "QA list (MAP)" },
  { key: "qa_summary", label: "QA summary (Judge)" },
  { key: "ndcg", label: "nDCG@k" },
  { key: "p", label: "P@k" },
  { key: "r", label: "R@k" },
  { key: "mrr", label: "MRR@k" },
  { key: "map", label: "MAP@k" },
] as const;
type SummaryMetricKey = (typeof SUMMARY_METRICS)[number]["key"];

function readSummaryCell(c: GenerationSummaryCell, k: SummaryMetricKey): number | null {
  switch (k) {
    case "qa_overall":
      return c.qa_overall;
    case "qa_factoid":
      return c.qa_by_qtype.factoid ?? null;
    case "qa_yesno":
      return c.qa_by_qtype.yesno ?? null;
    case "qa_list":
      return c.qa_by_qtype.list ?? null;
    case "qa_summary":
      return c.qa_by_qtype.summary ?? null;
    case "factoid_strict_acc":
      return c.factoid_strict_acc ?? null;
    case "list_map":
      return c.list_map ?? null;
    case "ndcg":
    case "p":
    case "r":
    case "mrr":
    case "map":
      return c.retrieval[k] ?? null;
  }
}

const fmtMetric = (v: number) => v.toFixed(4);
const fmtCorr = (v: number) => v.toFixed(2);

interface KProgress {
  current: number;
  total: number;
  done: boolean;
  elapsed_s?: number;
}

export default function GenerationPage() {
  const [retrievalRuns, setRetrievalRuns] = useState<RunSummary[]>([]);
  const [selectedRetrievalId, setSelectedRetrievalId] = useState<string | null>(null);
  // IDs checked for batch generation
  const [batchRetrievalIds, setBatchRetrievalIds] = useState<string[]>([]);
  const [genRuns, setGenRuns] = useState<GenerationRunSummary[]>([]);
  const [selectedGen, setSelectedGen] = useState<GenerationRunDetail | null>(null);
  const [showConfig, setShowConfig] = useState(false);

  const [ks, setKs] = useState<number[]>([1]);
  const [customK, setCustomK] = useState<string>("");
  const [qtypes, setQtypes] = useState<QuestionType[]>([...QTYPES]);
  const [comment, setComment] = useState<string>("");
  const [skipJudge, setSkipJudge] = useState(false);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<Record<number, KProgress>>({});
  const [currentBatchLabel, setCurrentBatchLabel] = useState<string>("");
  const [batchPos, setBatchPos] = useState<[number, number]>([0, 0]); // [current, total]
  const [errorLog, setErrorLog] = useState<string[]>([]);
  const [summary, setSummary] = useState<GenerationSummary | null>(null);
  const [summaryMetric, setSummaryMetric] = useState<SummaryMetricKey>("qa_overall");
  // Full gen-run details for the currently-selected retrieval, keyed by run_id.
  // Used by the side-by-side Spearman + confusion-matrix views (across-k).
  const [genDetailsByRunId, setGenDetailsByRunId] = useState<
    Record<string, GenerationRunDetail>
  >({});
  const wsRef = useRef<WebSocket | null>(null);
  // Refs hold the active batch state so WS callbacks always read fresh values
  // even when the component has re-rendered since the run started.
  const batchQueueRef = useRef<string[]>([]);
  const batchIndexRef = useRef<number>(0);
  const batchKsRef = useRef<number[]>([1]);
  const batchQtypesRef = useRef<QuestionType[]>([...QTYPES]);
  const batchCommentRef = useRef<string>("");
  const batchSkipJudgeRef = useRef<boolean>(false);
  const selectedRetrievalIdRef = useRef<string | null>(null);

  const refreshSummary = async () => {
    try {
      setSummary(await fetchGenSummary());
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchRuns()
      .then((r) => {
        setRetrievalRuns(r);
        if (!selectedRetrievalId && r.length > 0) {
          setSelectedRetrievalId(r[0].run_id);
        }
      })
      .catch((e) => console.error(e));
    refreshSummary().catch((e) => console.error(e));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const refreshGenRuns = async (retrievalId: string | null) => {
    if (!retrievalId) {
      setGenRuns([]);
      return;
    }
    try {
      const r = await fetchGenRuns(retrievalId);
      setGenRuns(r);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    refreshGenRuns(selectedRetrievalId).catch((e) => console.error(e));
    selectedRetrievalIdRef.current = selectedRetrievalId;
    setSelectedGen(null);
    // Drop any cached details from the previous retrieval so the side-by-side
    // views don't show stale matrices while the new ones load.
    setGenDetailsByRunId({});
  }, [selectedRetrievalId]);

  // Lazily fetch full details for every gen run shown in the list. Each detail
  // is small (one JSON file) and the user has a finite number of k per retrieval,
  // so we just fetch them all in parallel and cache by run_id.
  useEffect(() => {
    if (genRuns.length === 0) return;
    const missing = genRuns.filter((r) => !genDetailsByRunId[r.run_id]);
    if (missing.length === 0) return;
    let cancelled = false;
    Promise.all(
      missing.map((r) =>
        fetchGenRun(r.run_id)
          .then((d) => [r.run_id, d] as const)
          .catch((e) => {
            console.error("fetchGenRun", r.run_id, e);
            return null;
          }),
      ),
    ).then((pairs) => {
      if (cancelled) return;
      setGenDetailsByRunId((prev) => {
        const next = { ...prev };
        for (const p of pairs) {
          if (p) next[p[0]] = p[1];
        }
        return next;
      });
    });
    return () => {
      cancelled = true;
    };
  }, [genRuns, genDetailsByRunId]);

  // Reads entirely from refs so stale-closure issues can't affect batch sequencing.
  const openBatchItem = () => {
    const idx = batchIndexRef.current;
    const queue = batchQueueRef.current;
    if (idx >= queue.length) {
      setRunning(false);
      wsRef.current = null;
      refreshGenRuns(selectedRetrievalIdRef.current).catch(console.error);
      refreshSummary().catch(console.error);
      return;
    }
    const retrievalId = queue[idx];
    const run = retrievalRuns.find((r) => r.run_id === retrievalId);
    setCurrentBatchLabel(labelFor(run?.model));
    setBatchPos([idx + 1, queue.length]);
    setProgress({});

    const cfg: GenerationConfig = {
      retrieval_run_id: retrievalId,
      k_values: batchKsRef.current,
      qtypes: batchQtypesRef.current.length === QTYPES.length ? null : batchQtypesRef.current,
      comment: batchCommentRef.current,
      skip_judge: batchSkipJudgeRef.current,
    };

    let doneFired = false;
    const ws = openGenSocket(cfg, (e: GenerationEvent) => {
      if (e.type === "progress") {
        setProgress((p) => ({
          ...p,
          [e.k]: { current: e.current, total: e.total, done: p[e.k]?.done ?? false },
        }));
      } else if (e.type === "k_done") {
        setProgress((p) => ({
          ...p,
          [e.k]: {
            current: p[e.k]?.total ?? 0,
            total: p[e.k]?.total ?? 0,
            done: true,
            elapsed_s: e.elapsed_s,
          },
        }));
        refreshGenRuns(retrievalId).catch(console.error);
        refreshSummary().catch(console.error);
      } else if (e.type === "done") {
        doneFired = true;
        batchIndexRef.current++;
        refreshGenRuns(selectedRetrievalIdRef.current).catch(console.error);
        refreshSummary().catch(console.error);
        openBatchItem(); // reads from refs — no stale state
      } else if (e.type === "error") {
        setErrorLog((es) => [
          ...es,
          `[${retrievalId}] k=${e.k ?? ""} ${e.qid ?? ""}: ${e.message}`,
        ]);
      }
    });
    ws.onclose = () => {
      if (!doneFired) {
        setRunning(false);
        wsRef.current = null;
      }
    };
    ws.onerror = () => {
      setErrorLog((es) => [...es, "WebSocket error"]);
      setRunning(false);
      wsRef.current = null;
    };
    wsRef.current = ws;
  };

  const handleRun = () => {
    const ids =
      batchRetrievalIds.length > 0
        ? batchRetrievalIds
        : selectedRetrievalId
          ? [selectedRetrievalId]
          : [];
    if (ids.length === 0 || ks.length === 0 || running) return;
    // Snapshot all config into refs so every batch item uses the same settings.
    batchQueueRef.current = ids;
    batchIndexRef.current = 0;
    batchKsRef.current = [...new Set(ks)].sort((a, b) => a - b);
    batchQtypesRef.current = [...qtypes];
    batchCommentRef.current = comment;
    batchSkipJudgeRef.current = skipJudge;
    setRunning(true);
    setProgress({});
    setErrorLog([]);
    setShowConfig(false);
    openBatchItem();
  };

  const handleCancel = () => {
    wsRef.current?.close();
    wsRef.current = null;
    setRunning(false);
  };

  const handleSelectGen = async (id: string) => {
    if (selectedGen?.run_id === id) {
      setSelectedGen(null);
      return;
    }
    try {
      setSelectedGen(await fetchGenRun(id));
    } catch (e) {
      console.error(e);
    }
  };

  const handleDeleteGen = async (id: string) => {
    if (!confirm(`Delete generation run ${id}?`)) return;
    try {
      await deleteGenRun(id);
      if (selectedGen?.run_id === id) setSelectedGen(null);
      await refreshGenRuns(selectedRetrievalId);
      await refreshSummary();
    } catch (e) {
      console.error(e);
    }
  };

  const handleToggleStarGen = async (id: string) => {
    const run = genRuns.find((r) => r.run_id === id);
    if (!run) return;
    const next = !run.starred;
    try {
      await patchGenRunStarred(id, next);
      setGenRuns((rs) => rs.map((r) => (r.run_id === id ? { ...r, starred: next } : r)));
    } catch (e) {
      console.error(e);
    }
  };

  const handleUpdateGenComment = async (id: string, value: string) => {
    try {
      await patchGenRunComment(id, value);
      setGenRuns((rs) => rs.map((r) => (r.run_id === id ? { ...r, comment: value } : r)));
      if (selectedGen?.run_id === id) setSelectedGen({ ...selectedGen, comment: value });
      await refreshSummary();
    } catch (e) {
      console.error(e);
    }
  };

  const toggleK = (k: number) => {
    setKs((cur) => (cur.includes(k) ? cur.filter((x) => x !== k) : [...cur, k]));
  };
  const addCustomK = () => {
    const v = Number(customK);
    if (!Number.isInteger(v) || v < 1 || v > 20) return;
    if (!ks.includes(v)) setKs((cur) => [...cur, v]);
    setCustomK("");
  };
  const toggleQtype = (t: QuestionType) => {
    setQtypes((cur) => (cur.includes(t) ? cur.filter((x) => x !== t) : [...cur, t]));
  };
  const toggleBatch = (id: string) => {
    setBatchRetrievalIds((cur) =>
      cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id],
    );
  };

  const handleUpdateRetrievalComment = async (runId: string, comment: string) => {
    try {
      await patchRunComment(runId, comment);
      setRetrievalRuns((rs) =>
        rs.map((r) => (r.run_id === runId ? { ...r, comment } : r)),
      );
      await refreshSummary();
    } catch (e) {
      console.error(e);
    }
  };

  // Set of retrieval-run IDs that already have at least one generation run.
  // Source of truth = the gen summary endpoint, which lists exactly the
  // retrievals that have child gen runs.
  const hasGenIds = new Set<string>(
    summary?.rows.map((row) => row.retrieval_run_id) ?? [],
  );

  return (
    <div className="h-full flex">
      <aside className="w-80 shrink-0 h-full flex flex-col border-r border-border bg-panel">
        <div className="p-3 border-b border-border space-y-1">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-muted">
            Retrieval runs ({retrievalRuns.length})
          </h2>
          <p className="text-[11px] text-muted leading-tight">
            Click to browse its gen runs. ✓ = already has gen runs. Check boxes to
            queue for batch generation.
          </p>
          {batchRetrievalIds.length > 0 && (
            <div className="flex items-center justify-between text-[11px]">
              <span className="text-accent font-medium">{batchRetrievalIds.length} queued</span>
              <button
                className="text-muted hover:text-ink"
                onClick={() => setBatchRetrievalIds([])}
              >
                clear
              </button>
            </div>
          )}
        </div>
        <div className="flex-1 overflow-y-auto">
          <RetrievalSidebarSections
            runs={retrievalRuns}
            hasGenIds={hasGenIds}
            selectedRetrievalId={selectedRetrievalId}
            batchRetrievalIds={batchRetrievalIds}
            running={running}
            onSelect={setSelectedRetrievalId}
            onToggleBatch={toggleBatch}
            onUpdateComment={handleUpdateRetrievalComment}
          />
          {retrievalRuns.length === 0 && (
            <div className="p-4 text-sm text-muted">
              No retrieval runs. Go to the Retrieval page and run one first.
            </div>
          )}
        </div>
      </aside>
      <div className="flex-1 overflow-y-auto px-8 py-6">
        <div className="max-w-6xl mx-auto space-y-6">
          <div className="flex items-baseline justify-between">
            <h1 className="text-xl font-semibold">Generation evaluation</h1>
            <button
              className={`btn ${showConfig ? "btn-primary" : ""}`}
              onClick={() => setShowConfig((s) => !s)}
              disabled={!selectedRetrievalId && batchRetrievalIds.length === 0}
            >
              {showConfig ? "Hide config" : "+ New generation run"}
            </button>
          </div>

          {showConfig && (selectedRetrievalId || batchRetrievalIds.length > 0) && (
            <ConfigCard
              ks={ks}
              customK={customK}
              setCustomK={setCustomK}
              addCustomK={addCustomK}
              toggleK={toggleK}
              qtypes={qtypes}
              toggleQtype={toggleQtype}
              comment={comment}
              setComment={setComment}
              skipJudge={skipJudge}
              setSkipJudge={setSkipJudge}
              batchCount={batchRetrievalIds.length > 0 ? batchRetrievalIds.length : 1}
              running={running}
              onRun={handleRun}
              onCancel={handleCancel}
              onClose={() => setShowConfig(false)}
            />
          )}

          {(running || Object.keys(progress).length > 0) && (
            <ProgressCard
              progress={progress}
              batchPos={batchPos}
              batchLabel={currentBatchLabel}
              running={running}
            />
          )}

          {summary && summary.rows.length > 0 && (
            <SummaryTable
              summary={summary}
              metric={summaryMetric}
              setMetric={setSummaryMetric}
              selectedRetrievalId={selectedRetrievalId}
              onCellClick={async (retrievalId, runId) => {
                setSelectedRetrievalId(retrievalId);
                try {
                  setSelectedGen(await fetchGenRun(runId));
                } catch (e) {
                  console.error(e);
                }
              }}
            />
          )}

          {summary && selectedRetrievalId && (
            (() => {
              const row = summary.rows.find(
                (r) => r.retrieval_run_id === selectedRetrievalId,
              );
              return row ? (
                <AcrossKTable row={row} ks={summary.ks} />
              ) : null;
            })()
          )}

          {selectedRetrievalId && genRuns.length > 0 && (
            <SpearmanAcrossK genRuns={genRuns} detailsByRunId={genDetailsByRunId} />
          )}

          {selectedRetrievalId && genRuns.length > 0 && (
            <ConfusionAcrossK genRuns={genRuns} detailsByRunId={genDetailsByRunId} />
          )}

          <GenRunsList
            runs={genRuns}
            selectedId={selectedGen?.run_id ?? null}
            onSelect={handleSelectGen}
            onDelete={handleDeleteGen}
            onToggleStar={handleToggleStarGen}
            onUpdateComment={handleUpdateGenComment}
          />

          {selectedGen && (
            <GenRunDetailView run={selectedGen} onUpdateComment={handleUpdateGenComment} />
          )}

          {!selectedGen && genRuns.length === 0 && !running && selectedRetrievalId && (
            <div className="border border-dashed border-border rounded-md bg-panel p-12 text-center">
              <h2 className="text-lg font-medium mb-2">No generation runs yet</h2>
              <p className="text-sm text-muted mb-4">
                Click "+ New generation run" to feed this retrieval's top-k passages to the LLM.
              </p>
            </div>
          )}

          {errorLog.length > 0 && (
            <div className="border border-accent/40 rounded-md bg-panel p-3 text-xs text-accent">
              <div className="font-medium mb-1">Errors ({errorLog.length})</div>
              <ul className="space-y-0.5 max-h-32 overflow-y-auto">
                {errorLog.map((e, i) => (
                  <li key={i}>{e}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// -------------------- sidebar (retrieval runs) --------------------

function RetrievalSidebarSections({
  runs,
  hasGenIds,
  selectedRetrievalId,
  batchRetrievalIds,
  running,
  onSelect,
  onToggleBatch,
  onUpdateComment,
}: {
  runs: RunSummary[];
  hasGenIds: Set<string>;
  selectedRetrievalId: string | null;
  batchRetrievalIds: string[];
  running: boolean;
  onSelect: (id: string) => void;
  onToggleBatch: (id: string) => void;
  onUpdateComment: (id: string, v: string) => void;
}) {
  const withGen = runs.filter((r) => hasGenIds.has(r.run_id));
  const withoutGen = runs.filter((r) => !hasGenIds.has(r.run_id));

  return (
    <>
      <SidebarGroup
        title={`✓ With generation runs (${withGen.length})`}
        runs={withGen}
        hasGenIds={hasGenIds}
        defaultOpen
        selectedRetrievalId={selectedRetrievalId}
        batchRetrievalIds={batchRetrievalIds}
        running={running}
        onSelect={onSelect}
        onToggleBatch={onToggleBatch}
        onUpdateComment={onUpdateComment}
      />
      <SidebarGroup
        title={`○ No generation runs yet (${withoutGen.length})`}
        runs={withoutGen}
        hasGenIds={hasGenIds}
        defaultOpen={withGen.length === 0}
        selectedRetrievalId={selectedRetrievalId}
        batchRetrievalIds={batchRetrievalIds}
        running={running}
        onSelect={onSelect}
        onToggleBatch={onToggleBatch}
        onUpdateComment={onUpdateComment}
      />
    </>
  );
}

function SidebarGroup({
  title,
  runs,
  hasGenIds,
  defaultOpen,
  selectedRetrievalId,
  batchRetrievalIds,
  running,
  onSelect,
  onToggleBatch,
  onUpdateComment,
}: {
  title: string;
  runs: RunSummary[];
  hasGenIds: Set<string>;
  defaultOpen: boolean;
  selectedRetrievalId: string | null;
  batchRetrievalIds: string[];
  running: boolean;
  onSelect: (id: string) => void;
  onToggleBatch: (id: string) => void;
  onUpdateComment: (id: string, v: string) => void;
}) {
  const [open, setOpen] = useState(defaultOpen);
  // Stable order: known retrieval models first by hash, then alphabetic by model.
  const byModel = new Map<string, RunSummary[]>();
  for (const r of runs) {
    const list = byModel.get(r.model) ?? [];
    list.push(r);
    byModel.set(r.model, list);
  }
  const modelGroups = Array.from(byModel.entries()).sort(([a], [b]) => a.localeCompare(b));

  if (runs.length === 0) return null;
  return (
    <div className="border-b border-border">
      <button
        className="w-full px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted bg-bg/40 hover:bg-bg flex items-center justify-between"
        onClick={() => setOpen((v) => !v)}
      >
        <span>{title}</span>
        <span>{open ? "▼" : "▶"}</span>
      </button>
      {open && modelGroups.map(([model, list]) => {
        const c = modelColor(model);
        return (
          <div key={model}>
            <div className={`px-3 py-1 text-[10px] font-semibold uppercase bg-bg/20 border-b border-border/40 flex items-center gap-1.5 ${c.text}`}>
              <span className={`inline-block w-1.5 h-1.5 rounded-full ${c.dot}`} />
              {labelFor(model)}
              <span className="ml-auto text-muted">{list.length}</span>
            </div>
            {list.map((r) => (
              <SidebarRow
                key={r.run_id}
                run={r}
                hasGen={hasGenIds.has(r.run_id)}
                isSelected={selectedRetrievalId === r.run_id}
                isBatched={batchRetrievalIds.includes(r.run_id)}
                running={running}
                onSelect={onSelect}
                onToggleBatch={onToggleBatch}
                onUpdateComment={onUpdateComment}
              />
            ))}
          </div>
        );
      })}
    </div>
  );
}

function SidebarRow({
  run: r,
  hasGen,
  isSelected,
  isBatched,
  running,
  onSelect,
  onToggleBatch,
  onUpdateComment,
}: {
  run: RunSummary;
  hasGen: boolean;
  isSelected: boolean;
  isBatched: boolean;
  running: boolean;
  onSelect: (id: string) => void;
  onToggleBatch: (id: string) => void;
  onUpdateComment: (id: string, v: string) => void;
}) {
  const c = modelColor(r.model);
  return (
    <div
      className={`flex items-start gap-2 px-2 py-2 border-b border-border/50 hover:bg-bg ${
        isSelected ? "bg-bg" : ""
      }`}
    >
      <span className={`mt-1 inline-block w-1 self-stretch rounded-full ${c.bar}`} aria-hidden />
      <input
        type="checkbox"
        className="mt-0.5 accent-ink shrink-0"
        checked={isBatched}
        onChange={() => onToggleBatch(r.run_id)}
        onClick={(e) => e.stopPropagation()}
        disabled={running}
      />
      <div className="flex-1 min-w-0">
        <button
          className="block w-full text-left"
          onClick={() => onSelect(r.run_id)}
        >
          <div className="text-[11px] text-muted leading-tight flex items-center gap-1">
            {hasGen && <span className="text-emerald-600 font-semibold">✓</span>}
            <span>
              {new Date(r.ended_at * 1000).toLocaleString(undefined, {
                month: "short",
                day: "2-digit",
                hour: "2-digit",
                minute: "2-digit",
              })}
            </span>
            <span className="text-muted/60">·</span>
            <span>{r.n_queries} q</span>
            <span className="text-muted/60">·</span>
            <span>top-{r.config?.save_topn ?? 20}</span>
          </div>
        </button>
        <CommentInline
          value={r.comment ?? ""}
          onSave={(v) => onUpdateComment(r.run_id, v)}
        />
      </div>
    </div>
  );
}

// -------------------- config --------------------

function ConfigCard({
  ks,
  customK,
  setCustomK,
  addCustomK,
  toggleK,
  qtypes,
  toggleQtype,
  comment,
  setComment,
  skipJudge,
  setSkipJudge,
  batchCount,
  running,
  onRun,
  onCancel,
  onClose,
}: {
  ks: number[];
  customK: string;
  setCustomK: (s: string) => void;
  addCustomK: () => void;
  toggleK: (k: number) => void;
  qtypes: QuestionType[];
  toggleQtype: (t: QuestionType) => void;
  comment: string;
  setComment: (s: string) => void;
  skipJudge: boolean;
  setSkipJudge: (v: boolean) => void;
  batchCount: number;
  running: boolean;
  onRun: () => void;
  onCancel: () => void;
  onClose: () => void;
}) {
  return (
    <section className="border border-border rounded-md bg-panel p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium">
          New generation run
          {batchCount > 1 && (
            <span className="ml-2 text-xs text-accent font-normal">
              × {batchCount} retrieval runs
            </span>
          )}
        </h2>
        <button className="text-xs text-muted hover:text-ink" onClick={onClose}>
          ×
        </button>
      </div>
      <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <span className="text-xs text-muted mr-1">k values</span>
          {PRESET_KS.map((k) => (
            <button
              key={k}
              className={`chip ${ks.includes(k) ? "chip-active" : ""}`}
              onClick={() => toggleK(k)}
              disabled={running}
            >
              {k}
            </button>
          ))}
          <input
            type="number"
            min={1}
            max={20}
            placeholder="custom"
            className="input !w-20 !py-1"
            value={customK}
            onChange={(e) => setCustomK(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") addCustomK();
            }}
            disabled={running}
          />
          <button className="btn !py-0.5 !text-xs" onClick={addCustomK} disabled={running}>
            + add
          </button>
          {ks.filter((k) => !PRESET_KS.includes(k)).map((k) => (
            <button
              key={k}
              className="chip chip-active"
              onClick={() => toggleK(k)}
              disabled={running}
            >
              {k} ×
            </button>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <span className="text-xs text-muted mr-1">qtypes</span>
          {QTYPES.map((t) => (
            <label key={t} className="flex items-center gap-1 text-sm">
              <input
                type="checkbox"
                className="accent-ink"
                checked={qtypes.includes(t)}
                onChange={() => toggleQtype(t)}
                disabled={running}
              />
              <span className="capitalize">{t}</span>
            </label>
          ))}
        </div>
        <label className="flex items-center gap-2 text-xs text-muted">
          comment
          <input
            type="text"
            className="flex-1 border border-border rounded-md bg-bg px-2 py-1 text-sm text-ink min-w-[280px]"
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            disabled={running}
            placeholder="optional — what is this generation run for?"
          />
        </label>
        <label className="flex items-center gap-2 text-xs text-muted cursor-pointer">
          <input
            type="checkbox"
            className="accent-ink"
            checked={skipJudge}
            onChange={(e) => setSkipJudge(e.target.checked)}
            disabled={running}
          />
          <span>
            Skip LLM judge for summary
            <span className="ml-1 text-[10px] text-muted/70">
              (ROUGE-L &amp; BERTScore still computed — saves ~1 LLM call per summary question)
            </span>
          </span>
        </label>
        <div className="flex items-center gap-2">
          {running ? (
            <button className="btn" onClick={onCancel}>
              Cancel
            </button>
          ) : (
            <button
              className="btn btn-primary"
              onClick={onRun}
              disabled={ks.length === 0 || qtypes.length === 0}
            >
              Run{batchCount > 1 ? ` (${batchCount} retrievals)` : ""}
            </button>
          )}
          <span className="text-xs text-muted">
            {ks.length} k × {qtypes.length} qtypes. Each k = one saved gen run.
          </span>
        </div>
      </div>
    </section>
  );
}

// -------------------- progress --------------------

function ProgressCard({
  progress,
  batchPos,
  batchLabel,
  running,
}: {
  progress: Record<number, KProgress>;
  batchPos: [number, number];
  batchLabel: string;
  running: boolean;
}) {
  const ks = Object.keys(progress)
    .map(Number)
    .sort((a, b) => a - b);
  return (
    <section className="border border-border rounded-md bg-panel p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium">Progress</h2>
        {batchPos[1] > 1 && (
          <span className="text-xs text-muted">
            retrieval {batchPos[0]}/{batchPos[1]}
            {batchLabel ? ` — ${batchLabel}` : ""}
            {!running && " (done)"}
          </span>
        )}
      </div>
      <div className="space-y-2">
        {ks.map((k) => {
          const p = progress[k];
          const pct = p.total > 0 ? Math.round((p.current / p.total) * 100) : 0;
          return (
            <div key={k} className="flex items-center gap-3 text-sm">
              <span className="w-16 font-mono">k={k}</span>
              <div className="flex-1 h-2 bg-bg rounded-full overflow-hidden">
                <div
                  className={`h-full ${p.done ? "bg-ink" : "bg-accent"} transition-all`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="text-xs text-muted font-mono w-32 text-right">
                {p.done
                  ? `done in ${p.elapsed_s?.toFixed(1)}s`
                  : `${p.current}/${p.total}`}
              </span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

// -------------------- gen runs list --------------------

function CommentInline({
  value,
  onSave,
}: {
  value: string;
  onSave: (v: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);
  if (!editing) {
    return (
      <button
        className={`text-[11px] italic hover:text-ink ${value ? "text-amber-700" : "text-muted"}`}
        onClick={(e) => {
          e.stopPropagation();
          setEditing(true);
        }}
      >
        {value ? value : "+ add comment"}
      </button>
    );
  }
  const commit = () => {
    setEditing(false);
    if (draft !== value) onSave(draft);
  };
  return (
    <input
      autoFocus
      type="text"
      className="px-1.5 py-0.5 text-[11px] border border-border rounded bg-bg"
      value={draft}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") commit();
        if (e.key === "Escape") {
          setDraft(value);
          setEditing(false);
        }
      }}
      onClick={(e) => e.stopPropagation()}
    />
  );
}

function SummaryTable({
  summary,
  metric,
  setMetric,
  selectedRetrievalId,
  onCellClick,
}: {
  summary: GenerationSummary;
  metric: SummaryMetricKey;
  setMetric: (k: SummaryMetricKey) => void;
  selectedRetrievalId: string | null;
  onCellClick: (retrievalId: string, runId: string) => void;
}) {
  return (
    <section className="border border-border rounded-md bg-panel overflow-hidden">
      <div className="px-3 py-2 text-xs font-medium text-muted border-b border-border flex items-center gap-3">
        <span>Model × k generation summary</span>
        <select
          className="bg-bg border border-border rounded px-2 py-0.5 text-xs"
          value={metric}
          onChange={(e) => setMetric(e.target.value as SummaryMetricKey)}
        >
          {SUMMARY_METRICS.map((m) => (
            <option key={m.key} value={m.key}>
              {m.label}
            </option>
          ))}
        </select>
        <span className="ml-auto text-[11px] text-muted">
          click any cell to load that gen run
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-muted bg-bg/40">
              <th className="px-3 py-1.5 text-left font-medium">model</th>
              <th className="px-3 py-1.5 text-left font-medium">retrieval run</th>
              {summary.ks.map((k) => (
                <th key={k} className="px-3 py-1.5 text-right font-medium">
                  k={k}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {summary.rows.map((row) => {
              const c = modelColor(row.retrieval_model);
              const isSelected = selectedRetrievalId === row.retrieval_run_id;
              return (
                <tr
                  key={row.retrieval_run_id}
                  className={`border-t border-border/50 ${isSelected ? "bg-bg/60" : ""}`}
                >
                  <td className="px-3 py-1.5 font-medium whitespace-nowrap">
                    <span className={`inline-flex items-center gap-1.5`}>
                      <span className={`inline-block w-2 h-2 rounded-full ${c.dot}`} />
                      <span className={c.text}>{labelFor(row.retrieval_model)}</span>
                    </span>
                  </td>
                  <td
                    className="px-3 py-1.5 font-mono text-[11px] text-muted truncate max-w-[260px]"
                    title={`${row.retrieval_run_id}${
                      row.retrieval_comment ? ` — ${row.retrieval_comment}` : ""
                    }`}
                  >
                    {row.retrieval_run_id}
                    {row.retrieval_comment && (
                      <div className="italic text-amber-700">{row.retrieval_comment}</div>
                    )}
                  </td>
                  {summary.ks.map((k) => {
                    const cell = row.cells[String(k)];
                    if (!cell) {
                      return (
                        <td key={k} className="px-3 py-1.5 text-right text-muted">
                          –
                        </td>
                      );
                    }
                    const v = readSummaryCell(cell, metric);
                    return (
                      <td
                        key={k}
                        className="px-3 py-1.5 text-right font-mono tabular-nums cursor-pointer hover:bg-accent/10"
                        title={cell.comment ? `${cell.run_id}\n${cell.comment}` : cell.run_id}
                        onClick={() => onCellClick(row.retrieval_run_id, cell.run_id)}
                      >
                        {v == null ? "—" : fmtMetric(v)}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

// -------------------- across-k summary (per retrieval run) --------------------

interface AcrossKMetricRow {
  qtype: QuestionType;
  label: string;
  // returns the value at this k, or null if not available
  read: (cell: GenerationSummaryCell) => number | null;
}

const ACROSS_K_ROWS: AcrossKMetricRow[] = [
  { qtype: "factoid", label: "MRR", read: (c) => c.qa_by_qtype.factoid ?? null },
  { qtype: "factoid", label: "Strict Acc", read: (c) => c.factoid_strict_acc ?? null },
  { qtype: "yesno", label: "Accuracy", read: (c) => c.qa_by_qtype.yesno ?? null },
  { qtype: "list", label: "F1", read: (c) => c.qa_by_qtype.list ?? null },
  { qtype: "list", label: "MAP", read: (c) => c.list_map ?? null },
  { qtype: "summary", label: "ROUGE-L", read: (c) => c.summary_rouge_l ?? null },
  { qtype: "summary", label: "BERTScore", read: (c) => c.summary_bert_score ?? null },
  {
    qtype: "summary",
    label: "Judge (LLM)",
    read: (c) => (c.skip_judge ? null : c.qa_by_qtype.summary ?? null),
  },
];

function AcrossKTable({ row, ks }: { row: GenerationSummaryRow; ks: number[] }) {
  // Show only ks that actually have a saved cell, sorted ascending.
  const presentKs = ks.filter((k) => row.cells[String(k)]);
  const c = modelColor(row.retrieval_model);
  return (
    <section className="border border-border rounded-md bg-panel overflow-hidden">
      <div className="px-3 py-2 text-xs font-medium border-b border-border flex items-center gap-2">
        <span className={`inline-block w-2 h-2 rounded-full ${c.dot}`} />
        <span className={c.text}>{labelFor(row.retrieval_model)}</span>
        <span className="text-muted">— metrics across k</span>
        <span className="ml-auto text-[11px] text-muted">
          one row per (qtype, metric); columns = k
        </span>
      </div>
      {presentKs.length === 0 ? (
        <div className="p-4 text-xs text-muted">
          No generation runs yet for this retrieval. Click "+ New generation run".
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-muted bg-bg/40">
                <th className="px-3 py-1.5 text-left font-medium">Type</th>
                <th className="px-3 py-1.5 text-left font-medium">Metric</th>
                {presentKs.map((k) => (
                  <th key={k} className="px-3 py-1.5 text-right font-medium">
                    k={k}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ACROSS_K_ROWS.map((mrow) => {
                // Skip rows that are entirely empty for this retrieval.
                const hasAny = presentKs.some((k) => {
                  const cell = row.cells[String(k)];
                  return cell ? mrow.read(cell) != null : false;
                });
                if (!hasAny) return null;
                return (
                  <tr
                    key={`${mrow.qtype}-${mrow.label}`}
                    className="border-t border-border/50"
                  >
                    <td className="px-3 py-1.5 capitalize text-muted">{mrow.qtype}</td>
                    <td className="px-3 py-1.5 font-mono">{mrow.label}</td>
                    {presentKs.map((k) => {
                      const cell = row.cells[String(k)];
                      if (!cell) {
                        return (
                          <td key={k} className="px-3 py-1.5 text-right text-muted">
                            –
                          </td>
                        );
                      }
                      const v = mrow.read(cell);
                      return (
                        <td
                          key={k}
                          className="px-3 py-1.5 text-right font-mono tabular-nums"
                          title={cell.run_id}
                        >
                          {v == null ? "—" : fmtMetric(v)}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

// -------------------- side-by-side Spearman + confusion --------------------

// Sort gen runs by k ascending so the side-by-side view reads left-to-right.
function sortByK<T extends { k: number }>(arr: T[]): T[] {
  return [...arr].sort((a, b) => a.k - b.k);
}

function MiniCorrelationMatrix({ matrix }: { matrix: SpearmanMatrix }) {
  const colorFor = (v: number, isDiag: boolean) => {
    if (isDiag) return "bg-bg/40 text-ink";
    const a = Math.abs(v);
    if (v >= 0) {
      if (a >= 0.5) return "bg-emerald-200 text-emerald-900";
      if (a >= 0.3) return "bg-emerald-100 text-emerald-900";
      if (a >= 0.1) return "bg-emerald-50 text-emerald-900";
    } else {
      if (a >= 0.5) return "bg-red-200 text-red-900";
      if (a >= 0.3) return "bg-red-100 text-red-900";
      if (a >= 0.1) return "bg-red-50 text-red-900";
    }
    return "text-muted";
  };
  return (
    <table className="text-[11px]">
      <thead>
        <tr className="text-muted">
          <th></th>
          {matrix.variables.map((v) => (
            <th key={v} className="px-1.5 py-0.5 font-medium text-right">
              {v}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {matrix.matrix.map((row, i) => (
          <tr key={i}>
            <td className="px-1.5 py-0.5 text-muted font-medium">{matrix.variables[i]}</td>
            {row.map((v, j) => (
              <td
                key={j}
                className={`px-1.5 py-0.5 text-right font-mono tabular-nums ${colorFor(v, i === j)}`}
              >
                {v.toFixed(2)}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function SpearmanAcrossK({
  genRuns,
  detailsByRunId,
}: {
  genRuns: GenerationRunSummary[];
  detailsByRunId: Record<string, GenerationRunDetail>;
}) {
  // Resolve loaded details, dedupe per k (keep latest by ended_at).
  const sortedRuns = sortByK(genRuns);
  const byK = new Map<number, GenerationRunDetail>();
  for (const r of sortedRuns) {
    const d = detailsByRunId[r.run_id];
    if (!d) continue;
    const existing = byK.get(r.k);
    if (!existing || (d.ended_at ?? 0) > (existing.ended_at ?? 0)) {
      byK.set(r.k, d);
    }
  }
  const ks = Array.from(byK.keys()).sort((a, b) => a - b);
  const totalLoaded = ks.length;
  const pendingCount = genRuns.length - totalLoaded;
  if (totalLoaded === 0) {
    return (
      <section className="border border-border rounded-md bg-panel p-4 text-xs text-muted">
        Loading Spearman matrices for {genRuns.length} gen run(s)…
      </section>
    );
  }
  return (
    <section className="border border-border rounded-md bg-panel overflow-hidden">
      <div className="px-3 py-2 text-xs font-medium border-b border-border flex items-center gap-2">
        <span className="text-ink">Spearman correlation across k</span>
        <span className="text-muted">— each row is one qtype</span>
        {pendingCount > 0 && (
          <span className="ml-auto text-[11px] text-muted">
            {pendingCount} more loading…
          </span>
        )}
      </div>
      <div className="p-3 space-y-4">
        {QTYPES.map((qtype) => {
          // For each qtype, render one mini-matrix per k that has a matrix.
          const cells = ks
            .map((k) => {
              const det = byK.get(k);
              const m = det?.correlations?.[qtype];
              return m ? { k, matrix: m } : null;
            })
            .filter((x): x is { k: number; matrix: SpearmanMatrix } => x !== null);
          if (cells.length === 0) return null;
          return (
            <div key={qtype}>
              <div className="text-xs font-semibold text-ink capitalize mb-2">
                {qtype}{" "}
                <span className="text-muted font-normal">
                  ({QTYPE_METRIC_LABEL[qtype]})
                </span>
              </div>
              <div className="flex flex-wrap gap-4">
                {cells.map((c) => (
                  <div key={c.k} className="border border-border/60 rounded p-2 bg-bg/30">
                    <div className="text-[11px] font-mono text-muted mb-1">
                      k={c.k}{" "}
                      <span className="text-muted/70">· n={c.matrix.n}</span>
                    </div>
                    <MiniCorrelationMatrix matrix={c.matrix} />
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function ConfusionAcrossK({
  genRuns,
  detailsByRunId,
}: {
  genRuns: GenerationRunSummary[];
  detailsByRunId: Record<string, GenerationRunDetail>;
}) {
  const sortedRuns = sortByK(genRuns);
  const byK = new Map<number, GenerationRunDetail>();
  for (const r of sortedRuns) {
    const d = detailsByRunId[r.run_id];
    if (!d) continue;
    const existing = byK.get(r.k);
    if (!existing || (d.ended_at ?? 0) > (existing.ended_at ?? 0)) {
      byK.set(r.k, d);
    }
  }
  const ks = Array.from(byK.keys()).sort((a, b) => a - b);
  const cells = ks
    .map((k) => {
      const det = byK.get(k);
      if (!det) return null;
      const yesnoQs = Object.values(det.per_query).filter((q) => q.qtype === "yesno");
      const stats = computeConfusion(yesnoQs);
      return stats ? { k, stats } : null;
    })
    .filter((x): x is { k: number; stats: ConfusionStats } => x !== null);
  if (cells.length === 0) return null;
  return (
    <section className="border border-border rounded-md bg-panel overflow-hidden">
      <div className="px-3 py-2 text-xs font-medium border-b border-border flex items-center gap-2">
        <span className="text-ink">Yes/No confusion across k</span>
        <span className="text-muted">— one matrix per k</span>
      </div>
      <div className="p-3 flex flex-wrap gap-4">
        {cells.map((c) => (
          <div key={c.k} className="border border-border/60 rounded p-2 bg-bg/30">
            <div className="text-[11px] font-mono text-muted mb-1 flex items-baseline justify-between gap-2">
              <span>
                k={c.k}{" "}
                <span className="text-muted/70">· n={c.stats.total}</span>
              </span>
              <span className="text-ink">
                acc={(c.stats.accuracy * 100).toFixed(1)}%
              </span>
            </div>
            <MiniConfusion stats={c.stats} />
          </div>
        ))}
      </div>
    </section>
  );
}

function GenRunsList({
  runs,
  selectedId,
  onSelect,
  onDelete,
  onToggleStar,
  onUpdateComment,
}: {
  runs: GenerationRunSummary[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onToggleStar: (id: string) => void;
  onUpdateComment: (id: string, v: string) => void;
}) {
  if (runs.length === 0) return null;
  // Starred runs float to the top.
  const sorted = [...runs].sort((a, b) => {
    if (a.starred && !b.starred) return -1;
    if (!a.starred && b.starred) return 1;
    return (b.ended_at ?? 0) - (a.ended_at ?? 0);
  });
  const starredCount = runs.filter((r) => r.starred).length;
  return (
    <section className="border border-border rounded-md bg-panel overflow-hidden">
      <div className="px-3 py-2 text-xs font-medium text-muted border-b border-border flex items-center gap-2">
        <span>Generation runs for this retrieval ({runs.length})</span>
        {starredCount > 0 && (
          <span className="text-amber-500 font-medium">★ {starredCount} starred</span>
        )}
      </div>
      <ul>
        {sorted.map((r) => {
          const sel = selectedId === r.run_id;
          return (
            <li
              key={r.run_id}
              className={`px-3 py-2 border-b border-border/50 text-sm flex items-center gap-3 cursor-pointer hover:bg-bg/60 ${
                sel ? "bg-bg" : ""
              } ${r.starred ? "border-l-2 border-l-amber-400" : ""}`}
              onClick={() => onSelect(r.run_id)}
            >
              <button
                title={r.starred ? "Unstar" : "Star this run"}
                className={`text-base leading-none shrink-0 ${
                  r.starred ? "text-amber-400 hover:text-amber-600" : "text-muted hover:text-amber-400"
                }`}
                onClick={(e) => { e.stopPropagation(); onToggleStar(r.run_id); }}
              >
                {r.starred ? "★" : "☆"}
              </button>
              <span className={`chip text-[10px] pointer-events-none ${sel ? "chip-active" : ""}`}>
                k={r.k}
              </span>
              <span className="text-xs text-muted font-mono">
                {new Date(r.ended_at * 1000).toLocaleString()}
              </span>
              <span className="text-xs text-muted">
                {r.elapsed_s.toFixed(1)}s · {r.n_queries} q
              </span>
              <span className="ml-auto flex gap-3 text-xs font-mono">
                {QTYPES.map((t) => {
                  const v = r.by_qtype?.[t];
                  const n = r.n_per_qtype?.[t] ?? 0;
                  return (
                    <span key={t} className="text-muted">
                      <span className="capitalize">{t.slice(0, 3)}</span>{" "}
                      {v == null ? "—" : fmtMetric(v)}{" "}
                      <span className="text-[10px]">n={n}</span>
                    </span>
                  );
                })}
              </span>
              <CommentInline
                value={r.comment ?? ""}
                onSave={(v) => onUpdateComment(r.run_id, v)}
              />
              <button
                title="Delete this run"
                className="shrink-0 px-1.5 py-0.5 text-xs rounded text-muted hover:text-white hover:bg-red-500 transition-colors"
                onClick={(e) => { e.stopPropagation(); onDelete(r.run_id); }}
              >
                🗑
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

// -------------------- selected gen run detail --------------------

function GenRunDetailView({
  run,
  onUpdateComment,
}: {
  run: GenerationRunDetail;
  onUpdateComment: (id: string, v: string) => void;
}) {
  const c = modelColor(run.retrieval_model);
  return (
    <div className="space-y-6">
      <div className="border border-border rounded-md bg-panel px-4 py-3 space-y-1.5">
        <div className="flex items-baseline justify-between">
          <div>
            <span className="font-mono font-medium">k={run.k}</span>
            <span className="text-xs text-muted ml-2">{run.run_id}</span>
          </div>
          <span className="text-xs text-muted">
            retrieval:{" "}
            <span className={`inline-flex items-center gap-1 ${c.text}`}>
              <span className={`inline-block w-2 h-2 rounded-full ${c.dot}`} />
              {labelFor(run.retrieval_model)}
            </span>{" "}
            · {run.elapsed_s.toFixed(1)}s ·{" "}
            {Object.keys(run.per_query).length} queries · num_ctx={run.config.num_ctx}
            {run.config.skip_judge && (
              <span className="ml-2 text-accent">· judge skipped</span>
            )}
          </span>
        </div>
        <CommentInline
          value={run.comment ?? ""}
          onSave={(v) => onUpdateComment(run.run_id, v)}
        />
      </div>

      <AggregateTable run={run} />

      <YesnoConfusionMatrix
        queries={Object.values(run.per_query).filter((q) => q.qtype === "yesno")}
      />

      {QTYPES.map((qtype) => {
        const matrix = run.correlations[qtype];
        if (!matrix) return null;
        const queries: GenerationPerQuery[] = Object.values(run.per_query).filter(
          (q) => q.qtype === qtype,
        );
        if (queries.length === 0) return null;
        return (
          <QtypeSection key={qtype} qtype={qtype} matrix={matrix} queries={queries} />
        );
      })}

      {run.extra_correlations?.rouge_l && (
        <ExtraMetricSection
          title="ROUGE-L"
          metricKey="rouge_l"
          matrix={run.extra_correlations.rouge_l}
          queries={Object.values(run.per_query)}
        />
      )}
      {run.extra_correlations?.bert_score && (
        <ExtraMetricSection
          title="BERTScore"
          metricKey="bert_score"
          matrix={run.extra_correlations.bert_score}
          queries={Object.values(run.per_query)}
        />
      )}
    </div>
  );
}

interface ConfusionStats {
  total: number;
  tp: number;
  tn: number;
  fp: number;
  fn: number;
  unkYes: number;
  unkNo: number;
  accuracy: number;
  yesPrec: number;
  yesRec: number;
  noPrec: number;
  noRec: number;
}

function computeConfusion(queries: GenerationPerQuery[]): ConfusionStats | null {
  if (queries.length === 0) return null;
  const counts: Record<string, Record<string, number>> = {
    yes: { yes: 0, no: 0, unknown: 0 },
    no: { yes: 0, no: 0, unknown: 0 },
  };
  let total = 0;
  for (const q of queries) {
    const gold = q.extra_metrics?.gold_label as string | undefined;
    const pred = (q.extra_metrics?.pred_label as string | undefined) ?? "unknown";
    if (gold !== "yes" && gold !== "no") continue;
    if (!counts[gold]) continue;
    counts[gold][pred] = (counts[gold][pred] ?? 0) + 1;
    total += 1;
  }
  if (total === 0) return null;
  const tp = counts.yes.yes;
  const tn = counts.no.no;
  const fp = counts.no.yes;
  const fn = counts.yes.no;
  const unkYes = counts.yes.unknown;
  const unkNo = counts.no.unknown;
  return {
    total,
    tp,
    tn,
    fp,
    fn,
    unkYes,
    unkNo,
    accuracy: (tp + tn) / total,
    yesPrec: tp + fp > 0 ? tp / (tp + fp) : 0,
    yesRec: tp + fn + unkYes > 0 ? tp / (tp + fn + unkYes) : 0,
    noPrec: tn + fn > 0 ? tn / (tn + fn) : 0,
    noRec: tn + fp + unkNo > 0 ? tn / (tn + fp + unkNo) : 0,
  };
}

function MiniConfusion({ stats }: { stats: ConfusionStats }) {
  const cellBg = (gold: string, pred: string) =>
    gold === pred ? "bg-emerald-100 text-emerald-900" : "bg-red-50 text-red-900";
  return (
    <div>
      <table className="text-xs">
        <thead>
          <tr className="text-muted">
            <th className="px-1.5 py-0.5"></th>
            <th className="px-1.5 py-0.5"></th>
            <th colSpan={3} className="px-1.5 py-0.5 text-center font-medium">
              Predicted
            </th>
          </tr>
          <tr className="text-muted">
            <th className="px-1.5 py-0.5"></th>
            <th className="px-1.5 py-0.5"></th>
            <th className="px-2 py-0.5 font-medium">Yes</th>
            <th className="px-2 py-0.5 font-medium">No</th>
            <th className="px-2 py-0.5 font-medium text-muted/70">unp</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td rowSpan={2} className="px-1.5 py-0.5 text-muted font-medium align-middle">
              Gold
            </td>
            <td className="px-1.5 py-0.5 font-medium">Yes</td>
            <td className={`px-2 py-1 text-center font-mono ${cellBg("yes", "yes")}`}>
              {stats.tp}
            </td>
            <td className={`px-2 py-1 text-center font-mono ${cellBg("yes", "no")}`}>
              {stats.fn}
            </td>
            <td className="px-2 py-1 text-center font-mono text-muted">{stats.unkYes}</td>
          </tr>
          <tr>
            <td className="px-1.5 py-0.5 font-medium">No</td>
            <td className={`px-2 py-1 text-center font-mono ${cellBg("no", "yes")}`}>
              {stats.fp}
            </td>
            <td className={`px-2 py-1 text-center font-mono ${cellBg("no", "no")}`}>
              {stats.tn}
            </td>
            <td className="px-2 py-1 text-center font-mono text-muted">{stats.unkNo}</td>
          </tr>
        </tbody>
      </table>
      <div className="mt-1.5 grid grid-cols-2 gap-x-3 text-[10px] text-muted">
        <div>Yes P: <span className="font-mono text-ink">{(stats.yesPrec * 100).toFixed(0)}%</span></div>
        <div>Yes R: <span className="font-mono text-ink">{(stats.yesRec * 100).toFixed(0)}%</span></div>
        <div>No P: <span className="font-mono text-ink">{(stats.noPrec * 100).toFixed(0)}%</span></div>
        <div>No R: <span className="font-mono text-ink">{(stats.noRec * 100).toFixed(0)}%</span></div>
      </div>
    </div>
  );
}

function YesnoConfusionMatrix({ queries }: { queries: GenerationPerQuery[] }) {
  const stats = computeConfusion(queries);
  if (!stats) return null;
  const cellBg = (gold: string, pred: string) =>
    gold === pred ? "bg-emerald-100 text-emerald-900" : "bg-red-50 text-red-900";
  const { tp, tn, fp, fn, unkYes, unkNo, total, accuracy, yesPrec, yesRec, noPrec, noRec } = stats;

  return (
    <section className="border border-border rounded-md bg-panel overflow-hidden">
      <div className="px-3 py-2 text-xs font-medium text-ink border-b border-border flex items-baseline justify-between">
        <span>Yes/No confusion matrix</span>
        <span className="text-[11px] text-muted">
          accuracy = {(accuracy * 100).toFixed(2)}% · n = {total}
        </span>
      </div>
      <div className="p-3 overflow-x-auto">
        <table className="text-xs">
          <thead>
            <tr className="text-muted">
              <th className="px-2 py-1"></th>
              <th className="px-2 py-1"></th>
              <th colSpan={3} className="px-2 py-1 text-center font-medium">
                Predicted
              </th>
            </tr>
            <tr className="text-muted">
              <th className="px-2 py-1"></th>
              <th className="px-2 py-1"></th>
              <th className="px-3 py-1 font-medium">Yes</th>
              <th className="px-3 py-1 font-medium">No</th>
              <th className="px-3 py-1 font-medium text-muted/70">unparsed</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td rowSpan={2} className="px-2 py-1 text-muted font-medium align-middle">
                Gold
              </td>
              <td className="px-2 py-1 font-medium">Yes</td>
              <td className={`px-3 py-2 text-center font-mono ${cellBg("yes", "yes")}`}>
                {tp}
              </td>
              <td className={`px-3 py-2 text-center font-mono ${cellBg("yes", "no")}`}>
                {fn}
              </td>
              <td className="px-3 py-2 text-center font-mono text-muted">{unkYes}</td>
            </tr>
            <tr>
              <td className="px-2 py-1 font-medium">No</td>
              <td className={`px-3 py-2 text-center font-mono ${cellBg("no", "yes")}`}>
                {fp}
              </td>
              <td className={`px-3 py-2 text-center font-mono ${cellBg("no", "no")}`}>
                {tn}
              </td>
              <td className="px-3 py-2 text-center font-mono text-muted">{unkNo}</td>
            </tr>
          </tbody>
        </table>
        <div className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1 text-[11px] text-muted max-w-md">
          <div>
            Yes — precision: <span className="font-mono text-ink">{(yesPrec * 100).toFixed(1)}%</span>
          </div>
          <div>
            Yes — recall: <span className="font-mono text-ink">{(yesRec * 100).toFixed(1)}%</span>
          </div>
          <div>
            No — precision: <span className="font-mono text-ink">{(noPrec * 100).toFixed(1)}%</span>
          </div>
          <div>
            No — recall: <span className="font-mono text-ink">{(noRec * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>
    </section>
  );
}

function AggregateTable({ run }: { run: GenerationRunDetail }) {
  const extra = run.aggregate.extra_by_qtype ?? {};
  return (
    <section className="border border-border rounded-md bg-panel overflow-hidden">
      <div className="px-3 py-2 text-xs font-medium text-muted border-b border-border">
        Generation aggregate (k={run.k})
      </div>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-muted">
            <th className="px-3 py-1.5 text-left font-medium">Type</th>
            <th className="px-3 py-1.5 text-left font-medium">Metric</th>
            <th className="px-3 py-1.5 text-right font-medium">Score</th>
            <th className="px-3 py-1.5 text-right font-medium">n</th>
          </tr>
        </thead>
        <tbody>
          {QTYPES.map((t) => {
            const v = run.aggregate.by_qtype[t];
            const n = run.aggregate.n_per_qtype[t] ?? 0;
            if (v == null && n === 0) return null;
            const extraRows: [string, number | undefined][] =
              t === "summary"
                ? [
                    ["rouge_l", extra.summary?.rouge_l],
                    ["bert_score", extra.summary?.bert_score],
                  ]
                : t === "factoid"
                  ? [["strict_acc", extra.factoid?.strict_acc]]
                  : t === "list"
                    ? [["map", extra.list?.map]]
                    : [];
            return (
              <>
                <tr key={t} className="border-t border-border/50">
                  <td className="px-3 py-1.5 capitalize" rowSpan={1 + extraRows.length}>
                    {t}
                  </td>
                  <td className="px-3 py-1.5 text-muted font-mono">
                    {QTYPE_METRIC_LABEL[t]}
                    {t === "summary" && run.config.skip_judge && (
                      <span className="ml-1 text-[10px] text-muted/60">(skipped)</span>
                    )}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono tabular-nums">
                    {v == null ? "—" : fmtMetric(v)}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono tabular-nums text-muted">
                    {n}
                  </td>
                </tr>
                {extraRows.map(([key, val]) => (
                  <tr key={`${t}-${key}`} className="border-t border-border/30">
                    <td className="px-3 py-1.5 text-muted font-mono">
                      {EXTRA_METRIC_LABEL[key] ?? key}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono tabular-nums">
                      {val == null ? "—" : fmtMetric(val)}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono tabular-nums text-muted">
                      —
                    </td>
                  </tr>
                ))}
              </>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}

function QtypeSection({
  qtype,
  matrix,
  queries,
}: {
  qtype: QuestionType;
  matrix: SpearmanMatrix;
  queries: GenerationPerQuery[];
}) {
  return (
    <section className="border border-border rounded-md bg-panel overflow-hidden">
      <div className="px-3 py-2 text-xs font-medium border-b border-border flex items-baseline justify-between">
        <span className="text-ink capitalize">
          {qtype} <span className="text-muted">({QTYPE_METRIC_LABEL[qtype]})</span>
        </span>
        <span className="text-xs text-muted">n = {matrix.n}</span>
      </div>
      <CorrelationMatrix matrix={matrix} />
      <PerQueryTable qtype={qtype} queries={queries} />
    </section>
  );
}

function ExtraMetricSection({
  title,
  metricKey,
  matrix,
  queries,
}: {
  title: string;
  metricKey: keyof ExtraMetrics;
  matrix: SpearmanMatrix;
  queries: GenerationPerQuery[];
}) {
  const relevant = queries.filter((q) => q.extra_metrics?.[metricKey] != null);
  return (
    <section className="border border-border rounded-md bg-panel overflow-hidden">
      <div className="px-3 py-2 text-xs font-medium border-b border-border flex items-baseline justify-between">
        <span className="text-ink">{title}</span>
        <span className="text-xs text-muted">n = {matrix.n}</span>
      </div>
      <CorrelationMatrix matrix={matrix} />
      <ExtraMetricPerQueryTable title={title} metricKey={metricKey} queries={relevant} />
    </section>
  );
}

function CorrelationMatrix({ matrix }: { matrix: SpearmanMatrix }) {
  const colorFor = (v: number, isDiag: boolean) => {
    if (isDiag) return "bg-bg/40 text-ink";
    const a = Math.abs(v);
    if (v >= 0) {
      if (a >= 0.5) return "bg-emerald-200 text-emerald-900";
      if (a >= 0.3) return "bg-emerald-100 text-emerald-900";
      if (a >= 0.1) return "bg-emerald-50 text-emerald-900";
    } else {
      if (a >= 0.5) return "bg-red-200 text-red-900";
      if (a >= 0.3) return "bg-red-100 text-red-900";
      if (a >= 0.1) return "bg-red-50 text-red-900";
    }
    return "text-muted";
  };
  return (
    <div className="px-3 py-2 border-b border-border">
      <div className="text-[11px] text-muted mb-1">Spearman correlation</div>
      <div className="overflow-x-auto">
        <table className="text-xs">
          <thead>
            <tr className="text-muted">
              <th></th>
              {matrix.variables.map((v) => (
                <th key={v} className="px-2 py-1 font-medium text-right">
                  {v}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.matrix.map((row, i) => (
              <tr key={i}>
                <td className="px-2 py-1 text-muted font-medium">{matrix.variables[i]}</td>
                {row.map((v, j) => (
                  <td
                    key={j}
                    className={`px-2 py-1 text-right font-mono tabular-nums ${colorFor(v, i === j)}`}
                  >
                    {fmtCorr(v)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PerQueryTable({
  qtype,
  queries,
}: {
  qtype: QuestionType;
  queries: GenerationPerQuery[];
}) {
  const [open, setOpen] = useState(false);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const showRougeL = qtype === "summary";
  const showBertScore = qtype === "summary";
  const showPredLabel = qtype === "yesno";
  const showStrict = qtype === "factoid";
  const showMap = qtype === "list";
  const extraCols =
    (showPredLabel ? 1 : 0) +
    (showRougeL ? 1 : 0) +
    (showBertScore ? 1 : 0) +
    (showStrict ? 1 : 0) +
    (showMap ? 1 : 0);
  const totalCols = 7 + extraCols + 1; // +1 for the chevron column

  const toggleRow = (qid: string) =>
    setExpanded((m) => ({ ...m, [qid]: !m[qid] }));

  return (
    <div>
      <button
        className="w-full px-3 py-2 text-xs font-medium text-muted hover:bg-bg flex items-center justify-between border-b border-border"
        onClick={() => setOpen((o) => !o)}
      >
        <span>Per-query results ({queries.length}) — click a row to see the answer</span>
        <span>{open ? "▼" : "▶"}</span>
      </button>
      {open && (
        <div className="overflow-x-auto max-h-[480px] overflow-y-auto">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-panel">
              <tr className="text-muted border-b border-border">
                <th className="px-1 py-1.5 w-4"></th>
                <th className="px-2 py-1.5 text-left font-medium">question</th>
                <th className="px-2 py-1.5 text-right font-medium">nDCG</th>
                <th className="px-2 py-1.5 text-right font-medium">P</th>
                <th className="px-2 py-1.5 text-right font-medium">R</th>
                <th className="px-2 py-1.5 text-right font-medium">MRR</th>
                <th className="px-2 py-1.5 text-right font-medium">MAP</th>
                <th className="px-2 py-1.5 text-right font-medium">{QTYPE_METRIC_LABEL[qtype]}</th>
                {showStrict && <th className="px-2 py-1.5 text-right font-medium">Strict</th>}
                {showMap && <th className="px-2 py-1.5 text-right font-medium">MAP (list)</th>}
                {showPredLabel && <th className="px-2 py-1.5 text-right font-medium">Pred</th>}
                {showRougeL && <th className="px-2 py-1.5 text-right font-medium">ROUGE-L</th>}
                {showBertScore && (
                  <th className="px-2 py-1.5 text-right font-medium">BERTScore</th>
                )}
              </tr>
            </thead>
            <tbody>
              {queries.map((q) => {
                const isOpen = !!expanded[q.qid];
                return (
                  <Fragment key={q.qid}>
                    <tr
                      className="border-t border-border/30 cursor-pointer hover:bg-bg/40"
                      onClick={() => toggleRow(q.qid)}
                    >
                      <td className="px-1 py-1 text-muted text-center select-none">
                        {isOpen ? "▼" : "▶"}
                      </td>
                      <td className="px-2 py-1 max-w-md truncate" title={q.question}>
                        {q.question}
                      </td>
                      <td className="px-2 py-1 text-right font-mono tabular-nums">
                        {q.retrieval_metrics ? fmtMetric(q.retrieval_metrics.ndcg) : "—"}
                      </td>
                      <td className="px-2 py-1 text-right font-mono tabular-nums">
                        {q.retrieval_metrics ? fmtMetric(q.retrieval_metrics.p) : "—"}
                      </td>
                      <td className="px-2 py-1 text-right font-mono tabular-nums">
                        {q.retrieval_metrics ? fmtMetric(q.retrieval_metrics.r) : "—"}
                      </td>
                      <td className="px-2 py-1 text-right font-mono tabular-nums">
                        {q.retrieval_metrics ? fmtMetric(q.retrieval_metrics.mrr) : "—"}
                      </td>
                      <td className="px-2 py-1 text-right font-mono tabular-nums">
                        {q.retrieval_metrics ? fmtMetric(q.retrieval_metrics.map) : "—"}
                      </td>
                      <td className="px-2 py-1 text-right font-mono tabular-nums">
                        {q.qa_score == null ? "—" : fmtMetric(q.qa_score)}
                      </td>
                      {showStrict && (
                        <td className="px-2 py-1 text-right font-mono tabular-nums">
                          {q.extra_metrics?.strict_acc == null
                            ? "—"
                            : fmtMetric(q.extra_metrics.strict_acc)}
                        </td>
                      )}
                      {showMap && (
                        <td className="px-2 py-1 text-right font-mono tabular-nums">
                          {q.extra_metrics?.map == null
                            ? "—"
                            : fmtMetric(q.extra_metrics.map)}
                        </td>
                      )}
                      {showPredLabel && (
                        <td className="px-2 py-1 text-right font-mono tabular-nums text-muted capitalize">
                          {q.extra_metrics?.pred_label ?? "—"}
                        </td>
                      )}
                      {showRougeL && (
                        <td className="px-2 py-1 text-right font-mono tabular-nums">
                          {q.extra_metrics?.rouge_l == null
                            ? "—"
                            : fmtMetric(q.extra_metrics.rouge_l)}
                        </td>
                      )}
                      {showBertScore && (
                        <td className="px-2 py-1 text-right font-mono tabular-nums">
                          {q.extra_metrics?.bert_score == null
                            ? "—"
                            : fmtMetric(q.extra_metrics.bert_score)}
                        </td>
                      )}
                    </tr>
                    {isOpen && (
                      <tr className="bg-bg/20">
                        <td></td>
                        <td colSpan={totalCols - 1} className="px-3 py-2">
                          <div className="text-[11px] text-muted mb-1">
                            <span className="font-semibold text-ink">Q:</span>{" "}
                            {q.question}
                          </div>
                          <div className="text-[11px]">
                            <span className="font-semibold text-ink">A:</span>{" "}
                            <span className="whitespace-pre-wrap font-mono text-ink">
                              {q.answer || <span className="italic text-muted">(empty)</span>}
                            </span>
                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function ExtraMetricPerQueryTable({
  title,
  metricKey,
  queries,
}: {
  title: string;
  metricKey: keyof ExtraMetrics;
  queries: GenerationPerQuery[];
}) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <button
        className="w-full px-3 py-2 text-xs font-medium text-muted hover:bg-bg flex items-center justify-between border-b border-border"
        onClick={() => setOpen((o) => !o)}
      >
        <span>
          Per-query {title} ({queries.length})
        </span>
        <span>{open ? "▼" : "▶"}</span>
      </button>
      {open && (
        <div className="overflow-x-auto max-h-[480px] overflow-y-auto">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-panel">
              <tr className="text-muted border-b border-border">
                <th className="px-2 py-1.5 text-left font-medium">question</th>
                <th className="px-2 py-1.5 text-right font-medium">qtype</th>
                <th className="px-2 py-1.5 text-right font-medium">nDCG</th>
                <th className="px-2 py-1.5 text-right font-medium">P</th>
                <th className="px-2 py-1.5 text-right font-medium">R</th>
                <th className="px-2 py-1.5 text-right font-medium">MRR</th>
                <th className="px-2 py-1.5 text-right font-medium">{title}</th>
              </tr>
            </thead>
            <tbody>
              {queries.map((q) => (
                <tr key={q.qid} className="border-t border-border/30">
                  <td
                    className="px-2 py-1 max-w-md truncate"
                    title={q.question}
                  >
                    {q.question}
                  </td>
                  <td className="px-2 py-1 text-right capitalize text-muted">{q.qtype}</td>
                  <td className="px-2 py-1 text-right font-mono tabular-nums">
                    {q.retrieval_metrics ? fmtMetric(q.retrieval_metrics.ndcg) : "—"}
                  </td>
                  <td className="px-2 py-1 text-right font-mono tabular-nums">
                    {q.retrieval_metrics ? fmtMetric(q.retrieval_metrics.p) : "—"}
                  </td>
                  <td className="px-2 py-1 text-right font-mono tabular-nums">
                    {q.retrieval_metrics ? fmtMetric(q.retrieval_metrics.r) : "—"}
                  </td>
                  <td className="px-2 py-1 text-right font-mono tabular-nums">
                    {q.retrieval_metrics ? fmtMetric(q.retrieval_metrics.mrr) : "—"}
                  </td>
                  <td className="px-2 py-1 text-right font-mono tabular-nums">
                    {(q.extra_metrics?.[metricKey] as number | null | undefined) == null
                      ? "—"
                      : fmtMetric(q.extra_metrics![metricKey] as number)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
