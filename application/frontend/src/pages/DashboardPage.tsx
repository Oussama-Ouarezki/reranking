import { useEffect, useMemo, useRef, useState } from "react";

import {
  deleteRun,
  fetchQueries,
  fetchQueryStats,
  fetchRun,
  fetchRunDiff,
  fetchRuns,
  openEvalSocket,
  patchRunComment,
  type EvalConfig,
} from "../lib/api";
import { modelColor } from "../lib/modelColors";
import type {
  AggregateMetrics,
  DiffResponse,
  EvalEvent,
  MetricKey,
  ModelName,
  QueryItem,
  QueryStats,
  QuestionType,
  RunDetail,
  RunSummary,
} from "../lib/types";

interface ModelCategory {
  name: string;
  // Tailwind color classes for header / chips. Each category gets a distinct hue.
  header: string;       // text color for category title
  dot: string;          // category dot color
  chipIdle: string;     // unselected chip
  chipActive: string;   // selected chip
  models: ModelName[];
}

const MODEL_CATEGORIES: ModelCategory[] = [
  {
    name: "First-stage",
    header: "text-slate-700",
    dot: "bg-slate-500",
    chipIdle: "border-slate-300 text-slate-700 hover:bg-slate-50",
    chipActive: "bg-slate-600 border-slate-600 text-white",
    models: ["bm25"],
  },
  {
    name: "Pointwise (monoT5)",
    header: "text-blue-700",
    dot: "bg-blue-500",
    chipIdle: "border-blue-300 text-blue-700 hover:bg-blue-50",
    chipActive: "bg-blue-600 border-blue-600 text-white",
    models: ["monot5"],
  },
  {
    name: "Pairwise (duoT5)",
    header: "text-cyan-700",
    dot: "bg-cyan-500",
    chipIdle: "border-cyan-300 text-cyan-700 hover:bg-cyan-50",
    chipActive: "bg-cyan-600 border-cyan-600 text-white",
    models: ["duot5", "duot5_rolling"],
  },
  {
    name: "Listwise (LiT5)",
    header: "text-purple-700",
    dot: "bg-purple-500",
    chipIdle: "border-purple-300 text-purple-700 hover:bg-purple-50",
    chipActive: "bg-purple-600 border-purple-600 text-white",
    models: ["lit5"],
  },
  {
    name: "Cross-encoder (BioBERT)",
    header: "text-amber-700",
    dot: "bg-amber-500",
    chipIdle: "border-amber-300 text-amber-700 hover:bg-amber-50",
    chipActive: "bg-amber-600 border-amber-600 text-white",
    models: ["biobert", "bm25_biobert"],
  },
  {
    name: "Cascades",
    header: "text-emerald-700",
    dot: "bg-emerald-500",
    chipIdle: "border-emerald-300 text-emerald-700 hover:bg-emerald-50",
    chipActive: "bg-emerald-600 border-emerald-600 text-white",
    models: ["mono_duo", "mono_entropy_h50_duo"],
  },
  {
    name: "LiT5 fine-tuned",
    header: "text-pink-700",
    dot: "bg-pink-500",
    chipIdle: "border-pink-300 text-pink-700 hover:bg-pink-50",
    chipActive: "bg-pink-600 border-pink-600 text-white",
    models: [
      "lit5_finetuned",
      "lit5_bioasq_lora",
      "lit5_bioasq_lora_e1",
      "lit5_bioasq_lora_e3",
      "lit5_bioasq_lora_kaggle",
      "lit5_bioasq_lora_kaggle_e1",
      "lit5_bioasq_lora_kaggle_e2",
      "lit5_bioasq_lora_kaggle_e3",
      "lit5_bioasq_lora_kaggle_e4",
    ],
  },
  {
    name: "Qwen3 4B",
    header: "text-orange-700",
    dot: "bg-orange-500",
    chipIdle: "border-orange-300 text-orange-700 hover:bg-orange-50",
    chipActive: "bg-orange-600 border-orange-600 text-white",
    models: [
      "qwen3_reranker_4b",
      "qwen4b_linear_fusion",
      "qwen4b_linear_fusion_dynamic",
      "qwen4b_linear_fusion_dynamic_10",
      "qwen4b_linear_fusion_dynamic_gated",
    ],
  },
  {
    name: "Qwen3 0.6B",
    header: "text-rose-700",
    dot: "bg-rose-500",
    chipIdle: "border-rose-300 text-rose-700 hover:bg-rose-50",
    chipActive: "bg-rose-600 border-rose-600 text-white",
    models: [
      "qwen3_reranker_0_6b",
      "qwen06b_lf",
      "qwen06b_lf_999",
      "qwen06b_lf_999_lit5",
      "qwen06b_lf_999_duot5_unc_lit5",
      "qwen06b_lf_lit5",
      "qwen06b_lf_duot5_unc_lit5",
    ],
  },
];

const MODELS: ModelName[] = MODEL_CATEGORIES.flatMap((c) => c.models);

// Loose-typed so legacy run records (with model names no longer in ModelName)
// still resolve to a label without a TypeScript error.
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
const QTYPES: QuestionType[] = ["factoid", "yesno", "list", "summary"];

interface ModelProgress {
  current: number;
  total: number;
  done: boolean;
  elapsed_s?: number;
}

const RANKING_ROWS: { key: keyof AggregateMetrics; label: string; ks: number[] }[] = [
  { key: "ndcg_at", label: "nDCG", ks: [1, 5, 10, 20] },
  { key: "p_at", label: "P", ks: [1, 5, 10, 20] },
  { key: "r_at", label: "R", ks: [1, 5, 10, 20] },
  { key: "mrr_at", label: "MRR", ks: [1, 5, 10, 20] },
  { key: "map_at", label: "MAP", ks: [1, 5, 10, 20] },
];

const fmtPct = (v: number) => v.toFixed(4);
const fmtDelta = (d: number) => {
  const sign = d >= 0 ? "+" : "−";
  return `${sign}${Math.abs(d).toFixed(4)}`;
};

/** Headroom Utilization = (model − bm25) / (oracle − bm25) × 100.
 *  Returns null when oracle === bm25 (no room to gain) or bm25 is unavailable. */
function headroomPct(
  model: number,
  bm25: number,
  oracle: number,
): number | null {
  const denom = oracle - bm25;
  if (denom <= 0) return null;
  return ((model - bm25) / denom) * 100;
}

function fmtHeadroom(v: number | null): string {
  if (v === null) return "—";
  return `${v.toFixed(1)}%`;
}

export default function DashboardPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [runA, setRunA] = useState<RunDetail | null>(null);
  const [runB, setRunB] = useState<RunDetail | null>(null);
  const [bm25Run, setBm25Run] = useState<RunDetail | null>(null);
  const [queries, setQueries] = useState<Record<string, QueryItem>>({});
  const [showConfig, setShowConfig] = useState(false);
  const [cfg, setCfg] = useState<EvalConfig>({
    models: ["bm25", "monot5"],
    n_questions: null,
    comment: "",
    bm25_inject_mode: "off",
  });
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<Partial<Record<ModelName, ModelProgress>>>({});
  const [errorLog, setErrorLog] = useState<string[]>([]);
  const [diff, setDiff] = useState<DiffResponse | null>(null);
  const [queryStats, setQueryStats] = useState<QueryStats | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const refreshRuns = async () => {
    const r = await fetchRuns();
    setRuns(r);
    // Keep the latest BM25 run detail cached as the headroom baseline.
    const latestBm25 = r
      .filter((s) => s.model === "bm25")
      .sort((a, b) => b.started_at - a.started_at)[0];
    if (latestBm25) {
      fetchRun(latestBm25.run_id)
        .then(setBm25Run)
        .catch((e) => console.error("bm25 fetch:", e));
    }
  };

  useEffect(() => {
    refreshRuns().catch((e) => console.error(e));
    fetchQueries()
      .then((qs) => {
        const m: Record<string, QueryItem> = {};
        for (const q of qs) m[q.id] = q;
        setQueries(m);
      })
      .catch((e) => console.error("queries:", e));
    fetchQueryStats()
      .then(setQueryStats)
      .catch((e) => console.error("query stats:", e));
  }, []);

  const onEvent = (e: EvalEvent) => {
    if (e.type === "progress") {
      setProgress((p) => ({
        ...p,
        [e.model]: {
          current: e.current,
          total: e.total,
          done: p[e.model]?.done ?? false,
        },
      }));
    } else if (e.type === "model_done") {
      setProgress((p) => ({
        ...p,
        [e.model]: {
          current: p[e.model]?.total ?? 0,
          total: p[e.model]?.total ?? 0,
          done: true,
          elapsed_s: e.elapsed_s,
        },
      }));
      refreshRuns().catch((err) => console.error(err));
    } else if (e.type === "done") {
      setRunning(false);
      wsRef.current = null;
      refreshRuns().catch((err) => console.error(err));
    } else if (e.type === "error") {
      setErrorLog((es) => [...es, `${e.model ?? ""} ${e.qid ?? ""}: ${e.message}`]);
    }
  };

  const handleRun = () => {
    if (running || cfg.models.length === 0) return;
    setRunning(true);
    setProgress({});
    setErrorLog([]);
    setShowConfig(false);
    const ws = openEvalSocket(cfg, onEvent);
    ws.onclose = () => {
      setRunning(false);
      wsRef.current = null;
    };
    ws.onerror = () => {
      setErrorLog((es) => [...es, "WebSocket error"]);
      setRunning(false);
      wsRef.current = null;
    };
    wsRef.current = ws;
  };

  const handleCancel = () => {
    wsRef.current?.close();
    wsRef.current = null;
    setRunning(false);
  };

  const toggleModel = (m: ModelName) => {
    setCfg((c) => ({
      ...c,
      models: c.models.includes(m) ? c.models.filter((x) => x !== m) : [...c.models, m],
    }));
  };

  const setN = (n: number | null) => setCfg((c) => ({ ...c, n_questions: n }));
  const setComment = (s: string) => setCfg((c) => ({ ...c, comment: s }));
  const setBm25InjectMode = (m: EvalConfig["bm25_inject_mode"]) =>
    setCfg((c) => ({ ...c, bm25_inject_mode: m }));

  const pickA = async (id: string) => {
    if (runA?.run_id === id) {
      setRunA(null);
      setDiff(null);
      return;
    }
    try {
      setRunA(await fetchRun(id));
    } catch (e) {
      console.error(e);
    }
  };

  const pickB = async (id: string) => {
    if (runB?.run_id === id) {
      setRunB(null);
      setDiff(null);
      return;
    }
    try {
      setRunB(await fetchRun(id));
    } catch (e) {
      console.error(e);
    }
  };

  // When both runs are picked, fetch the per-qtype + global diff (A vs baseline B).
  useEffect(() => {
    if (runA && runB) {
      fetchRunDiff(runA.run_id, runB.run_id)
        .then(setDiff)
        .catch((e) => {
          console.error(e);
          setDiff(null);
        });
    } else {
      setDiff(null);
    }
  }, [runA?.run_id, runB?.run_id]);

  const updateComment = async (runId: string, comment: string) => {
    try {
      await patchRunComment(runId, comment);
      setRuns((rs) => rs.map((r) => (r.run_id === runId ? { ...r, comment } : r)));
      if (runA?.run_id === runId) setRunA({ ...runA, comment });
      if (runB?.run_id === runId) setRunB({ ...runB, comment });
    } catch (e) {
      console.error(e);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm(`Delete retrieval run ${id}?`)) return;
    try {
      await deleteRun(id);
      if (runA?.run_id === id) setRunA(null);
      if (runB?.run_id === id) setRunB(null);
      await refreshRuns();
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="h-full flex">
      <RunsSidebar
        runs={runs}
        runA={runA}
        runB={runB}
        onPickA={pickA}
        onPickB={pickB}
        onDelete={handleDelete}
        onNewRun={() => setShowConfig((s) => !s)}
        showConfig={showConfig}
        onUpdateComment={updateComment}
      />
      <div className="flex-1 overflow-y-auto px-8 py-6">
        <div className="max-w-6xl mx-auto space-y-6">
          <div className="flex items-baseline justify-between">
            <h1 className="text-xl font-semibold">Retrieval evaluation</h1>
            <span className="text-xs text-muted">
              Saves top-20 passages per query for downstream generation runs
            </span>
          </div>
          {queryStats && (
            <div className="text-[11px] text-muted border border-border rounded-md px-3 py-1.5 bg-bg/40">
              Metrics computed over <span className="font-mono font-semibold text-ink">{queryStats.evaluated}</span> queries
              {queryStats.total !== queryStats.evaluated && (
                <> (of {queryStats.total} total — {queryStats.total - queryStats.with_qrels} lack qrels)</>
              )}
            </div>
          )}

          {showConfig && (
            <ConfigCard
              cfg={cfg}
              toggleModel={toggleModel}
              setN={setN}
              setComment={setComment}
              setBm25InjectMode={setBm25InjectMode}
              running={running}
              onRun={handleRun}
              onCancel={handleCancel}
              onClose={() => setShowConfig(false)}
            />
          )}

          {(running || Object.keys(progress).length > 0) && (
            <ProgressCard models={cfg.models} progress={progress} />
          )}

          {runA && runB && (
            <PairwiseCompare a={runA} b={runB} diff={diff} bm25={bm25Run} queries={queries} onUpdateComment={updateComment} />
          )}
          {runA && !runB && <SingleRunView run={runA} bm25={bm25Run} queries={queries} onUpdateComment={updateComment} />}
          {!runA && runB && <SingleRunView run={runB} bm25={bm25Run} queries={queries} onUpdateComment={updateComment} />}
          {!runA && !runB && !running && (
            <EmptyState onNewRun={() => setShowConfig(true)} hasRuns={runs.length > 0} />
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

function RunsSidebar({
  runs,
  runA,
  runB,
  onPickA,
  onPickB,
  onDelete,
  onNewRun,
  showConfig,
  onUpdateComment,
}: {
  runs: RunSummary[];
  runA: RunDetail | null;
  runB: RunDetail | null;
  onPickA: (id: string) => void;
  onPickB: (id: string) => void;
  onDelete: (id: string) => void;
  onNewRun: () => void;
  showConfig: boolean;
  onUpdateComment: (id: string, comment: string) => void;
}) {
  const grouped = useMemo(() => {
    const m: Record<string, RunSummary[]> = {};
    for (const r of runs) {
      m[r.model] = m[r.model] || [];
      m[r.model].push(r);
    }
    return m;
  }, [runs]);

  // Order known models first, then any extra models (e.g. deepseek) alphabetically.
  const orderedModels = useMemo(() => {
    const all = Array.from(new Set([...MODELS, ...Object.keys(grouped)]));
    return all as ModelName[];
  }, [grouped]);

  const fmtTs = (ts: number) =>
    new Date(ts * 1000).toLocaleString(undefined, {
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });

  return (
    <aside className="w-80 shrink-0 h-full flex flex-col border-r border-border bg-panel">
      <div className="p-3 border-b border-border space-y-2">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-muted">
          Retrieval runs ({runs.length})
        </h2>
        <button
          className={`btn w-full ${showConfig ? "btn-primary" : ""}`}
          onClick={onNewRun}
        >
          {showConfig ? "Hide config" : "+ New retrieval run"}
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        {orderedModels.map((m) => {
          const list = grouped[m];
          if (!list || list.length === 0) return null;
          const c = modelColor(m);
          return (
            <div key={m}>
              <div className={`px-3 py-1.5 text-xs font-semibold uppercase bg-bg/40 border-b border-border flex items-center gap-2 ${c.text}`}>
                <span className={`inline-block w-2 h-2 rounded-full ${c.dot}`} />
                {MODEL_LABEL[m] ?? m}
              </div>
              {list.map((r) => {
                const isA = runA?.run_id === r.run_id;
                const isB = runB?.run_id === r.run_id;
                return (
                  <div
                    key={r.run_id}
                    className={`px-3 py-2 border-b border-border/50 text-sm flex gap-2 ${
                      isA || isB ? "bg-bg" : ""
                    }`}
                  >
                    <span className={`w-1 self-stretch rounded-full ${c.bar}`} aria-hidden />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <button
                          className={`chip text-[10px] ${isA ? "chip-active" : ""}`}
                          onClick={() => onPickA(r.run_id)}
                        >
                          A
                        </button>
                        <button
                          className={`chip text-[10px] ${isB ? "chip-active" : ""}`}
                          onClick={() => onPickB(r.run_id)}
                        >
                          B
                        </button>
                        <span className="text-xs flex-1 leading-tight">
                          {fmtTs(r.ended_at)}
                        </span>
                        <button
                          className="text-xs text-muted hover:text-accent"
                          onClick={() => onDelete(r.run_id)}
                        >
                          ×
                        </button>
                      </div>
                      <div className="text-[11px] text-muted mt-0.5 leading-tight">
                        {r.elapsed_s.toFixed(1)}s · {r.n_queries} q · top-{r.config?.save_topn ?? 20}
                      </div>
                      <CommentInline
                        value={r.comment ?? ""}
                        onSave={(v) => onUpdateComment(r.run_id, v)}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          );
        })}
        {runs.length === 0 && (
          <div className="p-4 text-sm text-muted">
            No retrieval runs yet. Click "+ New retrieval run" to start.
          </div>
        )}
      </div>
    </aside>
  );
}

function ConfigCard({
  cfg,
  toggleModel,
  setN,
  setComment,
  setBm25InjectMode,
  running,
  onRun,
  onCancel,
  onClose,
}: {
  cfg: EvalConfig;
  toggleModel: (m: ModelName) => void;
  setN: (n: number | null) => void;
  setComment: (s: string) => void;
  setBm25InjectMode: (m: EvalConfig["bm25_inject_mode"]) => void;
  running: boolean;
  onRun: () => void;
  onCancel: () => void;
  onClose: () => void;
}) {
  return (
    <section className="border border-border rounded-md bg-panel p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium">New retrieval run</h2>
        <button className="text-xs text-muted hover:text-ink" onClick={onClose}>
          ×
        </button>
      </div>
      <div className="space-y-2">
        {MODEL_CATEGORIES.map((cat) => {
          const allSelected = cat.models.every((m) => cfg.models.includes(m));
          const someSelected = cat.models.some((m) => cfg.models.includes(m));
          const toggleAll = () => {
            // If all are selected, clear them; otherwise select any missing.
            cat.models.forEach((m) => {
              const has = cfg.models.includes(m);
              if (allSelected && has) toggleModel(m);
              else if (!allSelected && !has) toggleModel(m);
            });
          };
          return (
            <div key={cat.name} className="flex flex-wrap items-center gap-1.5">
              <button
                type="button"
                onClick={toggleAll}
                disabled={running}
                className={`inline-flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider mr-1 ${cat.header} hover:opacity-80 disabled:opacity-60`}
                title={allSelected ? "Deselect all" : "Select all"}
              >
                <span className={`inline-block w-2 h-2 rounded-full ${cat.dot}`} />
                {cat.name}
                <span className="text-muted font-normal normal-case">
                  ({cat.models.filter((m) => cfg.models.includes(m)).length}/{cat.models.length})
                </span>
              </button>
              {cat.models.map((m) => {
                const checked = cfg.models.includes(m);
                return (
                  <button
                    key={m}
                    type="button"
                    onClick={() => toggleModel(m)}
                    disabled={running}
                    title={MODEL_LABEL[m]}
                    className={`text-xs px-2 py-1 rounded-md border transition-colors disabled:opacity-60 disabled:cursor-not-allowed ${
                      checked ? cat.chipActive : `bg-bg ${cat.chipIdle}`
                    }`}
                  >
                    {MODEL_LABEL[m] ?? m}
                  </button>
                );
              })}
              {!allSelected && someSelected ? null : null}
            </div>
          );
        })}
      </div>
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
        <label className="flex items-center gap-2 text-xs text-muted">
          # questions (blank = all, seed=42)
          <input
            type="number"
            min={1}
            className="w-24 border border-border rounded-md bg-bg px-2 py-1 text-sm text-ink"
            value={cfg.n_questions ?? ""}
            onChange={(e) => {
              const v = e.target.value;
              setN(v === "" ? null : Math.max(1, parseInt(v, 10)));
            }}
            disabled={running}
            placeholder="all"
          />
        </label>
        <label className="flex items-center gap-2 text-xs text-muted">
          BM25 inject
          <select
            className="border border-border rounded-md bg-bg px-2 py-1 text-sm text-ink"
            value={cfg.bm25_inject_mode ?? "off"}
            onChange={(e) =>
              setBm25InjectMode(e.target.value as EvalConfig["bm25_inject_mode"])
            }
            disabled={running}
            title="Prepend BM25 prior to each candidate's text before reranking"
          >
            <option value="off">off</option>
            <option value="raw">raw (e.g. 12.34)</option>
            <option value="norm">normalized (0-1)</option>
            <option value="bucket">bucket (low/med/high)</option>
          </select>
        </label>
        <label className="flex items-center gap-2 text-xs text-muted flex-1 min-w-[260px]">
          comment
          <input
            type="text"
            className="flex-1 border border-border rounded-md bg-bg px-2 py-1 text-sm text-ink"
            value={cfg.comment ?? ""}
            onChange={(e) => setComment(e.target.value)}
            disabled={running}
            placeholder="optional — what is this run for?"
          />
        </label>
        <div className="ml-auto">
          {running ? (
            <button className="btn" onClick={onCancel}>
              Cancel
            </button>
          ) : (
            <button
              className="btn btn-primary"
              onClick={onRun}
              disabled={cfg.models.length === 0}
            >
              Run
            </button>
          )}
        </div>
      </div>
      <p className="text-xs text-muted">
        BM25 first stage retrieves top 50 (configured in <code>config.py</code>); each run saves top-20 passages per query for downstream generation. The comment is editable later.
      </p>
    </section>
  );
}

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
        className={`block w-full text-left text-[11px] italic mt-1 leading-tight hover:text-ink ${value ? "text-amber-700" : "text-muted"}`}
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
      className="block w-full mt-1 px-1.5 py-0.5 text-[11px] border border-border rounded bg-bg"
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

function ProgressCard({
  models,
  progress,
}: {
  models: ModelName[];
  progress: Partial<Record<ModelName, ModelProgress>>;
}) {
  return (
    <section className="border border-border rounded-md bg-panel p-4 space-y-3">
      <h2 className="text-sm font-medium">Progress</h2>
      <div className="space-y-2">
        {models.map((m) => {
          const p = progress[m];
          if (!p) {
            return (
              <div key={m} className="flex items-center gap-3 text-sm text-muted">
                <span className="w-44 font-mono">{MODEL_LABEL[m]}</span>
                <span className="text-xs">queued</span>
              </div>
            );
          }
          const pct = p.total > 0 ? Math.round((p.current / p.total) * 100) : 0;
          return (
            <div key={m} className="flex items-center gap-3 text-sm">
              <span className="w-44 font-mono">{MODEL_LABEL[m]}</span>
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

function EmptyState({ onNewRun, hasRuns }: { onNewRun: () => void; hasRuns: boolean }) {
  return (
    <div className="border border-dashed border-border rounded-md bg-panel p-12 text-center">
      <h2 className="text-lg font-medium mb-2">
        {hasRuns ? "Pick a run from the sidebar" : "No runs yet"}
      </h2>
      <p className="text-sm text-muted mb-4">
        {hasRuns
          ? "Click A on a run to view it. Click B on another run to compare them side-by-side. Use the Generation page to feed these runs to the LLM."
          : "Start a new retrieval evaluation. Generation runs (LLM + QA) live on the Generation page."}
      </p>
      {!hasRuns && (
        <button className="btn btn-primary" onClick={onNewRun}>
          + New retrieval run
        </button>
      )}
    </div>
  );
}

function SingleRunView({
  run,
  bm25,
  queries,
  onUpdateComment,
}: {
  run: RunDetail;
  bm25: RunDetail | null;
  queries: Record<string, QueryItem>;
  onUpdateComment: (id: string, v: string) => void;
}) {
  return (
    <div className="space-y-6">
      <RunHeader run={run} onUpdateComment={onUpdateComment} />
      <SingleRankingTable run={run} bm25={bm25} />
      <SinglePerQtypeMetricsTable run={run} queries={queries} />
      <SinglePerQueryTable run={run} queries={queries} />
    </div>
  );
}

function SinglePerQtypeMetricsTable({
  run,
  queries,
}: {
  run: RunDetail;
  queries: Record<string, QueryItem>;
}) {
  const ks = [1, 5, 10, 20];

  const byQtype = useMemo(() => {
    const sums: Partial<
      Record<QuestionType, { metrics: AggregateMetrics; n: number }>
    > = {};
    const ensure = (qt: QuestionType) => {
      if (!sums[qt]) {
        sums[qt] = {
          metrics: {
            ndcg_at: {},
            mrr_at: {},
            p_at: {},
            r_at: {},
            map_at: {},
          } as AggregateMetrics,
          n: 0,
        };
      }
      return sums[qt]!;
    };
    for (const [qid, entry] of Object.entries(run.per_query)) {
      const qt = (entry.qtype ?? queries[qid]?.type ?? null) as QuestionType | null;
      if (!qt) continue;
      const acc = ensure(qt);
      acc.n += 1;
      for (const row of RANKING_ROWS) {
        const bucket = (acc.metrics[row.key] ??= {});
        for (const k of ks) {
          const v = entry.metrics?.[row.key]?.[k] ?? 0;
          bucket[k] = (bucket[k] ?? 0) + v;
        }
      }
    }
    const avg: Partial<Record<QuestionType, { metrics: AggregateMetrics; n: number }>> = {};
    for (const qt of QTYPES) {
      const s = sums[qt];
      if (!s || s.n === 0) continue;
      const m: AggregateMetrics = {
        ndcg_at: {},
        mrr_at: {},
        p_at: {},
        r_at: {},
        map_at: {},
      } as AggregateMetrics;
      for (const row of RANKING_ROWS) {
        const out: Record<number, number> = {};
        for (const k of ks) {
          out[k] = (s.metrics[row.key]?.[k] ?? 0) / s.n;
        }
        m[row.key] = out;
      }
      avg[qt] = { metrics: m, n: s.n };
    }
    return avg;
  }, [run, queries]);

  const presentQtypes = QTYPES.filter((qt) => byQtype[qt]);
  if (presentQtypes.length === 0) return null;

  return (
    <section className="border border-border rounded-md bg-panel overflow-hidden">
      <div className="px-3 py-2 text-xs font-medium text-muted border-b border-border">
        Per-question-type retrieval metrics
        <span className="ml-2 text-[10px] text-muted">(mean of per-query values within each qtype)</span>
      </div>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-muted bg-bg/40">
            <th className="px-3 py-1.5 text-left font-medium">qtype</th>
            <th className="px-3 py-1.5 text-left font-medium">metric</th>
            {ks.map((k) => (
              <th key={k} className="px-3 py-1.5 text-right font-medium">@{k}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {presentQtypes.flatMap((qt) => {
            const block = byQtype[qt]!;
            return RANKING_ROWS.map((row, ri) => (
              <tr
                key={`${qt}-${row.label}`}
                className={`border-t ${ri === 0 ? "border-border" : "border-border/30"}`}
              >
                <td className="px-3 py-1.5">
                  {ri === 0 && (
                    <span
                      className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-medium ${QTYPE_BADGE[qt]}`}
                    >
                      {qt} ({block.n})
                    </span>
                  )}
                </td>
                <td className="px-3 py-1.5 font-medium">{row.label}</td>
                {ks.map((k) => (
                  <td
                    key={k}
                    className="px-3 py-1.5 text-right font-mono tabular-nums"
                  >
                    {fmtPct(block.metrics[row.key]?.[k] ?? 0)}
                  </td>
                ))}
              </tr>
            ));
          })}
        </tbody>
      </table>
    </section>
  );
}

function RunHeader({
  run,
  onUpdateComment,
}: {
  run: RunDetail;
  onUpdateComment: (id: string, v: string) => void;
}) {
  const c = modelColor(run.model);
  return (
    <div className="border border-border rounded-md bg-panel px-4 py-3 space-y-1.5">
      <div className="flex items-baseline justify-between">
        <div className="flex items-center gap-2">
          <span className={`inline-block w-2.5 h-2.5 rounded-full ${c.dot}`} />
          <span className={`font-mono font-medium ${c.text}`}>
            {MODEL_LABEL[run.model] ?? run.model}
          </span>
          <span className="text-xs text-muted ml-2">{run.run_id}</span>
        </div>
        <span className="text-xs text-muted">
          {new Date(run.ended_at * 1000).toLocaleString()} · {run.elapsed_s.toFixed(1)}s ·{" "}
          {Object.keys(run.per_query).length} queries
        </span>
      </div>
      <CommentInline
        value={run.comment ?? ""}
        onSave={(v) => onUpdateComment(run.run_id, v)}
      />
    </div>
  );
}

function SingleRankingTable({ run, bm25 }: { run: RunDetail; bm25: RunDetail | null }) {
  const ks = [1, 5, 10, 20];
  const isBm25 = run.model === "bm25";
  return (
    <section className="border border-border rounded-md bg-panel overflow-hidden">
      <div className="px-3 py-2 text-xs font-medium text-muted border-b border-border flex items-center justify-between">
        <span>Retrieval metrics</span>
        {!isBm25 && bm25 && (
          <span className="text-[10px] text-muted">
            Headroom = (model − BM25) / (oracle − BM25) × 100 · oracle = 1.0
          </span>
        )}
      </div>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-muted">
            <th className="px-3 py-1.5 text-left font-medium">Metric</th>
            {ks.map((k) => (
              <th key={k} className="px-3 py-1.5 text-right font-medium">@{k}</th>
            ))}
            {!isBm25 && bm25 && ks.map((k) => (
              <th key={`hr${k}`} className="px-3 py-1.5 text-right font-medium text-violet-600">
                Headroom@{k}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {RANKING_ROWS.map((row) => (
            <tr key={row.label} className="border-t border-border/50">
              <td className="px-3 py-1.5 font-medium">{row.label}</td>
              {ks.map((k) => (
                <td key={k} className="px-3 py-1.5 text-right font-mono tabular-nums">
                  {fmtPct(run.aggregate[row.key]?.[k] ?? 0)}
                </td>
              ))}
              {!isBm25 && bm25 && ks.map((k) => {
                const model  = run.aggregate[row.key]?.[k] ?? 0;
                const base   = bm25.aggregate[row.key]?.[k] ?? 0;
                const hr = headroomPct(model, base, 1.0);
                return (
                  <td key={`hr${k}`}
                    className="px-3 py-1.5 text-right font-mono tabular-nums text-violet-600">
                    {fmtHeadroom(hr)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function PairwiseCompare({
  a,
  b,
  diff,
  bm25,
  queries,
  onUpdateComment,
}: {
  a: RunDetail;
  b: RunDetail;
  diff: DiffResponse | null;
  bm25: RunDetail | null;
  queries: Record<string, QueryItem>;
  onUpdateComment: (id: string, v: string) => void;
}) {
  const cA = modelColor(a.model);
  const cB = modelColor(b.model);
  return (
    <div className="space-y-6">
      <div className="border border-border rounded-md bg-panel px-4 py-3 space-y-1.5">
        <div className="flex items-baseline flex-wrap gap-x-2">
          <span className="text-xs text-muted mr-1">A:</span>
          <span className={`inline-block w-2 h-2 rounded-full ${cA.dot}`} />
          <span className={`font-mono font-medium ${cA.text}`}>{MODEL_LABEL[a.model] ?? a.model}</span>
          <span className="text-xs text-muted ml-1">{a.run_id}</span>
          <span className="mx-2 text-muted">vs (baseline)</span>
          <span className="text-xs text-muted mr-1">B:</span>
          <span className={`inline-block w-2 h-2 rounded-full ${cB.dot}`} />
          <span className={`font-mono font-medium ${cB.text}`}>{MODEL_LABEL[b.model] ?? b.model}</span>
          <span className="text-xs text-muted ml-1">{b.run_id}</span>
          {diff && (
            <span className="ml-auto text-xs text-muted">
              {diff.n_overlapping} overlapping queries
            </span>
          )}
        </div>
        <div className="grid grid-cols-2 gap-x-6 gap-y-1">
          <div>
            <span className="text-[11px] text-muted">A comment:</span>
            <CommentInline
              value={a.comment ?? ""}
              onSave={(v) => onUpdateComment(a.run_id, v)}
            />
          </div>
          <div>
            <span className="text-[11px] text-muted">B comment:</span>
            <CommentInline
              value={b.comment ?? ""}
              onSave={(v) => onUpdateComment(b.run_id, v)}
            />
          </div>
        </div>
      </div>
      <PairwiseRankingTable a={a} b={b} bm25={bm25} />
      {diff && <PerQtypeDiff a={a} b={b} diff={diff} />}
      <PerQueryDiff a={a} b={b} queries={queries} />
    </div>
  );
}

function PerQtypeDiff({
  a,
  b,
  diff,
}: {
  a: RunDetail;
  b: RunDetail;
  diff: DiffResponse;
}) {
  const ks = [1, 5, 10, 20];
  const presentQtypes = QTYPES.filter((qt) => diff.by_qtype[qt]);
  if (presentQtypes.length === 0) return null;
  return (
    <section className="border border-border rounded-md bg-panel overflow-hidden">
      <div className="px-3 py-2 text-xs font-medium text-muted border-b border-border">
        Per-question-type Δ — {MODEL_LABEL[a.model] ?? a.model} vs baseline {MODEL_LABEL[b.model] ?? b.model}
        <span className="ml-2 text-[10px] text-muted">(green = baseline higher, red = A higher)</span>
      </div>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-muted bg-bg/40">
            <th className="px-3 py-1.5 text-left font-medium">qtype</th>
            <th className="px-3 py-1.5 text-left font-medium">metric</th>
            {ks.map((k) => (
              <th key={k} className="px-3 py-1.5 text-right font-medium">
                Δ@{k}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {presentQtypes.flatMap((qt) => {
            const block = diff.by_qtype[qt]!;
            return RANKING_ROWS.map((row) => (
              <tr key={`${qt}-${row.label}`} className="border-t border-border/50">
                <td className="px-3 py-1.5 font-mono text-muted">{qt}</td>
                <td className="px-3 py-1.5 font-medium">{row.label}</td>
                {ks.map((k) => {
                  const raw = block.delta[row.key]?.[k];
                  const d = raw !== undefined ? -raw : undefined;
                  if (d === undefined) {
                    return (
                      <td key={k} className="px-3 py-1.5 text-right text-muted">
                        –
                      </td>
                    );
                  }
                  return (
                    <td
                      key={k}
                      className={`px-3 py-1.5 text-right font-mono tabular-nums ${
                        d > 0
                          ? "text-emerald-700"
                          : d < 0
                          ? "text-accent"
                          : "text-muted"
                      }`}
                    >
                      {fmtDelta(d)}
                    </td>
                  );
                })}
              </tr>
            ));
          })}
        </tbody>
      </table>
    </section>
  );
}

function PairwiseRankingTable({
  a,
  b,
  bm25,
}: {
  a: RunDetail;
  b: RunDetail;
  bm25: RunDetail | null;
}) {
  return (
    <section className="border border-border rounded-md bg-panel overflow-hidden">
      <div className="px-3 py-2 text-xs font-medium text-muted border-b border-border flex items-center justify-between">
        <span>Retrieval metrics — A vs B</span>
        <span className="text-[10px] text-muted">
          Headroom = (B − A) / (1 − A) × 100
        </span>
      </div>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-muted">
            <th className="px-3 py-1.5 text-left font-medium">metric</th>
            <th className="px-3 py-1.5 text-right font-medium">A = {MODEL_LABEL[a.model] ?? a.model}</th>
            <th className="px-3 py-1.5 text-right font-medium">B = {MODEL_LABEL[b.model] ?? b.model}</th>
            <th className="px-3 py-1.5 text-right font-medium">Δ (B−A)</th>
            <th className="px-3 py-1.5 text-right font-medium text-violet-600">Headroom B%</th>
          </tr>
        </thead>
        <tbody>
          {RANKING_ROWS.flatMap((row) =>
            row.ks.map((k) => {
              const av = a.aggregate[row.key]?.[k] ?? 0;
              const bv = b.aggregate[row.key]?.[k] ?? 0;
              const d  = bv - av;
              const hr = headroomPct(bv, av, 1.0);
              return (
                <tr key={`${row.label}@${k}`} className="border-t border-border/50">
                  <td className="px-3 py-1.5 font-mono">{row.label}@{k}</td>
                  <td className="px-3 py-1.5 text-right font-mono tabular-nums">{fmtPct(av)}</td>
                  <td className="px-3 py-1.5 text-right font-mono tabular-nums">{fmtPct(bv)}</td>
                  <td className={`px-3 py-1.5 text-right font-mono tabular-nums ${
                    d > 0 ? "text-emerald-700" : d < 0 ? "text-accent" : "text-muted"
                  }`}>
                    {fmtDelta(d)}
                  </td>
                  <td className={`px-3 py-1.5 text-right font-mono tabular-nums ${
                    hr !== null && hr >= 80 ? "text-emerald-700"
                    : hr !== null && hr >= 50 ? "text-violet-600"
                    : hr !== null && hr < 0  ? "text-accent"
                    : "text-muted"
                  }`}>
                    {fmtHeadroom(hr)}
                  </td>
                </tr>
              );
            }),
          )}
        </tbody>
      </table>
    </section>
  );
}

const METRIC_OPTS: { key: MetricKey; label: string }[] = [
  { key: "ndcg_at", label: "nDCG" },
  { key: "p_at", label: "P" },
  { key: "r_at", label: "R" },
  { key: "mrr_at", label: "MRR" },
  { key: "map_at", label: "MAP" },
];
const K_OPTS = [1, 5, 10, 20];

const QTYPE_BADGE: Record<QuestionType, string> = {
  factoid: "bg-sky-100 text-sky-700",
  yesno: "bg-emerald-100 text-emerald-700",
  list: "bg-amber-100 text-amber-700",
  summary: "bg-violet-100 text-violet-700",
};

function MetricKControls({
  metric,
  k,
  setMetric,
  setK,
  qtypeFilter,
  setQtypeFilter,
}: {
  metric: MetricKey;
  k: number;
  setMetric: (m: MetricKey) => void;
  setK: (k: number) => void;
  qtypeFilter: QuestionType | "all";
  setQtypeFilter: (q: QuestionType | "all") => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <span className="text-muted">metric</span>
      <select
        className="border border-border rounded bg-bg px-1.5 py-0.5 text-ink"
        value={metric}
        onChange={(e) => setMetric(e.target.value as MetricKey)}
      >
        {METRIC_OPTS.map((m) => (
          <option key={m.key} value={m.key}>{m.label}</option>
        ))}
      </select>
      <span className="text-muted">@</span>
      <select
        className="border border-border rounded bg-bg px-1.5 py-0.5 text-ink"
        value={k}
        onChange={(e) => setK(parseInt(e.target.value, 10))}
      >
        {K_OPTS.map((kk) => (
          <option key={kk} value={kk}>{kk}</option>
        ))}
      </select>
      <span className="text-muted ml-2">qtype</span>
      <select
        className="border border-border rounded bg-bg px-1.5 py-0.5 text-ink"
        value={qtypeFilter}
        onChange={(e) => setQtypeFilter(e.target.value as QuestionType | "all")}
      >
        <option value="all">all</option>
        {QTYPES.map((q) => (
          <option key={q} value={q}>{q}</option>
        ))}
      </select>
    </div>
  );
}

function SinglePerQueryTable({
  run,
  queries,
}: {
  run: RunDetail;
  queries: Record<string, QueryItem>;
}) {
  const [metric, setMetric] = useState<MetricKey>("ndcg_at");
  const [k, setK] = useState<number>(10);
  const [qtypeFilter, setQtypeFilter] = useState<QuestionType | "all">("all");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [limit, setLimit] = useState<number>(50);

  const rows = useMemo(() => {
    const out: { qid: string; qtype: QuestionType | null; text: string; v: number }[] = [];
    for (const [qid, entry] of Object.entries(run.per_query)) {
      const qt = (entry.qtype ?? queries[qid]?.type ?? null) as QuestionType | null;
      if (qtypeFilter !== "all" && qt !== qtypeFilter) continue;
      const v = entry.metrics?.[metric]?.[k] ?? 0;
      out.push({ qid, qtype: qt, text: queries[qid]?.text ?? "", v });
    }
    out.sort((x, y) => (sortDir === "desc" ? y.v - x.v : x.v - y.v));
    return out;
  }, [run, queries, metric, k, qtypeFilter, sortDir]);

  const visible = rows.slice(0, limit);
  return (
    <section className="border border-border rounded-md bg-panel overflow-hidden">
      <div className="px-3 py-2 border-b border-border flex items-center justify-between gap-3 flex-wrap">
        <span className="text-xs font-medium text-muted">
          Per-query metrics ({rows.length} {qtypeFilter === "all" ? "queries" : qtypeFilter})
        </span>
        <MetricKControls
          metric={metric}
          k={k}
          setMetric={setMetric}
          setK={setK}
          qtypeFilter={qtypeFilter}
          setQtypeFilter={setQtypeFilter}
        />
      </div>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-muted bg-bg/40">
            <th className="px-3 py-1.5 text-left font-medium">qid</th>
            <th className="px-3 py-1.5 text-left font-medium">qtype</th>
            <th className="px-3 py-1.5 text-left font-medium">question</th>
            <th
              className="px-3 py-1.5 text-right font-medium cursor-pointer select-none"
              onClick={() => setSortDir((d) => (d === "desc" ? "asc" : "desc"))}
              title="Click to toggle sort direction"
            >
              {METRIC_OPTS.find((m) => m.key === metric)?.label}@{k} {sortDir === "desc" ? "↓" : "↑"}
            </th>
          </tr>
        </thead>
        <tbody>
          {visible.map((r) => (
            <tr key={r.qid} className="border-t border-border/50">
              <td className="px-3 py-1.5 font-mono text-muted">{r.qid}</td>
              <td className="px-3 py-1.5">
                {r.qtype && (
                  <span className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-medium ${QTYPE_BADGE[r.qtype]}`}>
                    {r.qtype}
                  </span>
                )}
              </td>
              <td className="px-3 py-1.5 truncate max-w-md" title={r.text}>{r.text}</td>
              <td className="px-3 py-1.5 text-right font-mono tabular-nums">{r.v.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > visible.length && (
        <div className="px-3 py-2 border-t border-border text-xs text-center">
          <button className="text-muted hover:text-ink" onClick={() => setLimit((n) => n + 100)}>
            Show {Math.min(100, rows.length - visible.length)} more ({rows.length - visible.length} hidden)
          </button>
        </div>
      )}
      {rows.length === 0 && (
        <div className="px-3 py-6 text-center text-xs text-muted">No queries match the filter.</div>
      )}
    </section>
  );
}

function PerQueryDiff({
  a,
  b,
  queries,
}: {
  a: RunDetail;
  b: RunDetail;
  queries: Record<string, QueryItem>;
}) {
  const [metric, setMetric] = useState<MetricKey>("ndcg_at");
  const [k, setK] = useState<number>(10);
  const [qtypeFilter, setQtypeFilter] = useState<QuestionType | "all">("all");
  const [sortMode, setSortMode] = useState<"abs" | "delta_desc" | "delta_asc">("abs");
  const [limit, setLimit] = useState<number>(50);

  const rows = useMemo(() => {
    const out: {
      qid: string;
      qtype: QuestionType | null;
      text: string;
      av: number;
      bv: number;
      d: number;
    }[] = [];
    for (const qid of Object.keys(a.per_query)) {
      const eA = a.per_query[qid];
      const eB = b.per_query[qid];
      if (!eB) continue;
      const qt = (eA.qtype ?? eB.qtype ?? queries[qid]?.type ?? null) as QuestionType | null;
      if (qtypeFilter !== "all" && qt !== qtypeFilter) continue;
      const av = eA.metrics?.[metric]?.[k] ?? 0;
      const bv = eB.metrics?.[metric]?.[k] ?? 0;
      out.push({ qid, qtype: qt, text: queries[qid]?.text ?? "", av, bv, d: av - bv });
    }
    if (sortMode === "abs") out.sort((x, y) => Math.abs(y.d) - Math.abs(x.d));
    else if (sortMode === "delta_desc") out.sort((x, y) => y.d - x.d);
    else out.sort((x, y) => x.d - y.d);
    return out;
  }, [a, b, queries, metric, k, qtypeFilter, sortMode]);

  const visible = rows.slice(0, limit);
  const wins = rows.filter((r) => r.d > 0).length;
  const losses = rows.filter((r) => r.d < 0).length;
  const ties = rows.length - wins - losses;

  return (
    <section className="border border-border rounded-md bg-panel overflow-hidden">
      <div className="px-3 py-2 border-b border-border flex items-center justify-between gap-3 flex-wrap">
        <span className="text-xs font-medium text-muted">
          Per-query Δ ({rows.length} overlapping ·{" "}
          <span className="text-emerald-700">A wins {wins}</span> ·{" "}
          <span className="text-accent">A loses {losses}</span> · ties {ties})
        </span>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <MetricKControls
            metric={metric}
            k={k}
            setMetric={setMetric}
            setK={setK}
            qtypeFilter={qtypeFilter}
            setQtypeFilter={setQtypeFilter}
          />
          <span className="text-muted ml-2">sort</span>
          <select
            className="border border-border rounded bg-bg px-1.5 py-0.5 text-ink"
            value={sortMode}
            onChange={(e) => setSortMode(e.target.value as typeof sortMode)}
          >
            <option value="abs">|Δ| desc</option>
            <option value="delta_desc">Δ desc (A wins)</option>
            <option value="delta_asc">Δ asc (A loses)</option>
          </select>
        </div>
      </div>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-muted bg-bg/40">
            <th className="px-3 py-1.5 text-left font-medium">qid</th>
            <th className="px-3 py-1.5 text-left font-medium">qtype</th>
            <th className="px-3 py-1.5 text-left font-medium">question</th>
            <th className="px-3 py-1.5 text-right font-medium">A</th>
            <th className="px-3 py-1.5 text-right font-medium">B</th>
            <th className="px-3 py-1.5 text-right font-medium">Δ (A−B)</th>
          </tr>
        </thead>
        <tbody>
          {visible.map((r) => (
            <tr key={r.qid} className="border-t border-border/50">
              <td className="px-3 py-1.5 font-mono text-muted">{r.qid}</td>
              <td className="px-3 py-1.5">
                {r.qtype && (
                  <span className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-medium ${QTYPE_BADGE[r.qtype]}`}>
                    {r.qtype}
                  </span>
                )}
              </td>
              <td className="px-3 py-1.5 truncate max-w-md" title={r.text}>{r.text}</td>
              <td className="px-3 py-1.5 text-right font-mono tabular-nums">{r.av.toFixed(4)}</td>
              <td className="px-3 py-1.5 text-right font-mono tabular-nums">{r.bv.toFixed(4)}</td>
              <td className={`px-3 py-1.5 text-right font-mono tabular-nums ${
                r.d > 0 ? "text-emerald-700" : r.d < 0 ? "text-accent" : "text-muted"
              }`}>
                {fmtDelta(r.d)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > visible.length && (
        <div className="px-3 py-2 border-t border-border text-xs text-center">
          <button className="text-muted hover:text-ink" onClick={() => setLimit((n) => n + 100)}>
            Show {Math.min(100, rows.length - visible.length)} more ({rows.length - visible.length} hidden)
          </button>
        </div>
      )}
      {rows.length === 0 && (
        <div className="px-3 py-6 text-center text-xs text-muted">No overlapping queries match the filter.</div>
      )}
    </section>
  );
}
