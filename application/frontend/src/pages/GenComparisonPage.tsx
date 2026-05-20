import { useEffect, useMemo, useState } from "react";
import { fetchGenSummary, fetchQueryStats } from "../lib/api";
import { modelColor } from "../lib/modelColors";
import type {
  GenerationSummary,
  GenerationSummaryCell,
  GenerationSummaryRow,
  QueryStats,
} from "../lib/types";

// ---- metric definitions ---------------------------------------------------

interface MetricDef {
  key: string;
  label: string;
  group: "QA" | "Retrieval" | "Summary extras";
  read: (c: GenerationSummaryCell) => number | null;
}

const METRICS: MetricDef[] = [
  { key: "qa_overall", label: "QA score (overall)", group: "QA", read: (c) => c.qa_overall },
  { key: "qa_factoid", label: "Factoid · MRR", group: "QA", read: (c) => c.qa_by_qtype.factoid ?? null },
  { key: "factoid_strict_acc", label: "Factoid · Strict Acc", group: "QA", read: (c) => c.factoid_strict_acc ?? null },
  { key: "qa_yesno", label: "Yes/No · Accuracy", group: "QA", read: (c) => c.qa_by_qtype.yesno ?? null },
  { key: "qa_list", label: "List · F1", group: "QA", read: (c) => c.qa_by_qtype.list ?? null },
  { key: "list_map", label: "List · MAP", group: "QA", read: (c) => c.list_map ?? null },
  {
    key: "qa_summary",
    label: "Summary · Judge (LLM)",
    group: "QA",
    read: (c) => (c.skip_judge ? null : c.qa_by_qtype.summary ?? null),
  },
  { key: "summary_rouge_l", label: "Summary · ROUGE-L", group: "Summary extras", read: (c) => c.summary_rouge_l ?? null },
  { key: "summary_bert_score", label: "Summary · BERTScore", group: "Summary extras", read: (c) => c.summary_bert_score ?? null },
  { key: "ndcg", label: "nDCG@k", group: "Retrieval", read: (c) => c.retrieval.ndcg ?? null },
  { key: "p", label: "P@k", group: "Retrieval", read: (c) => c.retrieval.p ?? null },
  { key: "r", label: "R@k", group: "Retrieval", read: (c) => c.retrieval.r ?? null },
  { key: "mrr", label: "MRR@k", group: "Retrieval", read: (c) => c.retrieval.mrr ?? null },
  { key: "map", label: "MAP@k", group: "Retrieval", read: (c) => c.retrieval.map ?? null },
];

const PRESET_KS = [1, 3, 5, 10, 20];

const fmt = (v: number | null | undefined) =>
  v == null ? "—" : v.toFixed(4);

const FRIENDLY_MODEL_LABEL: Record<string, string> = {
  bm25: "BM25",
  monot5: "BM25 → monoT5",
  duot5: "BM25 → duoT5",
  duot5_rolling: "BM25 → duoT5 (rolling 20, stride 10)",
  lit5: "BM25 → LiT5",
  mono_duo: "BM25 → monoT5 → duoT5",
  monot5_lit5: "BM25 → monoT5 → LiT5",
  bge_v2_m3: "BM25 → BGE-v2-m3",
  qwen3_reranker_4b: "BM25 → Qwen3-Reranker-4B",
  qwen3_reranker_0_6b: "BM25 → Qwen3-0.6B (pure)",
  rank_zephyr: "BM25 → RankZephyr",
  lit5_finetuned: "BM25 → LiT5 (fine-tuned)",
  lit5_bioasq_lora: "BM25 → LiT5 (LoRA, epoch 2)",
  lit5_bioasq_lora_e1: "BM25 → LiT5 (LoRA q,v, epoch 1)",
  lit5_bioasq_lora_e3: "BM25 → LiT5 (LoRA q,v, epoch 3)",
  lit5_bioasq_lora_kaggle: "BM25 → LiT5 (LoRA q,k,v,o, Kaggle)",
  lit5_bioasq_lora_kaggle_e1: "BM25 → LiT5 (LoRA Kaggle, epoch 1)",
  lit5_bioasq_lora_kaggle_e2: "BM25 → LiT5 (LoRA Kaggle, epoch 2)",
  lit5_bioasq_lora_kaggle_e3: "BM25 → LiT5 (LoRA Kaggle, epoch 3)",
  lit5_bioasq_lora_kaggle_e4: "BM25 → LiT5 (LoRA Kaggle, epoch 4)",
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
  deepseek: "DeepSeek",
};
const labelFor = (m?: string) => (m && (FRIENDLY_MODEL_LABEL[m] ?? m)) || "?";

// One identifiable row in the table = one retrieval run.
function rowDisplay(row: GenerationSummaryRow): string {
  const base = labelFor(row.retrieval_model as string);
  return row.retrieval_comment ? `${base} — ${row.retrieval_comment}` : base;
}

// ---- page ----------------------------------------------------------------

export default function GenComparisonPage() {
  const [summary, setSummary] = useState<GenerationSummary | null>(null);
  const [selectedRowIds, setSelectedRowIds] = useState<string[]>([]);
  const [customKInput, setCustomKInput] = useState<string>("");
  const [customKs, setCustomKs] = useState<number[]>([]);
  const [queryStats, setQueryStats] = useState<QueryStats | null>(null);

  useEffect(() => {
    fetchGenSummary()
      .then((s) => {
        setSummary(s);
        // Pre-select first 2 rows by default.
        setSelectedRowIds(s.rows.slice(0, 2).map((r) => r.retrieval_run_id));
      })
      .catch(console.error);
    fetchQueryStats().then(setQueryStats).catch(console.error);
  }, []);

  const allRows = summary?.rows ?? [];
  const selectedRows = useMemo(
    () => allRows.filter((r) => selectedRowIds.includes(r.retrieval_run_id)),
    [allRows, selectedRowIds],
  );

  // Compute the list of k columns: presets ∪ custom ∪ any k actually present
  // in selected rows. Sorted ascending.
  const ks = useMemo(() => {
    const set = new Set<number>(PRESET_KS);
    for (const k of customKs) set.add(k);
    for (const row of selectedRows) {
      for (const k of Object.keys(row.cells)) {
        const n = Number(k);
        if (Number.isFinite(n)) set.add(n);
      }
    }
    return Array.from(set).sort((a, b) => a - b);
  }, [customKs, selectedRows]);

  const toggleRow = (id: string) =>
    setSelectedRowIds((cur) =>
      cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id],
    );

  const addCustomK = () => {
    const v = Number(customKInput);
    if (!Number.isInteger(v) || v < 1 || v > 100) return;
    if (!customKs.includes(v) && !PRESET_KS.includes(v))
      setCustomKs((cur) => [...cur, v]);
    setCustomKInput("");
  };
  const removeCustomK = (k: number) =>
    setCustomKs((cur) => cur.filter((x) => x !== k));

  return (
    <div className="h-full overflow-y-auto px-8 py-6">
      <div className="max-w-[1400px] mx-auto space-y-6">
        <div>
          <h1 className="text-xl font-semibold">Generation comparison</h1>
          <p className="text-xs text-muted mt-1">
            Pick one or more retrieval runs (models). Rows are metrics, columns
            are k. Bright green = global row max across all (model, k) cells.
            Pale green = local row max within each retrieval model's k-block.
          </p>
          {queryStats && (
            <div className="mt-2 inline-block text-[11px] text-muted border border-border rounded-md px-3 py-1.5 bg-bg/40">
              Metrics computed over <span className="font-mono font-semibold text-ink">{queryStats.evaluated}</span> queries
              {queryStats.total !== queryStats.evaluated && (
                <> (of {queryStats.total} total — {queryStats.total - queryStats.with_qrels} lack qrels and are excluded so QA and retrieval averages share a denominator)</>
              )}
            </div>
          )}
        </div>

        <ModelPicker
          rows={allRows}
          selectedRowIds={selectedRowIds}
          onToggle={toggleRow}
          onClearAll={() => setSelectedRowIds([])}
          onSelectAll={() => setSelectedRowIds(allRows.map((r) => r.retrieval_run_id))}
        />

        <KPicker
          customKs={customKs}
          customKInput={customKInput}
          setCustomKInput={setCustomKInput}
          addCustomK={addCustomK}
          removeCustomK={removeCustomK}
          activeKs={ks}
        />

        {selectedRows.length === 0 ? (
          <div className="border border-dashed border-border rounded-md bg-panel p-8 text-center text-sm text-muted">
            Select at least one model above.
          </div>
        ) : (
          <ComparisonTable rows={selectedRows} ks={ks} />
        )}
      </div>
    </div>
  );
}

// ---- model picker ---------------------------------------------------------

function ModelPicker({
  rows,
  selectedRowIds,
  onToggle,
  onClearAll,
  onSelectAll,
}: {
  rows: GenerationSummaryRow[];
  selectedRowIds: string[];
  onToggle: (id: string) => void;
  onClearAll: () => void;
  onSelectAll: () => void;
}) {
  return (
    <section className="border border-border rounded-md bg-panel p-3 space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted uppercase tracking-wider">
          Models ({selectedRowIds.length} of {rows.length} selected)
        </span>
        <div className="flex items-center gap-2 text-xs">
          <button className="text-muted hover:text-ink" onClick={onSelectAll}>
            select all
          </button>
          <span className="text-muted/40">·</span>
          <button className="text-muted hover:text-ink" onClick={onClearAll}>
            clear
          </button>
        </div>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {rows.map((r) => {
          const isSelected = selectedRowIds.includes(r.retrieval_run_id);
          const c = modelColor(r.retrieval_model as string);
          return (
            <button
              key={r.retrieval_run_id}
              onClick={() => onToggle(r.retrieval_run_id)}
              className={`flex items-center gap-1.5 px-2 py-1 rounded border text-xs transition-colors ${
                isSelected
                  ? `${c.chip} font-medium`
                  : "bg-bg border-border text-muted hover:text-ink"
              }`}
              title={r.retrieval_run_id}
            >
              <span className={`inline-block w-2 h-2 rounded-full ${c.dot}`} />
              <span>{rowDisplay(r)}</span>
            </button>
          );
        })}
      </div>
    </section>
  );
}

// ---- k picker -------------------------------------------------------------

function KPicker({
  customKs,
  customKInput,
  setCustomKInput,
  addCustomK,
  removeCustomK,
  activeKs,
}: {
  customKs: number[];
  customKInput: string;
  setCustomKInput: (s: string) => void;
  addCustomK: () => void;
  removeCustomK: (k: number) => void;
  activeKs: number[];
}) {
  return (
    <section className="border border-border rounded-md bg-panel p-3 flex flex-wrap items-center gap-2">
      <span className="text-xs font-medium text-muted uppercase tracking-wider mr-2">
        k columns
      </span>
      {PRESET_KS.map((k) => (
        <span
          key={k}
          className="chip chip-active text-[11px] font-mono"
          title="preset"
        >
          k={k}
        </span>
      ))}
      {customKs.map((k) => (
        <button
          key={k}
          className="chip chip-active text-[11px] font-mono"
          onClick={() => removeCustomK(k)}
          title="click to remove"
        >
          k={k} ×
        </button>
      ))}
      <input
        type="number"
        min={1}
        max={100}
        placeholder="custom k"
        className="border border-border rounded bg-bg px-2 py-0.5 text-xs w-24 ml-2"
        value={customKInput}
        onChange={(e) => setCustomKInput(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") addCustomK();
        }}
      />
      <button className="btn !py-0.5 !text-xs" onClick={addCustomK}>
        + add
      </button>
      <span className="text-[11px] text-muted ml-auto">
        showing {activeKs.length} k value{activeKs.length === 1 ? "" : "s"}
      </span>
    </section>
  );
}

// ---- main comparison table -----------------------------------------------

function ComparisonTable({
  rows,
  ks,
}: {
  rows: GenerationSummaryRow[];
  ks: number[];
}) {
  // Group metrics by their group attribute.
  const groups: { name: MetricDef["group"]; metrics: MetricDef[] }[] = [];
  for (const m of METRICS) {
    const last = groups[groups.length - 1];
    if (last && last.name === m.group) last.metrics.push(m);
    else groups.push({ name: m.group, metrics: [m] });
  }

  // Pre-compute per-row max for highlighting (the max across all (model, k) cells).
  const rowMaxByMetric = useMemo(() => {
    const out = new Map<string, number>();
    for (const m of METRICS) {
      let best = -Infinity;
      for (const row of rows) {
        for (const k of ks) {
          const cell = row.cells[String(k)];
          if (!cell) continue;
          const v = m.read(cell);
          if (v != null && v > best) best = v;
        }
      }
      if (best > -Infinity) out.set(m.key, best);
    }
    return out;
  }, [rows, ks]);

  // Local max per (metric, retrieval model): max k inside that model's column block.
  const localMaxByMetricModel = useMemo(() => {
    const out = new Map<string, Map<string, number>>();
    for (const m of METRICS) {
      const inner = new Map<string, number>();
      for (const row of rows) {
        let best = -Infinity;
        for (const k of ks) {
          const cell = row.cells[String(k)];
          if (!cell) continue;
          const v = m.read(cell);
          if (v != null && v > best) best = v;
        }
        if (best > -Infinity) inner.set(row.retrieval_run_id, best);
      }
      out.set(m.key, inner);
    }
    return out;
  }, [rows, ks]);

  return (
    <section className="border border-border rounded-md bg-panel overflow-hidden">
      <div className="overflow-x-auto">
        <table className="text-xs border-collapse">
          <thead>
            {/* Model header row spans its k columns */}
            <tr className="bg-bg/40">
              <th
                rowSpan={2}
                className="px-3 py-1.5 text-left font-medium text-muted border-b border-border sticky left-0 bg-panel z-10"
              >
                Metric
              </th>
              {rows.map((row) => {
                const c = modelColor(row.retrieval_model as string);
                return (
                  <th
                    key={row.retrieval_run_id}
                    colSpan={ks.length}
                    className={`px-3 py-1.5 text-center font-medium border-l border-border border-b ${c.text}`}
                    title={row.retrieval_run_id}
                  >
                    <span className="inline-flex items-center gap-1.5">
                      <span className={`inline-block w-2 h-2 rounded-full ${c.dot}`} />
                      {rowDisplay(row)}
                    </span>
                  </th>
                );
              })}
            </tr>
            <tr className="bg-bg/30 text-muted">
              {rows.flatMap((row) =>
                ks.map((k, i) => (
                  <th
                    key={`${row.retrieval_run_id}-k${k}`}
                    className={`px-2 py-1 text-right text-[11px] font-mono border-b border-border ${
                      i === 0 ? "border-l border-border" : ""
                    }`}
                  >
                    k={k}
                  </th>
                )),
              )}
            </tr>
          </thead>
          <tbody>
            {groups.map((g, gi) => (
              <Group
                key={g.name}
                group={g}
                rows={rows}
                ks={ks}
                rowMaxByMetric={rowMaxByMetric}
                localMaxByMetricModel={localMaxByMetricModel}
                isFirst={gi === 0}
              />
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Group({
  group,
  rows,
  ks,
  rowMaxByMetric,
  localMaxByMetricModel,
  isFirst,
}: {
  group: { name: MetricDef["group"]; metrics: MetricDef[] };
  rows: GenerationSummaryRow[];
  ks: number[];
  rowMaxByMetric: Map<string, number>;
  localMaxByMetricModel: Map<string, Map<string, number>>;
  isFirst: boolean;
}) {
  const totalCols = 1 + rows.length * ks.length;
  return (
    <>
      <tr className={`bg-bg/20 ${isFirst ? "" : "border-t-2 border-border"}`}>
        <td
          colSpan={totalCols}
          className="px-3 py-1 text-[10px] uppercase font-semibold tracking-wider text-muted"
        >
          {group.name}
        </td>
      </tr>
      {group.metrics.map((m) => {
        const maxVal = rowMaxByMetric.get(m.key);
        const localMaxByModel = localMaxByMetricModel.get(m.key);
        return (
          <tr key={m.key} className="border-t border-border/40">
            <td className="px-3 py-1.5 sticky left-0 bg-panel z-10 font-medium text-ink whitespace-nowrap">
              {m.label}
            </td>
            {rows.flatMap((row) =>
              ks.map((k, i) => {
                const cell = row.cells[String(k)];
                const v = cell ? m.read(cell) : null;
                const isMax =
                  v != null && maxVal != null && Math.abs(v - maxVal) < 1e-9;
                const localMax = localMaxByModel?.get(row.retrieval_run_id);
                const isLocalMax =
                  !isMax &&
                  v != null &&
                  localMax != null &&
                  Math.abs(v - localMax) < 1e-9;
                return (
                  <td
                    key={`${m.key}-${row.retrieval_run_id}-k${k}`}
                    className={`px-2 py-1 text-right font-mono tabular-nums ${
                      i === 0 ? "border-l border-border" : ""
                    } ${
                      isMax
                        ? "bg-emerald-200 text-emerald-900 font-semibold"
                        : isLocalMax
                          ? "bg-emerald-50 text-emerald-800"
                          : v == null
                            ? "text-muted"
                            : ""
                    }`}
                    title={cell?.run_id}
                  >
                    {fmt(v)}
                  </td>
                );
              }),
            )}
          </tr>
        );
      })}
    </>
  );
}
