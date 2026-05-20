import { useEffect, useMemo, useState } from "react";

import { fetchGenRuns, runStatTest } from "../lib/api";
import type {
  GenerationRunSummary,
  QuestionType,
  StatBlock,
  StatTestResponse,
} from "../lib/types";

const QTYPES: QuestionType[] = ["yesno", "factoid", "list", "summary"];
const QTYPE_LABELS: Record<QuestionType, string> = {
  yesno: "Yes / No",
  factoid: "Factoid",
  list: "List",
  summary: "Summary",
};

function fmt(v: number | null | undefined, digits = 4) {
  return typeof v === "number" ? v.toFixed(digits) : "—";
}

function fmtP(p: number) {
  if (p < 1e-6) return p.toExponential(2);
  if (p < 0.001) return p.toExponential(2);
  return p.toFixed(4);
}

function fmtRunLabel(r: GenerationRunSummary) {
  const model = r.retrieval_model ?? "?";
  const date = new Date(r.ended_at * 1000).toLocaleString(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
  return `${model} · k=${r.k} · ${date}${r.comment ? ` — ${r.comment}` : ""}`;
}

export default function StatisticalTestingPage() {
  const [runs, setRuns] = useState<GenerationRunSummary[]>([]);
  const [runA, setRunA] = useState<string | null>(null);
  const [runB, setRunB] = useState<string | null>(null);
  const [scope, setScope] = useState<"all" | QuestionType>("all");
  const [result, setResult] = useState<StatTestResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetchGenRuns().then(setRuns).catch((e) => setErr(String(e)));
  }, []);

  const handleRun = async () => {
    if (!runA || !runB || runA === runB) return;
    setBusy(true);
    setErr(null);
    try {
      const qtypes = scope === "all" ? undefined : [scope];
      const r = await runStatTest(runA, runB, qtypes);
      setResult(r);
    } catch (e) {
      setErr(String(e));
      setResult(null);
    } finally {
      setBusy(false);
    }
  };

  const visibleBlocks = useMemo(() => {
    if (!result) return [] as Array<{ label: string; block: StatBlock; key: string }>;
    const out: Array<{ label: string; block: StatBlock; key: string }> = [];
    for (const qt of QTYPES) {
      const block = result.by_qtype[qt];
      if (block) out.push({ label: QTYPE_LABELS[qt], block, key: qt });
    }
    return out;
  }, [result]);

  return (
    <div className="h-full overflow-y-auto px-8 py-6">
      <div className="max-w-5xl mx-auto space-y-5">
        <div>
          <h1 className="text-xl font-semibold">Statistical testing</h1>
          <p className="text-xs text-muted mt-1">
            Two-sided Wilcoxon signed-rank test on paired per-query QA scores.
            Per-qtype uses the metric for that type (Acc, MRR, F1, Judge); the
            global block pools all types. p &lt; 0.05 is highlighted as
            significant.
          </p>
        </div>

        <section className="border border-border rounded-md bg-panel p-3 space-y-3">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <RunPicker label="Run A (model under test)" runs={runs} value={runA} onChange={setRunA} />
            <RunPicker label="Run B (baseline)" runs={runs} value={runB} onChange={setRunB} />
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <label className="flex items-center gap-2 text-xs text-muted">
              Scope
              <select
                className="border border-border rounded bg-bg px-2 py-1 text-sm text-ink"
                value={scope}
                onChange={(e) => setScope(e.target.value as typeof scope)}
              >
                <option value="all">All qtypes + global</option>
                {QTYPES.map((q) => (
                  <option key={q} value={q}>{QTYPE_LABELS[q]} only</option>
                ))}
              </select>
            </label>
            <button
              className="btn btn-primary"
              disabled={!runA || !runB || runA === runB || busy}
              onClick={handleRun}
            >
              {busy ? "Running…" : "Run test"}
            </button>
            {runA && runB && runA === runB && (
              <span className="text-xs text-accent">Pick two different runs.</span>
            )}
          </div>
          {err && <div className="text-xs text-accent">{err}</div>}
        </section>

        {result && (
          <>
            <ResultHeader r={result} />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {visibleBlocks.map(({ label, block, key }) => (
                <BlockCard
                  key={key}
                  title={label}
                  metricLabel={result.metric_labels[key] ?? "Score"}
                  block={block}
                />
              ))}
            </div>
            <BlockCard
              title="Global (all qtypes pooled)"
              metricLabel={result.metric_labels.global ?? "QA score"}
              block={result.global_block}
              wide
            />
          </>
        )}
      </div>
    </div>
  );
}

function RunPicker({
  label,
  runs,
  value,
  onChange,
}: {
  label: string;
  runs: GenerationRunSummary[];
  value: string | null;
  onChange: (id: string | null) => void;
}) {
  return (
    <label className="block">
      <div className="text-xs text-muted mb-1">{label}</div>
      <select
        className="w-full border border-border rounded bg-bg px-2 py-1.5 text-sm text-ink"
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value || null)}
      >
        <option value="">— pick a generation run —</option>
        {runs.map((r) => (
          <option key={r.run_id} value={r.run_id}>{fmtRunLabel(r)}</option>
        ))}
      </select>
    </label>
  );
}

function ResultHeader({ r }: { r: StatTestResponse }) {
  return (
    <div className="border border-border rounded-md bg-panel px-4 py-3 text-xs text-muted flex items-baseline gap-4 flex-wrap">
      <span>
        A: <span className="font-mono text-ink">{r.retrieval_model_a ?? "?"}</span>
        {r.k_a != null && <> @k={r.k_a}</>}
      </span>
      <span>vs (baseline)</span>
      <span>
        B: <span className="font-mono text-ink">{r.retrieval_model_b ?? "?"}</span>
        {r.k_b != null && <> @k={r.k_b}</>}
      </span>
      <span className="ml-auto">{r.global_block.n_pairs} paired queries</span>
    </div>
  );
}

function BlockCard({
  title,
  metricLabel,
  block,
  wide,
}: {
  title: string;
  metricLabel: string;
  block: StatBlock;
  wide?: boolean;
}) {
  const delta = block.mean_delta;
  const sigBg = block.significant
    ? "border-emerald-400 bg-emerald-50/60"
    : "border-border bg-panel";
  return (
    <section className={`border rounded-md p-4 space-y-3 ${sigBg} ${wide ? "md:col-span-2" : ""}`}>
      <div className="flex items-baseline justify-between gap-2 flex-wrap">
        <h3 className="text-sm font-semibold text-ink">{title}</h3>
        <span className="text-[11px] text-muted font-mono">{metricLabel}</span>
      </div>
      {block.n_pairs === 0 ? (
        <div className="text-xs text-muted">No paired queries.</div>
      ) : (
        <>
          <div className="grid grid-cols-3 gap-3 text-xs">
            <Cell label="A mean" value={fmt(block.mean_a)} />
            <Cell label="B mean" value={fmt(block.mean_b)} />
            <Cell
              label="Δ (A−B)"
              value={fmt(delta)}
              color={delta > 0 ? "text-emerald-700" : delta < 0 ? "text-accent" : "text-muted"}
            />
            <Cell
              label="p-value"
              value={fmtP(block.p_value)}
              color={block.significant ? "text-emerald-700 font-semibold" : "text-ink"}
              hint={block.significant ? "p < 0.05" : "n.s."}
            />
            <Cell label="z" value={block.z.toFixed(3)} />
            <Cell label="W (statistic)" value={block.statistic.toFixed(1)} />
          </div>
          <div className="text-[11px] text-muted">
            n = {block.n_pairs} pairs ·{" "}
            <span className="text-emerald-700">A wins {block.n_a_wins}</span> ·{" "}
            <span className="text-accent">B wins {block.n_b_wins}</span> ·{" "}
            ties {block.n_ties} · median Δ {fmt(block.median_delta)}
          </div>
        </>
      )}
    </section>
  );
}

function Cell({
  label,
  value,
  color,
  hint,
}: {
  label: string;
  value: string;
  color?: string;
  hint?: string;
}) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-muted">{label}</div>
      <div className={`font-mono tabular-nums text-sm ${color ?? "text-ink"}`}>{value}</div>
      {hint && <div className="text-[10px] text-muted">{hint}</div>}
    </div>
  );
}
