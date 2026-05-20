import { useEffect, useState } from "react";

import { fetchGenRuns, runFailureAnalysis } from "../lib/api";
import type {
  FailureBucket,
  FailureRecord,
  FailureResponse,
  GenerationRunSummary,
  QuestionType,
  RetrievalMetricSet,
} from "../lib/types";

const QTYPE_BADGE: Record<QuestionType, string> = {
  factoid: "bg-sky-100 text-sky-700",
  yesno: "bg-emerald-100 text-emerald-700",
  list: "bg-amber-100 text-amber-700",
  summary: "bg-violet-100 text-violet-700",
};

function fmtRunLabel(r: GenerationRunSummary) {
  const date = new Date(r.ended_at * 1000).toLocaleString(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
  return `${r.retrieval_model ?? "?"} · k=${r.k} · ${date}${r.comment ? ` — ${r.comment}` : ""}`;
}

export default function FailureAnalysisPage() {
  const [runs, setRuns] = useState<GenerationRunSummary[]>([]);
  const [runA, setRunA] = useState<string | null>(null);
  const [runB, setRunB] = useState<string | null>(null);
  const [result, setResult] = useState<FailureResponse | null>(null);
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
      const r = await runFailureAnalysis(runA, runB);
      setResult(r);
    } catch (e) {
      setErr(String(e));
      setResult(null);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="h-full overflow-y-auto px-8 py-6">
      <div className="max-w-5xl mx-auto space-y-5">
        <div>
          <h1 className="text-xl font-semibold">Failure analysis</h1>
          <p className="text-xs text-muted mt-1">
            For each query present in both gen runs, flag a failure when the
            QA delta exceeds the per-qtype threshold. Direction is bidirectional
            — the "failed" side is whichever scored lower.
          </p>
        </div>

        <section className="border border-border rounded-md bg-panel p-3 space-y-3">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <RunPicker label="Run A (baseline)" runs={runs} value={runA} onChange={setRunA} />
            <RunPicker label="Run B (comparison)" runs={runs} value={runB} onChange={setRunB} />
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <button
              className="btn btn-primary"
              disabled={!runA || !runB || runA === runB || busy}
              onClick={handleRun}
            >
              {busy ? "Computing…" : "Find failures"}
            </button>
            {runA && runB && runA === runB && (
              <span className="text-xs text-accent">Pick two different runs.</span>
            )}
            <span className="text-[11px] text-muted ml-auto">
              Thresholds — Yes/No: any flip · Factoid (MRR): |Δ| &gt; 0.1 · List (F1): |Δ| &gt; 0.1 · Summary (Judge): |Δ| &gt; 0.5
            </span>
          </div>
          {err && <div className="text-xs text-accent">{err}</div>}
        </section>

        {result && <Summary r={result} />}
        {result && result.by_qtype.map((b) => (
          <BucketSection key={b.qtype} bucket={b} />
        ))}
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

function Summary({ r }: { r: FailureResponse }) {
  return (
    <section className="border border-border rounded-md bg-panel px-4 py-3 text-xs">
      <div className="flex items-baseline gap-4 flex-wrap mb-2">
        <span className="font-medium text-ink">
          A: <span className="font-mono">{r.retrieval_model_a ?? "?"}</span>
          {r.k_a != null && <> @k={r.k_a}</>}
        </span>
        <span className="text-muted">vs (baseline)</span>
        <span className="font-medium text-ink">
          B: <span className="font-mono">{r.retrieval_model_b ?? "?"}</span>
          {r.k_b != null && <> @k={r.k_b}</>}
        </span>
        <span className="ml-auto text-muted">{r.n_overlapping} overlapping queries</span>
      </div>
      <div className="flex items-baseline gap-6 flex-wrap">
        <span>
          <span className="text-muted">Total failures</span>{" "}
          <span className="font-mono font-semibold text-ink">{r.total_failures}</span>
        </span>
        <span>
          <span className="text-emerald-700">A failed (B better)</span>{" "}
          <span className="font-mono">{r.total_a_failed}</span>
        </span>
        <span>
          <span className="text-accent">B failed (A better)</span>{" "}
          <span className="font-mono">{r.total_b_failed}</span>
        </span>
      </div>
    </section>
  );
}

function BucketSection({ bucket }: { bucket: FailureBucket }) {
  const [expanded, setExpanded] = useState(true);
  return (
    <section className="border border-border rounded-md bg-panel overflow-hidden">
      <button
        className="w-full px-4 py-2 flex items-center justify-between gap-3 bg-bg/40 hover:bg-bg/60 transition-colors"
        onClick={() => setExpanded((e) => !e)}
      >
        <span className="flex items-baseline gap-3">
          <span className={`px-2 py-0.5 rounded text-[10px] font-medium ${QTYPE_BADGE[bucket.qtype]}`}>
            {bucket.qtype}
          </span>
          <span className="text-sm font-medium text-ink">
            {bucket.n_failures} failure{bucket.n_failures === 1 ? "" : "s"}
          </span>
          <span className="text-xs text-muted">
            of {bucket.n_pairs} paired · A failed {bucket.n_a_failed} · B failed {bucket.n_b_failed}
          </span>
        </span>
        <span className="text-xs text-muted font-mono">
          {bucket.metric_label} · threshold {bucket.qtype === "yesno" ? "any flip" : `|Δ| > ${bucket.threshold}`}
          <span className="ml-2">{expanded ? "▾" : "▸"}</span>
        </span>
      </button>
      {expanded && bucket.failures.length > 0 && (
        <ul className="divide-y divide-border">
          {bucket.failures.map((f) => (
            <FailureRow key={f.qid} f={f} />
          ))}
        </ul>
      )}
      {expanded && bucket.failures.length === 0 && (
        <div className="px-4 py-3 text-xs text-muted">No failures over this threshold.</div>
      )}
    </section>
  );
}

function FailureRow({ f }: { f: FailureRecord }) {
  const [open, setOpen] = useState(false);
  const aBetter = f.failed_model === "b";
  const hasRetrieval = !!(f.retrieval_a || f.retrieval_b);
  return (
    <li className="px-4 py-3">
      <div className="flex items-baseline gap-3 flex-wrap">
        <span className="font-mono text-[11px] text-muted shrink-0">{f.qid}</span>
        <span className="text-sm text-ink flex-1 min-w-[200px]">{f.question}</span>
        <span className="flex items-baseline gap-3 shrink-0 font-mono tabular-nums text-xs">
          <span className={aBetter ? "text-emerald-700 font-semibold" : "text-accent"}>
            A {f.score_a.toFixed(3)}
          </span>
          <span className="text-muted">→</span>
          <span className={aBetter ? "text-accent" : "text-emerald-700 font-semibold"}>
            B {f.score_b.toFixed(3)}
          </span>
          <span className={f.delta > 0 ? "text-emerald-700" : "text-accent"}>
            Δ {f.delta >= 0 ? "+" : "−"}{Math.abs(f.delta).toFixed(3)}
          </span>
          <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
            aBetter ? "bg-rose-50 text-accent" : "bg-amber-50 text-amber-700"
          }`}>
            {aBetter ? "B failed" : "A failed"}
          </span>
        </span>
        {(f.answer_a || f.answer_b) && (
          <button className="text-xs text-muted hover:text-ink" onClick={() => setOpen((o) => !o)}>
            {open ? "hide" : "answers"}
          </button>
        )}
      </div>
      {hasRetrieval && (
        <RetrievalMetricsRow
          a={f.retrieval_a}
          b={f.retrieval_b}
          delta={f.retrieval_delta}
          failedSide={f.failed_model}
        />
      )}
      {open && (
        <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
          <AnswerBlock label="A" answer={f.answer_a} />
          <AnswerBlock label="B" answer={f.answer_b} />
        </div>
      )}
    </li>
  );
}

const RETRIEVAL_KEYS: Array<{ key: keyof RetrievalMetricSet; label: string }> = [
  { key: "ndcg", label: "nDCG" },
  { key: "p", label: "P" },
  { key: "r", label: "R" },
  { key: "mrr", label: "MRR" },
  { key: "map", label: "MAP" },
];

function fmt(v: number | undefined | null, signed = false): string {
  if (v === undefined || v === null || Number.isNaN(v)) return "—";
  const s = Math.abs(v).toFixed(3);
  if (!signed) return s;
  if (v > 0) return `+${s}`;
  if (v < 0) return `−${s}`;
  return ` ${s}`;
}

function RetrievalMetricsRow({
  a,
  b,
  delta,
  failedSide,
}: {
  a?: RetrievalMetricSet | null;
  b?: RetrievalMetricSet | null;
  delta?: RetrievalMetricSet | null;
  failedSide: "a" | "b";
}) {
  // "Retrieval explains it" cue: the failed side's retrieval delta moves in
  // the same direction as its QA loss. If A failed, A's retrieval should be
  // worse than B's (delta = b - a > 0). If B failed, delta should be < 0.
  // We show a small badge per metric when that signal is present.
  const explainSign = failedSide === "a" ? 1 : -1;
  return (
    <div className="mt-2 border border-border/70 rounded bg-bg/40 px-3 py-2">
      <div className="flex items-center gap-2 mb-1">
        <span className="text-[10px] uppercase tracking-wider text-muted">retrieval @ k</span>
        <span className="text-[10px] text-muted">
          (does the QA failure track a retrieval gap?)
        </span>
      </div>
      <div className="grid grid-cols-[auto_repeat(5,minmax(0,1fr))] gap-x-4 gap-y-1 text-[11px] font-mono tabular-nums">
        <div className="text-muted" />
        {RETRIEVAL_KEYS.map(({ key, label }) => (
          <div key={`h-${key}`} className="text-muted text-center">{label}</div>
        ))}

        <div className="text-muted">A</div>
        {RETRIEVAL_KEYS.map(({ key }) => (
          <div key={`a-${key}`} className="text-center text-ink/90">{fmt(a?.[key])}</div>
        ))}

        <div className="text-muted">B</div>
        {RETRIEVAL_KEYS.map(({ key }) => (
          <div key={`b-${key}`} className="text-center text-ink/90">{fmt(b?.[key])}</div>
        ))}

        <div className="text-muted">Δ (B−A)</div>
        {RETRIEVAL_KEYS.map(({ key }) => {
          const v = delta?.[key];
          const explains = v !== undefined && v !== null && Math.sign(v) === explainSign && v !== 0;
          const cls =
            v === undefined || v === null
              ? "text-muted"
              : v === 0
              ? "text-muted"
              : explains
              ? "text-accent font-semibold"
              : "text-emerald-700";
          return (
            <div key={`d-${key}`} className={`text-center ${cls}`}>
              {fmt(v, true)}
            </div>
          );
        })}
      </div>
      <div className="mt-1 text-[10px] text-muted">
        Red Δ = retrieval is worse on the failed side ({failedSide.toUpperCase()}), so the QA
        failure may be attributable to retrieval. Green Δ = retrieval favors the failed side,
        so retrieval likely isn't the cause.
      </div>
    </div>
  );
}

function AnswerBlock({ label, answer }: { label: string; answer: string | null | undefined }) {
  return (
    <div className="border border-border rounded bg-bg/40 p-2">
      <div className="text-[10px] uppercase tracking-wider text-muted mb-1">{label}</div>
      <div className="text-xs whitespace-pre-wrap text-ink/90">{answer || "—"}</div>
    </div>
  );
}
