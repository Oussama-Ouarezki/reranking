import type { QuestionType } from "../lib/types";

interface Props {
  score: number | null;
  metricLabel: string | null;
  extra: Record<string, unknown> | null;
  qtype: QuestionType | null;
}

const EXTRA_LABELS: Record<string, string> = {
  rouge_l: "ROUGE-L",
  bert_score: "BERTScore",
  strict_acc: "Strict Acc",
  map: "MAP",
  pred_label: "Predicted",
  gold_label: "Gold",
};

function valueColor(v: number) {
  if (v >= 0.8) return "text-emerald-700 font-semibold";
  if (v >= 0.6) return "text-amber-600";
  if (v >= 0.4) return "text-orange-500";
  return "text-muted";
}

function fmt(v: unknown): string {
  if (v == null) return "—";
  if (typeof v === "number") return v.toFixed(4);
  return String(v);
}

export default function GenerationMetricCard({ score, metricLabel, extra, qtype }: Props) {
  const extras = extra ? Object.entries(extra).filter(([, v]) => v != null) : [];
  return (
    <div className="border border-border rounded-lg bg-panel overflow-hidden shadow-sm">
      <div className="px-4 py-2 text-xs font-semibold text-muted border-b border-border bg-bg/40 uppercase tracking-wider flex items-center justify-between">
        <span>Generation metrics</span>
        {qtype && <span className="font-mono normal-case text-[10px] text-muted/80">{qtype}</span>}
      </div>
      <div className="px-4 py-2.5 flex items-baseline gap-6 flex-wrap">
        <div className="flex items-baseline gap-2">
          <span className="text-xs font-semibold text-ink/70 uppercase tracking-wider">
            {metricLabel ?? "Score"}
          </span>
          <span
            className={`font-mono tabular-nums text-base ${
              typeof score === "number" ? valueColor(score) : "text-muted"
            }`}
          >
            {fmt(score)}
          </span>
        </div>
        {extras.length > 0 && (
          <>
            <span className="text-muted/40">·</span>
            {extras.map(([k, v]) => (
              <div key={k} className="flex items-baseline gap-2">
                <span className="text-xs text-muted">{EXTRA_LABELS[k] ?? k}</span>
                <span
                  className={`font-mono tabular-nums text-xs ${
                    typeof v === "number" ? valueColor(v as number) : "text-ink/80"
                  }`}
                >
                  {fmt(v)}
                </span>
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}
