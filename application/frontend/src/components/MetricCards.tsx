import type { RankingMetrics } from "../lib/types";

interface Props {
  metrics: RankingMetrics;
}

const KS = [1, 5, 10, 20];
const ROWS: { key: keyof RankingMetrics; label: string }[] = [
  { key: "ndcg_at", label: "nDCG" },
  { key: "mrr_at",  label: "MRR"  },
  { key: "p_at",    label: "P"    },
  { key: "r_at",    label: "Recall"},
  { key: "map_at",  label: "MAP"  },
];

function valueColor(v: number) {
  if (v >= 0.8) return "text-emerald-700 font-semibold";
  if (v >= 0.6) return "text-amber-600";
  if (v >= 0.4) return "text-orange-500";
  return "text-muted";
}

export default function MetricCards({ metrics }: Props) {
  return (
    <div className="border border-border rounded-lg bg-panel overflow-hidden shadow-sm">
      <div className="px-4 py-2 text-xs font-semibold text-muted border-b border-border bg-bg/40 uppercase tracking-wider">
        Ranking metrics
      </div>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-muted bg-bg/20">
            <th className="px-4 py-1.5 text-left font-medium"></th>
            {KS.map((k) => (
              <th key={k} className="px-3 py-1.5 text-right font-medium text-muted/70">@{k}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {ROWS.map((row, i) => (
            <tr key={row.key} className={`border-t border-border/50 ${i % 2 === 0 ? "" : "bg-bg/20"}`}>
              <td className="px-4 py-1.5 font-semibold text-ink/70">{row.label}</td>
              {KS.map((k) => {
                const v = metrics[row.key]?.[k] ?? 0;
                return (
                  <td key={k} className={`px-3 py-1.5 text-right font-mono tabular-nums ${valueColor(v)}`}>
                    {v.toFixed(4)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
