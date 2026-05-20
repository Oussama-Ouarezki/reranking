import { useEffect, useRef, useState } from "react";
import type { ChatMessage, RetrievedDoc } from "../lib/types";
import MetricCards from "./MetricCards";
import GenerationMetricCard from "./GenerationMetricCard";

const TYPE_COLORS: Record<string, string> = {
  factoid: "text-blue-700 bg-blue-50 border-blue-200",
  list:    "text-violet-700 bg-violet-50 border-violet-200",
  yesno:   "text-amber-700 bg-amber-50 border-amber-200",
  summary: "text-teal-700 bg-teal-50 border-teal-200",
};

interface Props {
  messages: ChatMessage[];
}

export default function ChatThread({ messages }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex-1 overflow-y-auto px-6 py-6">
      <div className="max-w-3xl mx-auto space-y-6">
        {messages.length === 0 && (
          <div className="flex flex-col items-center gap-3 pt-24 text-center">
            <div className="w-12 h-12 rounded-2xl bg-accent/10 flex items-center justify-center">
              <svg width="22" height="22" viewBox="0 0 22 22" fill="none" className="text-accent">
                <circle cx="11" cy="11" r="4" fill="currentColor" />
                <circle cx="3.5" cy="6.5" r="2.5" fill="currentColor" opacity=".4" />
                <circle cx="18.5" cy="6.5" r="2.5" fill="currentColor" opacity=".4" />
                <circle cx="3.5" cy="15.5" r="2.5" fill="currentColor" opacity=".4" />
                <circle cx="18.5" cy="15.5" r="2.5" fill="currentColor" opacity=".4" />
              </svg>
            </div>
            <p className="text-muted text-sm">Pick a query from the sidebar or type a biomedical question.</p>
          </div>
        )}
        {messages.map((m) => (
          <Message key={m.id} m={m} />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

function Message({ m }: { m: ChatMessage }) {
  if (m.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-2xl px-4 py-3 rounded-2xl rounded-tr-sm bg-ink text-white shadow-sm">
          <div className="text-sm whitespace-pre-wrap leading-relaxed">{m.content}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 flex-wrap">
        {m.model && (
          <span className="chip font-mono text-[10px] tracking-wide">{m.model}</span>
        )}
        {m.question_type && (
          <span className={`chip capitalize ${TYPE_COLORS[m.question_type] ?? ""}`}>
            {m.question_type}
          </span>
        )}
        {m.no_relevant && (
          <span className="chip border-rose-300 text-rose-600 bg-rose-50">no relevant docs</span>
        )}
        {m.n_relevant_retrieved != null && m.n_relevant_retrieved > 0 && (
          <span className="chip border-emerald-300 text-emerald-700 bg-emerald-50">
            {m.n_relevant_retrieved} gold retrieved
          </span>
        )}
        {m.timings && <TimingsChips t={m.timings} />}
      </div>
      <div>
        {m.loading ? (
          <div className="flex items-center gap-1.5 text-muted py-1">
            <span className="w-2 h-2 rounded-full bg-muted/60 animate-bounce [animation-delay:0ms]" />
            <span className="w-2 h-2 rounded-full bg-muted/60 animate-bounce [animation-delay:150ms]" />
            <span className="w-2 h-2 rounded-full bg-muted/60 animate-bounce [animation-delay:300ms]" />
          </div>
        ) : (
          <div className="text-sm whitespace-pre-wrap leading-relaxed">{m.content}</div>
        )}
      </div>
      {m.metrics && <MetricCards metrics={m.metrics} />}
      {(m.qa_score != null || m.qa_metric_label) && (
        <GenerationMetricCard
          score={m.qa_score ?? null}
          metricLabel={m.qa_metric_label ?? null}
          extra={m.qa_extra ?? null}
          qtype={m.question_type ?? null}
        />
      )}
      {m.retrieved && m.retrieved.length > 0 && <RetrievedAccordion docs={m.retrieved} />}
    </div>
  );
}

function TimingsChips({ t }: { t: NonNullable<ChatMessage["timings"]> }) {
  const fmt = (s: number) => (s < 1 ? `${Math.round(s * 1000)}ms` : `${s.toFixed(2)}s`);
  return (
    <span className="text-[11px] text-muted font-mono">
      retrieve <b>{fmt(t.retrieve_s)}</b> · rerank <b>{fmt(t.rerank_s)}</b> · gen <b>{fmt(t.generate_s)}</b> · total <b className="text-ink">{fmt(t.total_s)}</b>
    </span>
  );
}

function RetrievedAccordion({ docs }: { docs: RetrievedDoc[] }) {
  const [open, setOpen] = useState(false);
  const goldCount = docs.filter((d) => d.is_relevant).length;

  return (
    <div className="border border-border rounded-lg bg-panel overflow-hidden shadow-sm">
      <button
        className="w-full px-4 py-2.5 text-xs font-medium text-muted hover:bg-bg flex items-center justify-between transition-colors duration-100"
        onClick={() => setOpen(!open)}
      >
        <span className="flex items-center gap-2">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none" className="opacity-60">
            <path d="M1 3h10M1 6h7M1 9h5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
          Retrieved documents · top {docs.length}
          {goldCount > 0 && (
            <span className="text-emerald-600 font-semibold">· {goldCount} gold</span>
          )}
        </span>
        <span className="text-muted/60">{open ? "▾" : "▸"}</span>
      </button>
      {open && (
        <ul className="divide-y divide-border">
          {docs.map((d) => (
            <li
              key={d.docid}
              className={`px-4 py-2.5 text-sm ${
                d.is_relevant ? "bg-emerald-50/60 border-l-2 border-l-emerald-500" : ""
              }`}
            >
              <div className="flex items-baseline gap-2 mb-1">
                <span className={`text-xs font-mono shrink-0 ${d.is_relevant ? "text-emerald-600 font-semibold" : "text-muted"}`}>
                  #{d.rank}
                </span>
                {d.is_relevant && (
                  <span className="text-[10px] font-bold text-emerald-600 uppercase tracking-widest">gold</span>
                )}
                <a
                  href={`https://pubmed.ncbi.nlm.nih.gov/${d.docid}/`}
                  target="_blank"
                  rel="noreferrer"
                  className={`font-medium hover:underline leading-snug ${d.is_relevant ? "text-emerald-900" : "text-ink"}`}
                >
                  {d.title || d.docid}
                </a>
                <span className="ml-auto text-xs text-muted font-mono shrink-0">{d.score.toFixed(3)}</span>
              </div>
              <div className="text-xs text-muted leading-snug pl-7">{d.snippet}</div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
