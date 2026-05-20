import { useEffect, useMemo, useState } from "react";
import { fetchQueries } from "../lib/api";
import type { QueryItem, QuestionType } from "../lib/types";

const TYPES: (QuestionType | "all")[] = ["all", "factoid", "list", "yesno", "summary"];

const TYPE_COLORS: Record<string, string> = {
  factoid: "text-blue-700 bg-blue-50 border-blue-200",
  list:    "text-violet-700 bg-violet-50 border-violet-200",
  yesno:   "text-amber-700 bg-amber-50 border-amber-200",
  summary: "text-teal-700 bg-teal-50 border-teal-200",
};

const TYPE_ACTIVE: Record<string, string> = {
  all:     "bg-ink text-white border-ink",
  factoid: "bg-blue-600 text-white border-blue-600",
  list:    "bg-violet-600 text-white border-violet-600",
  yesno:   "bg-amber-500 text-white border-amber-500",
  summary: "bg-teal-600 text-white border-teal-600",
};

interface Props {
  onPick: (q: QueryItem) => void;
  activeId: string | null;
}

export default function QuerySidebar({ onPick, activeId }: Props) {
  const [queries, setQueries] = useState<QueryItem[]>([]);
  const [filter, setFilter] = useState<QuestionType | "all">("all");
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchQueries()
      .then(setQueries)
      .catch((e) => console.error(e))
      .finally(() => setLoading(false));
  }, []);

  const filtered = useMemo(() => {
    return queries.filter((q) => {
      if (filter !== "all" && q.type !== filter) return false;
      if (search && !q.text.toLowerCase().includes(search.toLowerCase())) return false;
      return true;
    });
  }, [queries, filter, search]);

  return (
    <aside className="w-80 shrink-0 h-full flex flex-col border-r border-border bg-panel">
      <div className="p-3 border-b border-border space-y-2.5">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-muted">
            Test queries
          </h2>
          <span className="text-xs text-muted font-mono">{queries.length}</span>
        </div>
        <input
          className="input text-xs"
          placeholder="Search queries…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <div className="flex flex-wrap gap-1">
          {TYPES.map((t) => (
            <button
              key={t}
              className={`chip capitalize transition-all duration-150 ${
                filter === t ? TYPE_ACTIVE[t] : "hover:border-muted/50"
              }`}
              onClick={() => setFilter(t)}
            >
              {t}
            </button>
          ))}
        </div>
      </div>
      <div className="flex-1 overflow-y-auto">
        {loading && (
          <div className="flex items-center gap-2 p-4 text-sm text-muted">
            <span className="inline-flex gap-1">
              <span className="animate-pulse">•</span>
              <span className="animate-pulse [animation-delay:150ms]">•</span>
              <span className="animate-pulse [animation-delay:300ms]">•</span>
            </span>
            Loading…
          </div>
        )}
        {!loading &&
          filtered.map((q) => (
            <button
              key={q.id}
              onClick={() => onPick(q)}
              className={`sidebar-item ${activeId === q.id ? "sidebar-item-active" : ""}`}
            >
              <div className="flex items-start gap-2">
                {q.type && (
                  <span className={`chip mt-0.5 shrink-0 capitalize text-[10px] ${TYPE_COLORS[q.type] ?? ""}`}>
                    {q.type}
                  </span>
                )}
                <span className="text-sm leading-snug">{q.text}</span>
              </div>
            </button>
          ))}
        {!loading && filtered.length === 0 && (
          <div className="p-4 text-sm text-muted text-center">No queries match.</div>
        )}
      </div>
    </aside>
  );
}
