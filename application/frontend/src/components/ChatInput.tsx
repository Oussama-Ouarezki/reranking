import { useState } from "react";

const TOPK_OPTIONS = [5, 10, 20, 50];

interface Props {
  topK: number;
  generate: boolean;
  busy: boolean;
  onSend: (text: string) => void;
  onTopKChange: (k: number) => void;
  onGenerateChange: (g: boolean) => void;
  initialText?: string;
}

export default function ChatInput({
  topK,
  generate,
  busy,
  onSend,
  onTopKChange,
  onGenerateChange,
  initialText,
}: Props) {
  const [text, setText] = useState(initialText ?? "");

  const submit = () => {
    const t = text.trim();
    if (!t || busy) return;
    onSend(t);
    setText("");
  };

  return (
    <div className="border-t border-border bg-panel/95 backdrop-blur-sm">
      <div className="max-w-3xl mx-auto px-4 py-3 space-y-2.5">
        <div className="flex items-center gap-4 text-xs text-muted">
          <label className="flex items-center gap-1.5">
            <span className="font-medium">Top-K</span>
            <select
              className="input !py-0.5 !px-2 !w-auto text-xs"
              value={topK}
              onChange={(e) => onTopKChange(Number(e.target.value))}
              disabled={busy}
            >
              {TOPK_OPTIONS.map((k) => (
                <option key={k} value={k}>{k}</option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-1.5 cursor-pointer select-none group">
            <div className={`w-8 h-4 rounded-full transition-colors duration-200 flex items-center px-0.5 ${generate ? "bg-accent" : "bg-border"}`}>
              <div className={`w-3 h-3 rounded-full bg-white shadow-sm transition-transform duration-200 ${generate ? "translate-x-4" : "translate-x-0"}`} />
              <input
                type="checkbox"
                checked={generate}
                onChange={(e) => onGenerateChange(e.target.checked)}
                disabled={busy}
                className="sr-only"
              />
            </div>
            <span className="font-medium">Generate answer</span>
          </label>
        </div>
        <div className="flex gap-2 items-end">
          <textarea
            className="input resize-none flex-1 leading-relaxed"
            rows={2}
            placeholder="Ask a biomedical question…"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            disabled={busy}
          />
          <button
            className="btn btn-primary h-[68px] w-14 flex-col gap-0.5 text-xs"
            onClick={submit}
            disabled={busy || !text.trim()}
          >
            {busy ? (
              <span className="animate-spin text-base">⟳</span>
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <path d="M14 8H2M9 3l5 5-5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                <span>Send</span>
              </>
            )}
          </button>
        </div>
        <p className="text-[10px] text-muted/60 text-right">Enter to send · Shift+Enter for newline</p>
      </div>
    </div>
  );
}
