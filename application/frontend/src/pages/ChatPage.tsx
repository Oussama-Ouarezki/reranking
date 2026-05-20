import { useState } from "react";
import QuerySidebar from "../components/QuerySidebar";
import ChatThread from "../components/ChatThread";
import ChatInput from "../components/ChatInput";
import ModelSidebar from "../components/ModelSidebar";
import { useApp } from "../lib/store";
import { chat } from "../lib/api";
import type { ChatMessage, QueryItem } from "../lib/types";

export default function ChatPage() {
  const {
    model, topK, generate, messages,
    setModel, setTopK, setGenerate,
    appendMessage, updateMessage,
  } = useApp();

  const [busy, setBusy] = useState(false);
  const [activeQid, setActiveQid] = useState<string | null>(null);
  const [pendingPrefill, setPendingPrefill] = useState<string>("");

  const handlePick = (q: QueryItem) => {
    setActiveQid(q.id);
    setPendingPrefill(q.text);
  };

  const handleSend = async (text: string) => {
    setBusy(true);
    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
    };
    const assistMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "",
      loading: true,
      model,
    };
    appendMessage(userMsg);
    appendMessage(assistMsg);

    // Build conversation history (chat memory)
    const history = messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    try {
      const resp = await chat({
        message: text,
        model,
        top_k: topK,
        history,
        query_id: activeQid,
        generate,
      });
      updateMessage(assistMsg.id, {
        loading: false,
        content: resp.answer ?? "(no answer generated)",
        retrieved: resp.retrieved,
        question_type: resp.question_type,
        no_relevant: resp.no_relevant,
        metrics: resp.metrics,
        timings: resp.timings,
        n_relevant_retrieved: resp.n_relevant_retrieved,
        qa_score: resp.qa_score ?? null,
        qa_metric_label: resp.qa_metric_label ?? null,
        qa_extra: resp.qa_extra ?? null,
      });
    } catch (e) {
      updateMessage(assistMsg.id, {
        loading: false,
        content: `Error: ${(e as Error).message}`,
      });
    } finally {
      setBusy(false);
      setActiveQid(null); // each turn re-detects type unless a new query is picked
    }
  };

  return (
    <div className="h-full flex">
      <QuerySidebar onPick={handlePick} activeId={activeQid} />
      <div className="flex-1 flex flex-col min-w-0">
        <ChatThread messages={messages} />
        <ChatInput
          topK={topK}
          generate={generate}
          busy={busy}
          onSend={handleSend}
          onTopKChange={setTopK}
          onGenerateChange={setGenerate}
          initialText={pendingPrefill}
          key={pendingPrefill /* force remount when sidebar prefills */}
        />
      </div>
      <ModelSidebar model={model} onModelChange={setModel} />
    </div>
  );
}
