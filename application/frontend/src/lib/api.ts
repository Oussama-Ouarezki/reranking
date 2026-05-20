import type {
  ChatResponse,
  ChatTurn,
  DiffResponse,
  EvalConfig,
  EvalEvent,
  EvalResults,
  FailureResponse,
  GenerationConfig,
  GenerationEvent,
  GenerationRunDetail,
  GenerationRunSummary,
  GenerationSummary,
  ModelName,
  QueryItem,
  QueryStats,
  QuestionType,
  RunDetail,
  RunSummary,
  StatTestResponse,
} from "./types";

export type { EvalConfig, GenerationConfig } from "./types";

export async function fetchQueries(): Promise<QueryItem[]> {
  const r = await fetch("/api/queries");
  if (!r.ok) throw new Error(`queries failed: ${r.status}`);
  return r.json();
}

export async function fetchQueryStats(): Promise<QueryStats> {
  const r = await fetch("/api/queries/stats");
  if (!r.ok) throw new Error(`query stats failed: ${r.status}`);
  return r.json();
}

export interface ChatRequestBody {
  message: string;
  model: ModelName;
  top_k: number;
  history: ChatTurn[];
  query_id?: string | null;
  generate?: boolean;
}

export async function chat(body: ChatRequestBody): Promise<ChatResponse> {
  const r = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...body, generate: body.generate ?? true }),
  });
  if (!r.ok) {
    const text = await r.text();
    throw new Error(`chat failed: ${r.status} ${text}`);
  }
  return r.json();
}

export async function fetchEvalCache(): Promise<{ cached: boolean; data?: EvalResults }> {
  const r = await fetch("/api/eval/cache");
  if (!r.ok) throw new Error(`eval cache failed: ${r.status}`);
  return r.json();
}

export async function fetchRuns(): Promise<RunSummary[]> {
  const r = await fetch("/api/eval/runs");
  if (!r.ok) throw new Error(`runs failed: ${r.status}`);
  const j = await r.json();
  return j.runs as RunSummary[];
}

export async function fetchRun(runId: string): Promise<RunDetail> {
  const r = await fetch(`/api/eval/runs/${encodeURIComponent(runId)}`);
  if (!r.ok) throw new Error(`run ${runId} failed: ${r.status}`);
  return r.json();
}

export async function deleteRun(runId: string): Promise<void> {
  const r = await fetch(`/api/eval/runs/${encodeURIComponent(runId)}`, {
    method: "DELETE",
  });
  if (!r.ok) throw new Error(`delete run ${runId} failed: ${r.status}`);
}

export async function patchRunComment(runId: string, comment: string): Promise<void> {
  const r = await fetch(`/api/eval/runs/${encodeURIComponent(runId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ comment }),
  });
  if (!r.ok) throw new Error(`patch run ${runId} failed: ${r.status}`);
}

export async function fetchRunDiff(runId: string, baseline: string): Promise<DiffResponse> {
  const r = await fetch(
    `/api/eval/runs/${encodeURIComponent(runId)}/diff?baseline=${encodeURIComponent(baseline)}`,
  );
  if (!r.ok) throw new Error(`diff ${runId} vs ${baseline} failed: ${r.status}`);
  return r.json();
}

export function openEvalSocket(
  cfg: EvalConfig,
  onEvent: (e: EvalEvent) => void,
): WebSocket {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${window.location.host}/api/eval/run`);
  ws.onopen = () => ws.send(JSON.stringify(cfg));
  ws.onmessage = (msg) => {
    try {
      onEvent(JSON.parse(msg.data) as EvalEvent);
    } catch (err) {
      console.error("eval socket parse error", err);
    }
  };
  return ws;
}

// ---- generation ----

export async function fetchGenRuns(
  retrievalRunId?: string,
): Promise<GenerationRunSummary[]> {
  const url = retrievalRunId
    ? `/api/generation/runs?retrieval_run_id=${encodeURIComponent(retrievalRunId)}`
    : "/api/generation/runs";
  const r = await fetch(url);
  if (!r.ok) throw new Error(`gen runs failed: ${r.status}`);
  const j = await r.json();
  return j.runs as GenerationRunSummary[];
}

export async function fetchGenRun(runId: string): Promise<GenerationRunDetail> {
  const r = await fetch(`/api/generation/runs/${encodeURIComponent(runId)}`);
  if (!r.ok) throw new Error(`gen run ${runId} failed: ${r.status}`);
  return r.json();
}

export async function deleteGenRun(runId: string): Promise<void> {
  const r = await fetch(`/api/generation/runs/${encodeURIComponent(runId)}`, {
    method: "DELETE",
  });
  if (!r.ok) throw new Error(`delete gen run ${runId} failed: ${r.status}`);
}

export async function patchGenRunComment(runId: string, comment: string): Promise<void> {
  const r = await fetch(`/api/generation/runs/${encodeURIComponent(runId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ comment }),
  });
  if (!r.ok) throw new Error(`patch gen run ${runId} failed: ${r.status}`);
}

export async function patchGenRunStarred(runId: string, starred: boolean): Promise<void> {
  const r = await fetch(`/api/generation/runs/${encodeURIComponent(runId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ starred }),
  });
  if (!r.ok) throw new Error(`patch gen run ${runId} failed: ${r.status}`);
}

export async function fetchGenSummary(): Promise<GenerationSummary> {
  const r = await fetch("/api/generation/summary");
  if (!r.ok) throw new Error(`gen summary failed: ${r.status}`);
  return r.json();
}

// ---- statistical testing ----

export async function runStatTest(
  run_a: string,
  run_b: string,
  qtypes?: QuestionType[],
): Promise<StatTestResponse> {
  const r = await fetch("/api/statistical/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_a, run_b, qtypes: qtypes ?? null }),
  });
  if (!r.ok) throw new Error(`stat test failed: ${r.status} ${await r.text()}`);
  return r.json();
}

// ---- failure analysis ----

export async function runFailureAnalysis(
  run_a: string,
  run_b: string,
): Promise<FailureResponse> {
  const r = await fetch("/api/failure/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_a, run_b }),
  });
  if (!r.ok) throw new Error(`failure analysis failed: ${r.status} ${await r.text()}`);
  return r.json();
}

export function openGenSocket(
  cfg: GenerationConfig,
  onEvent: (e: GenerationEvent) => void,
): WebSocket {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${window.location.host}/api/generation/run`);
  ws.onopen = () => ws.send(JSON.stringify(cfg));
  ws.onmessage = (msg) => {
    try {
      onEvent(JSON.parse(msg.data) as GenerationEvent);
    } catch (err) {
      console.error("gen socket parse error", err);
    }
  };
  return ws;
}
