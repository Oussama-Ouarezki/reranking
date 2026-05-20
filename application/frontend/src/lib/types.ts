export type ModelName =
  | "bm25"
  | "monot5"
  | "duot5"
  | "duot5_rolling"
  | "lit5"
  | "mono_duo"
  // | "monot5_lit5"
  // | "mono_uncertain_duo_lit5"
  // | "mono_dynamic_duo_lit5"
  // | "mono_gated_duo"
  // | "mono_proximity_duo"
  // | "mono_proximity_duo_lit5"
  // | "lit5_duo"
  // | "mono_proximity_duo_0005"
  // | "mono_proximity_duo_005_top30"
  // | "mono_mau_duo_low_cost"
  // | "mono_mau_duo_pareto"
  // | "mono_gated_lit5_top20"
  // | "mono_gated_lit5_top40"
  // | "mono_gated_lit5_top50"
  // | "bge_v2_m3"
  | "qwen3_reranker_4b"
  | "qwen3_reranker_0_6b"
  // | "rank_zephyr"
  // | "mono_entropy_gated_duo"
  | "lit5_finetuned"
  | "lit5_bioasq_lora"
  | "lit5_bioasq_lora_e1"
  | "lit5_bioasq_lora_e3"
  | "lit5_bioasq_lora_kaggle"
  | "lit5_bioasq_lora_kaggle_e1"
  | "lit5_bioasq_lora_kaggle_e2"
  | "lit5_bioasq_lora_kaggle_e3"
  | "lit5_bioasq_lora_kaggle_e4"
  // | "mono_entropy_h50_lit5"
  | "mono_entropy_h50_duo"
  | "qwen4b_linear_fusion"
  | "qwen4b_linear_fusion_dynamic"
  | "qwen4b_linear_fusion_dynamic_10"
  | "qwen4b_linear_fusion_dynamic_gated"
  | "qwen06b_lf"
  | "qwen06b_lf_999"
  | "qwen06b_lf_999_lit5"
  | "qwen06b_lf_999_duot5_unc_lit5"
  | "qwen06b_lf_lit5"
  | "qwen06b_lf_duot5_unc_lit5"
  | "biobert"
  | "bm25_biobert"
  | "deepseek";
export type QuestionType = "factoid" | "list" | "yesno" | "summary";

export interface QueryItem {
  id: string;
  text: string;
  type: QuestionType | null;
  has_qrels: boolean;
}

export interface RetrievedDoc {
  rank: number;
  docid: string;
  title: string;
  snippet: string;
  score: number;
  corpus_type: string | null;
  is_relevant?: boolean;
}

export interface Timings {
  retrieve_s: number;
  rerank_s: number;
  generate_s: number;
  total_s: number;
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  answer: string | null;
  retrieved: RetrievedDoc[];
  question_type: QuestionType | null;
  top_k: number;
  model: ModelName;
  no_relevant: boolean;
  metrics: RankingMetrics | null;
  timings?: Timings | null;
  n_relevant_retrieved?: number;
  qa_score?: number | null;
  qa_metric_label?: string | null;
  qa_extra?: Record<string, unknown> | null;
}

export interface RankingMetrics {
  ndcg_at: Record<number, number>;
  mrr_at: Record<number, number>;
  p_at: Record<number, number>;
  r_at: Record<number, number>;
  map_at?: Record<number, number>;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  retrieved?: RetrievedDoc[];
  question_type?: QuestionType | null;
  no_relevant?: boolean;
  model?: ModelName;
  loading?: boolean;
  metrics?: RankingMetrics | null;
  timings?: Timings | null;
  n_relevant_retrieved?: number;
  qa_score?: number | null;
  qa_metric_label?: string | null;
  qa_extra?: Record<string, unknown> | null;
}

export interface AggregateMetrics {
  ndcg_at: Record<number, number>;
  mrr_at: Record<number, number>;
  p_at: Record<number, number>;
  r_at: Record<number, number>;
  map_at: Record<number, number>;
}

// ----- retrieval (eval router) -----

export type Bm25InjectMode = "off" | "raw" | "norm" | "bucket";

export interface EvalConfig {
  models: ModelName[];
  n_questions?: number | null;
  comment?: string;
  bm25_inject_mode?: Bm25InjectMode;
}

export interface EvalModelResult {
  aggregate: AggregateMetrics;
  elapsed_s: number;
  run_id?: string;
}

export interface PerQueryEntry {
  metrics?: RankingMetrics;
  qtype?: QuestionType | null;
  top_docids?: string[];
}

export interface RunConfig {
  save_topn: number;
  bm25_retrieve_k?: number;
  n_questions?: number;
  sampled_qids?: string[] | null;
  seed?: number | null;
}

export interface RunSummary {
  run_id: string;
  model: ModelName;
  started_at: number;
  ended_at: number;
  elapsed_s: number;
  config: RunConfig;
  n_queries: number;
  comment?: string;
}

export interface RunDetail {
  run_id: string;
  model: ModelName;
  started_at: number;
  ended_at: number;
  elapsed_s: number;
  config: RunConfig;
  comment?: string;
  aggregate: AggregateMetrics;
  per_query: Record<string, PerQueryEntry>;
}

export type MetricKey = "ndcg_at" | "mrr_at" | "p_at" | "r_at" | "map_at";

export interface DiffSide {
  ndcg_at: Record<number, number>;
  mrr_at: Record<number, number>;
  p_at: Record<number, number>;
  r_at: Record<number, number>;
  map_at: Record<number, number>;
}

export interface DiffBlock {
  a: Partial<DiffSide>;
  b: Partial<DiffSide>;
  delta: Partial<DiffSide>;
}

export interface DiffResponse {
  run_id: string;
  baseline: string;
  n_overlapping: number;
  global: DiffBlock;
  by_qtype: Partial<Record<QuestionType, DiffBlock>>;
}

export interface EvalResults {
  started_at: number;
  ended_at: number;
  config: { models: ModelName[]; save_topn: number };
  per_model: Partial<Record<ModelName, EvalModelResult>>;
  saved_run_ids: string[];
}

export type EvalEvent =
  | { type: "progress"; model: ModelName; current: number; total: number }
  | {
      type: "model_done";
      model: ModelName;
      aggregate: AggregateMetrics;
      elapsed_s: number;
      run_id: string;
    }
  | { type: "done"; results: EvalResults }
  | { type: "error"; message: string; model?: ModelName; qid?: string };

// ----- generation runs -----

export interface GenerationConfig {
  retrieval_run_id: string;
  k_values: number[];
  qtypes?: QuestionType[] | null;
  comment?: string;
  skip_judge?: boolean;
}

export interface SpearmanMatrix {
  variables: string[];
  matrix: number[][];
  n: number;
}

export interface YesnoExtraMetrics {
  pred_label?: string | null;
  gold_label?: string | null;
}

export interface SummaryExtraMetrics {
  rouge_l?: number | null;
  bert_score?: number | null;
}

export interface FactoidExtraMetrics {
  strict_acc?: number | null;
}

export interface ListExtraMetrics {
  map?: number | null;
}

export type ExtraMetrics = YesnoExtraMetrics &
  SummaryExtraMetrics &
  FactoidExtraMetrics &
  ListExtraMetrics;

export interface GenerationPerQuery {
  qid: string;
  qtype: QuestionType;
  question: string;
  answer: string;
  qa_score: number | null;
  extra_metrics?: ExtraMetrics | null;
  retrieval_metrics: {
    ndcg: number;
    p: number;
    r: number;
    mrr: number;
    map: number;
  } | null;
  top_docids: string[];
}

export interface GenerationAggregate {
  by_qtype: Partial<Record<QuestionType, number>>;
  n_per_qtype: Partial<Record<QuestionType, number>>;
  extra_by_qtype?: {
    yesno?: { macro_f1?: number };
    summary?: { rouge_l?: number; bert_score?: number };
    factoid?: { strict_acc?: number };
    list?: { map?: number };
  };
}

export interface GenerationRunSummary {
  run_id: string;
  retrieval_run_id: string;
  retrieval_model?: ModelName;
  k: number;
  started_at: number;
  ended_at: number;
  elapsed_s: number;
  n_queries: number;
  by_qtype: Partial<Record<QuestionType, number>> | null;
  n_per_qtype: Partial<Record<QuestionType, number>> | null;
  comment?: string;
  starred?: boolean;
}

export interface GenerationRunDetail {
  run_id: string;
  retrieval_run_id: string;
  retrieval_model: ModelName;
  k: number;
  started_at: number;
  ended_at: number;
  elapsed_s: number;
  config: {
    qtypes: QuestionType[] | null;
    num_ctx: number;
    skip_judge?: boolean;
    stateless?: boolean;
  };
  comment?: string;
  per_query: Record<string, GenerationPerQuery>;
  aggregate: GenerationAggregate;
  correlations: Partial<Record<QuestionType, SpearmanMatrix>>;
  extra_correlations?: Partial<Record<string, SpearmanMatrix>>;
}

export interface GenerationSummaryCell {
  run_id: string;
  n_queries: number;
  n_evaluated?: number;  // queries with both qrels and a type — the consistent set
  qa_overall: number | null;
  qa_by_qtype: Partial<Record<QuestionType, number>>;
  n_per_qtype?: Partial<Record<QuestionType, number>>;
  retrieval: Partial<Record<"ndcg" | "p" | "r" | "mrr" | "map", number | null>>;
  summary_rouge_l?: number | null;
  summary_bert_score?: number | null;
  factoid_strict_acc?: number | null;
  list_map?: number | null;
  skip_judge?: boolean;
  elapsed_s: number;
  comment?: string;
}

export interface GenerationSummaryRow {
  retrieval_run_id: string;
  retrieval_model: ModelName | string;
  retrieval_comment?: string;
  cells: Record<string, GenerationSummaryCell>;
}

export interface GenerationSummary {
  ks: number[];
  rows: GenerationSummaryRow[];
}

export type GenerationEvent =
  | { type: "progress"; k: number; current: number; total: number }
  | {
      type: "k_done";
      k: number;
      elapsed_s: number;
      run_id: string;
      aggregate: GenerationAggregate;
    }
  | {
      type: "done";
      started_at: number;
      ended_at: number;
      saved_run_ids: string[];
    }
  | { type: "error"; message: string; k?: number; qid?: string };

// ----- query-set stats -----

export interface QueryStats {
  total: number;
  with_type: number;
  with_qrels: number;
  evaluated: number;  // intersection — the consistent denominator
  excluded_ids: string[];
}

// ----- statistical testing -----

export interface StatBlock {
  n_pairs: number;
  n_a_wins: number;
  n_b_wins: number;
  n_ties: number;
  mean_a: number;
  mean_b: number;
  mean_delta: number;
  median_delta: number;
  w_plus: number;
  w_minus: number;
  statistic: number;
  z: number;
  p_value: number;
  significant: boolean;
}

export interface StatTestResponse {
  run_a: string;
  run_b: string;
  retrieval_model_a?: string | null;
  retrieval_model_b?: string | null;
  k_a?: number | null;
  k_b?: number | null;
  metric_labels: Record<string, string>;
  by_qtype: Partial<Record<QuestionType, StatBlock>>;
  global_block: StatBlock;
}

// ----- failure analysis -----

export interface RetrievalMetricSet {
  ndcg?: number;
  p?: number;
  r?: number;
  mrr?: number;
  map?: number;
}

export interface FailureRecord {
  qid: string;
  qtype: QuestionType;
  question: string;
  score_a: number;
  score_b: number;
  delta: number;
  failed_model: "a" | "b";
  threshold: number;
  answer_a?: string | null;
  answer_b?: string | null;
  retrieval_a?: RetrievalMetricSet | null;
  retrieval_b?: RetrievalMetricSet | null;
  retrieval_delta?: RetrievalMetricSet | null;
}

export interface FailureBucket {
  qtype: QuestionType;
  metric_label: string;
  threshold: number;
  n_pairs: number;
  n_failures: number;
  n_a_failed: number;
  n_b_failed: number;
  failures: FailureRecord[];
}

export interface FailureResponse {
  run_a: string;
  run_b: string;
  retrieval_model_a?: string | null;
  retrieval_model_b?: string | null;
  k_a?: number | null;
  k_b?: number | null;
  n_overlapping: number;
  total_failures: number;
  total_a_failed: number;
  total_b_failed: number;
  by_qtype: FailureBucket[];
}
