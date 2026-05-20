import type { ModelName } from "../lib/types";

interface Props {
  model: ModelName;
  onModelChange: (m: ModelName) => void;
}

const MODELS: { name: ModelName; label: string; description: string }[] = [
  {
    name: "bm25",
    label: "BM25",
    description: "First-stage only (Pyserini Lucene, k1=0.7, b=0.9). Top 50.",
  },
  {
    name: "monot5",
    label: "BM25 → monoT5",
    description: "BM25 top 50 → monoT5 pointwise (T5 P(true)). FP16.",
  },
  {
    name: "duot5",
    label: "BM25 → duoT5",
    description: "BM25 top 50 → duoT5 pairwise tournament on top 20. FP16.",
  },
  {
    name: "duot5_rolling",
    label: "BM25 → duoT5 (rolling window 20, stride 10)",
    description:
      "BM25 top 50 → duoT5 pairwise tournament inside a sliding window of 20 docs, stride 10. Win-probabilities aggregated across overlapping windows then averaged. FP16.",
  },
  {
    name: "mono_duo",
    label: "BM25 → monoT5 → duoT5",
    description:
      "Cascade: BM25 top 50 → monoT5 narrows to 20 → duoT5 tournament on those 20. FP16.",
  },
  // {
  //   name: "monot5_lit5",
  //   label: "BM25 → monoT5 (≥0.7) → LiT5",
  //   description:
  //     "Cascade: BM25 top 50 → monoT5 filters at P(true) ≥ 0.7 → LiT5 listwise reranks survivors. Docs below threshold kept in mono order below. FP16.",
  // },
  // {
  //   name: "mono_uncertain_duo_lit5",
  //   label: "BM25 → monoT5 → duoT5 (pos 15-25) → LiT5",
  //   description:
  //     "Cascade: BM25 top 50 → monoT5 ranks all → duoT5 reorders uncertain zone (positions 15–25) → LiT5 listwise on top-20. FP16.",
  // },
  // {
  //   name: "mono_dynamic_duo_lit5",
  //   label: "BM25 → monoT5 → duoT5 (dynamic margin) → LiT5",
  //   description:
  //     "Cascade: BM25 top 50 → monoT5 ranks all → duoT5 reorders adjacent pairs with score gap < 0.05 → LiT5 listwise on top-20. FP16.",
  // },
  // {
  //   name: "mono_gated_duo",
  //   label: "BM25 → monoT5 → duoT5 (gated τ=0.001)",
  //   description:
  //     "Cascade: BM25 top 50 → monoT5 ranks all → duoT5 tournament on top-20 only when top-1/top-2 score gap < 0.001 (Pareto knee). Skips duoT5 for ~50% of queries. FP16.",
  // },
  // {
  //   name: "mono_proximity_duo",
  //   label: "BM25 → monoT5 → duoT5 (proximity 0.001)",
  //   description:
  //     "Cascade: BM25 top 50 → monoT5 ranks all → any two docs in top-20 with P(true) gap < 0.001 are both flagged → duoT5 reorders flagged docs only → slot-preserving merge. FP16.",
  // },
  // {
  //   name: "mono_proximity_duo_lit5",
  //   label: "BM25 → monoT5 → duoT5 (proximity 0.001) → LiT5",
  //   description:
  //     "Cascade: proximity 0.001 duoT5 on uncertain docs in top-20 → slot-preserving merge → LiT5 listwise final pass on top-20. FP16.",
  // },
  // {
  //   name: "lit5_duo",
  //   label: "BM25 → LiT5 → duoT5 (top-10)",
  //   description:
  //     "Cascade: BM25 top 50 → LiT5 listwise sliding window rerank → duoT5 tournament on top-10 of LiT5 output. FP16.",
  // },
  // {
  //   name: "mono_proximity_duo_0005",
  //   label: "BM25 → monoT5 → duoT5 (proximity 0.0005)",
  //   description:
  //     "Cascade: same as proximity 0.001 but with tighter margin 0.0005 — flags fewer but more confidently uncertain doc pairs. FP16.",
  // },
  // {
  //   name: "mono_proximity_duo_005_top30",
  //   label: "BM25 → monoT5 → duoT5 (proximity 0.005, top-30)",
  //   description:
  //     "Cascade: BM25 top 50 → monoT5 ranks all → all-pairs proximity check in top-30 with margin 0.005 → duoT5 on flagged docs → slot-preserving merge. FP16.",
  // },
  // {
  //   name: "mono_mau_duo_low_cost",
  //   label: "BM25 → monoT5 → duoT5 (MAU τ=0.0001, low-cost)",
  //   description:
  //     "MAU sweep — gap-gated duoT5 at τ=0.0001: routes only the most uncertain ~21% of queries to duoT5, yielding a big nDCG jump (+0.011 vs monoT5) at low compute cost. FP16.",
  // },
  // {
  //   name: "mono_mau_duo_pareto",
  //   label: "BM25 → monoT5 → duoT5 (MAU τ=0.001, Pareto knee)",
  //   description:
  //     "MAU sweep — gap-gated duoT5 at τ=0.001: Pareto knee of the threshold sweep. Routes ~50% of queries to duoT5, recovering ~95% of achievable nDCG gain at half the cost of always running duoT5. FP16.",
  // },
  // {
  //   name: "mono_gated_lit5_top20",
  //   label: "BM25 → monoT5 → LiT5 top-20 (gated τ=0.001)",
  //   description:
  //     "Gap-gated LiT5: monoT5 ranks all 50 → if top-1/top-2 gap < 0.001 (uncertain), LiT5 listwise reranks the top-20 docs; otherwise monoT5 order is kept. FP16.",
  // },
  // {
  //   name: "mono_gated_lit5_top40",
  //   label: "BM25 → monoT5 → LiT5 top-40 (gated τ=0.001)",
  //   description:
  //     "Gap-gated LiT5: same gate as top-20 variant but LiT5 sees top-40 docs when triggered — deeper listwise pass at higher cost. FP16.",
  // },
  // {
  //   name: "mono_gated_lit5_top50",
  //   label: "BM25 → monoT5 → LiT5 top-50 (gated τ=0.001)",
  //   description:
  //     "Gap-gated LiT5: same gate but LiT5 sees all 50 BM25 candidates when triggered — full listwise rerank of the entire retrieved set. FP16.",
  // },
  {
    name: "lit5",
    label: "BM25 → LiT5",
    description: "BM25 top 50 → LiT5 listwise sliding window (size 20, stride 10). FP16.",
  },
  // {
  //   name: "bge_v2_m3",
  //   label: "BM25 → BGE-reranker-v2-m3",
  //   description: "BM25 top 50 → BAAI/bge-reranker-v2-m3 cross-encoder. FP16.",
  // },
  {
    name: "bm25_biobert",
    label: "BM25 → BioBERT",
    description:
      "BM25 top 50 → nboost/pt-biobert-base-msmarco cross-encoder. BioBERT fine-tuned on MS-MARCO passage ranking. FP16.",
  },
  {
    name: "qwen3_reranker_4b",
    label: "BM25 → Qwen3-Reranker-4B",
    description: "BM25 top 50 → Qwen3-Reranker-4B yes/no relevance. FP16.",
  },
  {
    name: "qwen3_reranker_0_6b",
    label: "BM25 → Qwen3-0.6B (pure)",
    description:
      "BM25 top-50 → Qwen3-Reranker-0.6B yes/no relevance, sorted by P(yes). No linear fusion, no further reranker. FP16. Test-set: nDCG@1 0.9315, nDCG@5 0.9094, nDCG@10 0.8880, MRR@10 0.9578.",
  },
  // {
  //   name: "rank_zephyr",
  //   label: "BM25 → RankZephyr (Q4)",
  //   description:
  //     "BM25 top 50 → RankZephyr 7B listwise sliding window via llama-cpp-python. Q4_K_M GGUF.",
  // },
  // {
  //   name: "mono_entropy_gated_duo",
  //   label: "BM25 → monoT5 → duoT5 (entropy H@20, τ=0.95)",
  //   description:
  //     "Entropy-gated cascade: BM25 top 50 → monoT5 ranks all → normalized rank-distribution entropy H@20 ≥ 0.95 triggers duoT5 tournament on top-20; otherwise monoT5 order kept. Saves ~44% of duoT5 calls vs always-duo. FP16.",
  // },
  {
    name: "lit5_finetuned",
    label: "BM25 → LiT5 (fine-tuned)",
    description:
      "BM25 top 50 → LiT5-Distill fine-tuned on BioASQ triples, sliding window (size 20, stride 10). FP16.",
  },
  {
    name: "lit5_bioasq_lora",
    label: "BM25 → LiT5 (LoRA fine-tuned, epoch 2)",
    description:
      "BM25 top 50 → LiT5-Distill base + LoRA adapter fine-tuned on BioASQ (r=16, α=16, dropout=0.1; epoch 2). LoRA weights merged at load time. Sliding window (size 20, stride 10). FP16.",
  },
  {
    name: "lit5_bioasq_lora_e1",
    label: "BM25 → LiT5 (LoRA q,v, epoch 1)",
    description:
      "BM25 top 50 → LiT5-Distill base + LoRA adapter fine-tuned on BioASQ (r=8, α=16, dropout=0.05, targets={q,v}; epoch 1). LoRA weights merged at load time. Sliding window (size 20, stride 10). FP16.",
  },
  {
    name: "lit5_bioasq_lora_e3",
    label: "BM25 → LiT5 (LoRA q,v, epoch 3)",
    description:
      "BM25 top 50 → LiT5-Distill base + LoRA adapter fine-tuned on BioASQ (r=8, α=16, dropout=0.05, targets={q,v}; epoch 3). LoRA weights merged at load time. Sliding window (size 20, stride 10). FP16.",
  },
  {
    name: "lit5_bioasq_lora_kaggle",
    label: "BM25 → LiT5 (LoRA q,k,v,o, Kaggle)",
    description:
      "BM25 top 50 → LiT5-Distill base + LoRA adapter fine-tuned on Kaggle (LR=5e-6, 4 epochs from start_epoch=2, 2400 windows; r=4, α=8, dropout=0.0, targets={q,k,v,o}). LoRA weights merged at load time. Sliding window (size 20, stride 10). FP16.",
  },
  {
    name: "lit5_bioasq_lora_kaggle_e1",
    label: "BM25 → LiT5 (LoRA Kaggle, epoch 1)",
    description:
      "Kaggle LoRA checkpoint at epoch 1 (train_loss=9.5696, nDCG@10=0.7347). Same config as Kaggle base: r=4, α=8, dropout=0.0, targets={q,k,v,o}, LR=5e-6. Sliding window (20/10). FP16.",
  },
  {
    name: "lit5_bioasq_lora_kaggle_e2",
    label: "BM25 → LiT5 (LoRA Kaggle, epoch 2)",
    description:
      "Kaggle LoRA checkpoint at epoch 2 (train_loss=9.5461, nDCG@10=0.7364). Same config as Kaggle base: r=4, α=8, dropout=0.0, targets={q,k,v,o}, LR=5e-6. Sliding window (20/10). FP16.",
  },
  {
    name: "lit5_bioasq_lora_kaggle_e3",
    label: "BM25 → LiT5 (LoRA Kaggle, epoch 3)",
    description:
      "Kaggle LoRA checkpoint at epoch 3 (train_loss=9.5302, nDCG@10=0.7495). Same config as Kaggle base: r=4, α=8, dropout=0.0, targets={q,k,v,o}, LR=5e-6. Sliding window (20/10). FP16.",
  },
  {
    name: "lit5_bioasq_lora_kaggle_e4",
    label: "BM25 → LiT5 (LoRA Kaggle, epoch 4)",
    description:
      "Kaggle LoRA checkpoint at epoch 4 (train_loss=9.4731, nDCG@10=0.7473). Same config as Kaggle base: r=4, α=8, dropout=0.0, targets={q,k,v,o}, LR=5e-6. Sliding window (20/10). FP16.",
  },
  {
    name: "mono_entropy_h50_duo",
    label: "BM25 → monoT5 → duoT5 (entropy H@50, τ=0.832)",
    description:
      "Cascade: BM25 top 50 → monoT5 ranks all → normalized rank-distribution entropy H@50 ≥ 0.832 triggers duoT5 tournament on top-20; otherwise monoT5 order kept. Pareto knee: nDCG@10=0.8854, saves 41.2% of duoT5 calls. FP16.",
  },
  // {
  //   name: "mono_entropy_h50_lit5",
  //   label: "BM25 → monoT5 → LiT5 (entropy H@50, τ=0.839)",
  //   description:
  //     "Best cascade from grid search: monoT5 ranks all 50 → normalized rank-distribution entropy H@50 ≥ 0.839 triggers LiT5 listwise on all 50 docs; otherwise monoT5 order kept. nDCG@10=0.866, saves 42% of LiT5 calls. FP16.",
  // },
  {
    name: "qwen4b_linear_fusion",
    label: "BM25 → Qwen3-4B + BM25 (linear, α=0.825)",
    description:
      "Linear fusion (no gate): Qwen3-Reranker-4B scores BM25 top 50, then re-rank by α·qwen_minmax + (1−α)·bm25_minmax with α=0.825 for every query. α chosen to maximise GLOBAL nDCG@10 on a 500-query BioASQ slice (+0.0094 vs Qwen alone). FP16.",
  },
  {
    name: "qwen4b_linear_fusion_dynamic",
    label: "BM25 → Qwen3-4B + BM25 (linear, dynamic α per qtype)",
    description:
      "Linear fusion with per-question-type α optimised for each type's natural metric: list→nDCG@3 (α=0.875), summary→nDCG@10 (α=0.800), yesno→nDCG@1 (α=0.750), factoid→nDCG@5 (α=0.875). Falls back to α=0.825 if qtype unknown. FP16.",
  },
  {
    name: "qwen4b_linear_fusion_dynamic_10",
    label: "BM25 → Qwen3-4B + BM25 (linear, dynamic α, all → nDCG@10)",
    description:
      "Linear fusion with per-question-type α — every type optimised for nDCG@10: list α=0.875, summary α=0.800, yesno α=0.750, factoid α=0.925. Differs from the mixed-target dynamic only on factoid (0.875 → 0.925). FP16.",
  },
  {
    name: "qwen4b_linear_fusion_dynamic_gated",
    label: "BM25 → Qwen3-4B + BM25 (linear, dynamic α, gated H@20)",
    description:
      "Per-question-type (α, τ) linear fusion gated on H@20 entropy of Qwen P(yes). For each query: if H@20 > τ → fuse with α, else keep pure Qwen. list (α=0.875, τ=0.8242, ~86% fused), summary (α=0.800, τ=0.4246, ~96% fused), yesno (α=0.750, τ=0.6494, ~98% fused), factoid (α=0.875, τ=0, always fuse). FP16.",
  },
  {
    name: "qwen06b_lf",
    label: "BM25 → Qwen3-0.6B + BM25 (linear, dyn α)",
    description:
      "BM25 top-50 → Qwen3-Reranker-0.6B P(yes) → linear fusion with BM25 using per-type α (summary/factoid/list 0.99, yesno 0.85, Recall@20-tuned). No further reranker. Test-set: nDCG@1 0.9077, nDCG@5 0.8977, nDCG@10 0.8810, MRR@10 0.9454.",
  },
  {
    name: "qwen06b_lf_999",
    label: "BM25 → Qwen3-0.6B + BM25 (linear, α=0.999)",
    description:
      "BM25 top-50 → Qwen3-Reranker-0.6B P(yes) → linear fusion with BM25 using uniform α=0.999 across all query types. Score = 0.999·qwen_minmax + 0.001·bm25_minmax (per-query min-max norm). α selected by fine sweep on BioASQ Task13BGoldenEnriched targeting global nDCG@10 — tiny BM25 nudge breaks ties in Qwen's score distribution. Test-set: nDCG@1 0.9256, nDCG@5 0.9104, nDCG@10 0.8896, MRR@10 0.9549.",
  },
  {
    name: "qwen06b_lf_999_lit5",
    label: "BM25 → Qwen3-0.6B + BM25 (LF, α=0.999) → LiT5 top-20",
    description:
      "BM25 top-50 → Qwen3-Reranker-0.6B P(yes) → linear fusion with BM25 (uniform α=0.999) → LiT5-Distill listwise reranks the top-20 in a single 20-passage window; positions 21+ kept in LF order. Test-set: nDCG@1 0.9464, nDCG@5 0.9159, nDCG@10 0.8922, MRR@10 0.9656.",
  },
  {
    name: "qwen06b_lf_999_duot5_unc_lit5",
    label: "BM25 → Qwen3-0.6B + BM25 (LF, α=0.999) → duoT5(15-25) → LiT5 top-20",
    description:
      "BM25 top-50 → Qwen3-Reranker-0.6B P(yes) → linear fusion with BM25 (uniform α=0.999) → duoT5 tournament on LF positions 15-25 (uncertainty band) — top-6 promoted into the head, forming a new top-20 — then LiT5-Distill listwise rerank. Test-set: nDCG@1 0.9524, nDCG@5 0.9195, nDCG@10 0.8939, MRR@10 0.9690 — best mid-rank metrics among all α=0.999 variants.",
  },
  {
    name: "qwen06b_lf_lit5",
    label: "BM25 → Qwen3-0.6B + BM25 (linear, dyn α) → LiT5 top-20",
    description:
      "BM25 top-50 → Qwen3-Reranker-0.6B P(yes) → linear fusion with BM25 using per-type α (summary/factoid/list 0.99, yesno 0.85, Recall@20-tuned) → LiT5-Distill listwise reranks the top-20 in a single 20-passage window; positions 21+ kept in LF order. Test-set: nDCG@1 0.9494, MRR@10 0.9669.",
  },
  {
    name: "qwen06b_lf_duot5_unc_lit5",
    label: "BM25 → Qwen3-0.6B + BM25 → duoT5(15-25) → LiT5 top-20  ★",
    description:
      "Best cascade on BioASQ Task13BGoldenEnriched. Same LF as qwen06b_lf_lit5, then duoT5 tournament on the LF positions 15-25 (uncertainty band) — top-6 promoted into the head, forming a new top-20 — then LiT5-Distill listwise rerank. Test-set: nDCG@1 0.9583, nDCG@5 0.9170, nDCG@10 0.8913, MRR@10 0.9715.",
  },
];

const SCRIPT_VARIANTS = [
  {
    name: "deepseek",
    description:
      "Run scripts/rerank_deepseek_zeroshot.py to produce a deepseek run; appears on the Dashboard once produced.",
  },
];

export default function ModelSidebar({ model, onModelChange }: Props) {
  return (
    <aside className="w-72 shrink-0 h-full flex flex-col border-l border-border bg-panel">
      <div className="px-3 py-2.5 border-b border-border bg-bg/40">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-muted">
          Reranker
        </h2>
      </div>
      <div className="flex-1 overflow-y-auto">
        {MODELS.map((m) => {
          const active = model === m.name;
          return (
            <label
              key={m.name}
              className={`block px-3 py-2.5 border-b border-border/50 cursor-pointer transition-colors duration-100 hover:bg-bg ${
                active ? "bg-bg border-l-2 border-l-accent pl-[10px]" : ""
              }`}
            >
              <div className="flex items-start gap-2">
                <input
                  type="radio"
                  name="reranker"
                  className="mt-1 accent-[#cc785c]"
                  checked={active}
                  onChange={() => onModelChange(m.name)}
                />
                <div className="flex-1">
                  <div className={`text-sm font-medium ${active ? "text-accent" : ""}`}>{m.label}</div>
                  <div className="text-xs text-muted leading-snug mt-0.5">{m.description}</div>
                </div>
              </div>
            </label>
          );
        })}

        <div className="px-3 py-2 border-t border-border bg-bg/40">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">
            Script-only models
          </h3>
        </div>
        {SCRIPT_VARIANTS.map((m) => (
          <div
            key={m.name}
            className="block px-3 py-2.5 border-b border-border/50 opacity-50"
          >
            <div className="flex items-start gap-2">
              <input type="radio" disabled className="mt-1" />
              <div className="flex-1">
                <div className="text-sm font-medium">{m.name}</div>
                <div className="text-xs text-muted leading-snug mt-0.5">{m.description}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}
