"""Generative QA metrics for BioASQ.

Type-aware deterministic metrics for factoid (Exact Match), yesno (Accuracy),
and list (token-level F1). Summary uses an LLM-as-judge that returns TRUE/FALSE
following the frames-style auto-rating pattern (krishna2024frames):

    Accuracy = #TRUE decisions / #total questions

Each scorer returns a float in [0, 1]; ``score_answer`` dispatches on qtype
and pulls gold answers from the query dict produced by
``data/bioasq/raw/Task13BGoldenEnriched/queries_full.jsonl``.
"""

from __future__ import annotations

import re
import string
from collections.abc import Iterable

from ..generation import ollama_client


_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})


def normalize(s: str) -> str:
    s = (s or "").lower().translate(_PUNCT_TABLE)
    toks = [t for t in s.split() if t and t not in _ARTICLES]
    return " ".join(toks)


def tokenize(s: str) -> list[str]:
    return normalize(s).split()


# ---------------------------------------------------------------- factoid


def exact_match(pred: str, gold_groups: list[list[str]]) -> float:
    """1.0 if the prediction matches any synonym in any group, else 0.0."""
    if not pred or not gold_groups:
        return 0.0
    p = normalize(pred)
    if not p:
        return 0.0
    for group in gold_groups:
        for g in group:
            if normalize(g) == p:
                return 1.0
    return 0.0


FACTOID_MAX_CANDIDATES = 5


def factoid_mrr(pred: str, gold_groups: list[list[str]]) -> float:
    """BioASQ official factoid metric.

    Parse up to 5 ranked candidate answers from the prediction (one per line),
    return 1/rank of the first candidate that matches any synonym in any gold
    group. 0.0 if no candidate matches or list is empty.
    """
    if not pred or not gold_groups:
        return 0.0
    candidates = _parse_list_items(pred)[:FACTOID_MAX_CANDIDATES]
    if not candidates:
        return 0.0
    gold_norms: set[str] = set()
    for group in gold_groups:
        for g in group:
            n = normalize(g)
            if n:
                gold_norms.add(n)
    for j, cand in enumerate(candidates, start=1):
        if normalize(cand) in gold_norms:
            return 1.0 / j
    return 0.0


# ---------------------------------------------------------------- yesno


_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
_NO_RE = re.compile(r"\bno\b", re.IGNORECASE)


def yesno_accuracy(pred: str, gold: str) -> float:
    if not pred or not gold:
        return 0.0
    yes_at = _YES_RE.search(pred)
    no_at = _NO_RE.search(pred)
    if yes_at and (not no_at or yes_at.start() < no_at.start()):
        choice = "yes"
    elif no_at:
        choice = "no"
    else:
        return 0.0
    return 1.0 if choice == gold.strip().lower() else 0.0


# ---------------------------------------------------------------- list


def _parse_list_items(pred: str) -> list[str]:
    """Pull '- item' lines (or comma-separated) out of a list-style answer."""
    items: list[str] = []
    for raw in pred.splitlines():
        line = raw.strip()
        if not line:
            continue
        # strip leading bullet/number markers
        line = re.sub(r"^[\-\*•\d]+[\.\)\s]+", "", line).strip()
        if line:
            items.append(line)
    if len(items) <= 1 and "," in pred:
        items = [p.strip() for p in pred.split(",") if p.strip()]
    return items


def _matches_any_group(item: str, gold_groups: list[list[str]]) -> int:
    """Return the index of the gold group this item matches, or -1."""
    n = normalize(item)
    if not n:
        return -1
    for i, group in enumerate(gold_groups):
        for g in group:
            if normalize(g) == n:
                return i
    return -1


def list_f1(pred: str, gold_groups: list[list[str]]) -> float:
    if not gold_groups:
        return 0.0
    items = _parse_list_items(pred)
    if not items:
        return 0.0
    matched_groups: set[int] = set()
    tp = 0
    for it in items:
        gi = _matches_any_group(it, gold_groups)
        if gi >= 0 and gi not in matched_groups:
            matched_groups.add(gi)
            tp += 1
    if tp == 0:
        return 0.0
    precision = tp / len(items)
    recall = tp / len(gold_groups)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------- ROUGE-L (no external deps)


def _lcs_length(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def rouge_l_score(pred: str, ideals: list[str]) -> float:
    """ROUGE-L F1 between pred and the best-matching ideal (token-level LCS)."""
    if not pred or not ideals:
        return 0.0
    pred_toks = tokenize(pred)
    if not pred_toks:
        return 0.0
    best = 0.0
    for ideal in ideals:
        ref_toks = tokenize(ideal)
        if not ref_toks:
            continue
        lcs = _lcs_length(pred_toks, ref_toks)
        p = lcs / len(pred_toks)
        r = lcs / len(ref_toks)
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        best = max(best, f1)
    return round(best, 4)


# ---------------------------------------------------------------- BERTScore (optional)


def bert_score_f1(pred: str, ideals: list[str]) -> float | None:
    """BERTScore F1 (max over ideals). Returns None if bert-score is not installed."""
    if not pred or not ideals:
        return None
    try:
        from bert_score import score as _bs  # type: ignore
    except ImportError:
        return None
    try:
        _, _, f1 = _bs([pred] * len(ideals), ideals, lang="en", verbose=False)
        return round(float(f1.max().item()), 4)
    except Exception:
        return None


# ---------------------------------------------------------------- yesno extras


def extract_yesno_label(pred: str) -> str | None:
    """Extract 'yes' or 'no' from a prediction string (first occurrence wins)."""
    yes_at = _YES_RE.search(pred)
    no_at = _NO_RE.search(pred)
    if yes_at and (not no_at or yes_at.start() < no_at.start()):
        return "yes"
    if no_at:
        return "no"
    return None


def yesno_macro_f1(pairs: list[tuple[str | None, str]]) -> float:
    """Macro-averaged F1 for binary yes/no over (pred_label, gold_label) pairs."""
    if not pairs:
        return 0.0

    def _f1(cls: str) -> float:
        tp = sum(1 for p, g in pairs if p == cls and g == cls)
        fp = sum(1 for p, g in pairs if p == cls and g != cls)
        fn = sum(1 for p, g in pairs if p != cls and g == cls)
        if tp == 0:
            return 0.0
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        re = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return (2 * pr * re / (pr + re)) if (pr + re) > 0 else 0.0

    return round((_f1("yes") + _f1("no")) / 2, 4)


# ---------------------------------------------------------------- summary (LLM-judge)


_JUDGE_SYSTEM = (
    "You are an automatic answer rater for biomedical question answering. "
    "Given a question, the model's predicted answer, and one or more ground "
    "truth answers, decide whether the predicted answer conveys the same "
    "meaning as the ground truth (semantically equivalent or a correct "
    "paraphrase). Respond with exactly one token: TRUE or FALSE. No "
    "explanation, no punctuation."
)


def _build_judge_messages(question: str, pred: str, ideals: list[str]) -> list[dict[str, str]]:
    refs = "\n".join(f"- {x.strip()}" for x in ideals if x and x.strip())
    user = (
        f"Question: {question}\n\n"
        f"Predicted answer: {pred.strip()}\n\n"
        f"Ground truth answer(s):\n{refs}\n\n"
        f"Decision (TRUE or FALSE):"
    )
    return [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def summary_judge_binary(question: str, pred: str, ideals: list[str]) -> float:
    if not pred or not ideals:
        return 0.0
    try:
        out = ollama_client.chat(
            _build_judge_messages(question, pred, ideals),
            temperature=0.0,
            max_tokens=4,
        )
    except Exception:
        return 0.0
    token = (out or "").strip().upper().split()[0] if out else ""
    token = token.strip(".,'\"")
    return 1.0 if token == "TRUE" else 0.0


# ---------------------------------------------------------------- dispatcher


def _coerce_groups(value) -> list[list[str]] | None:
    """BioASQ exact_answer is sometimes list[list[str]], sometimes list[str].

    Normalize to list[list[str]] so each inner group is a synonym set.
    """
    if not value:
        return None
    if isinstance(value, list) and value and isinstance(value[0], list):
        return [[str(x) for x in g] for g in value]
    if isinstance(value, list):
        return [[str(x) for x in value]]
    if isinstance(value, str):
        return [[value]]
    return None


def _coerce_ideals(value) -> list[str] | None:
    if not value:
        return None
    if isinstance(value, list):
        return [str(x) for x in value if x]
    if isinstance(value, str):
        return [value]
    return None


def score_answer(qtype: str, pred: str, query: dict[str, object]) -> float | None:
    """Return a 0..1 score for ``pred`` against gold answers in ``query``.

    Returns None when there is no gold answer to compare against.
    """
    if not pred:
        return None
    qtype = (qtype or "").lower()

    if qtype == "factoid":
        groups = _coerce_groups(query.get("exact_answer"))
        if not groups:
            return None
        return factoid_mrr(pred, groups)

    if qtype == "yesno":
        gold = query.get("exact_answer")
        if isinstance(gold, list) and gold:
            gold = gold[0]
        if not isinstance(gold, str):
            return None
        return yesno_accuracy(pred, gold)

    if qtype == "list":
        groups = _coerce_groups(query.get("exact_answer"))
        if not groups:
            return None
        return list_f1(pred, groups)

    if qtype == "summary":
        ideals = _coerce_ideals(query.get("ideal_answer"))
        if not ideals:
            return None
        question = query.get("text") or query.get("body") or ""
        return summary_judge_binary(str(question), pred, ideals)

    return None


def score_answer_full(
    qtype: str,
    pred: str,
    query: dict[str, object],
    skip_judge: bool = False,
) -> dict[str, object]:
    """Return all available metrics for a qtype as a dict.

    Always includes 'qa_score' (same scalar as score_answer()).
    Extra keys per type:
      yesno   → 'pred_label', 'gold_label'  (for aggregate macro F1)
      summary → 'rouge_l', 'bert_score'
                'qa_score' is None when skip_judge=True (LLM judge skipped)
    """
    result: dict[str, object] = {"qa_score": None}
    if not pred:
        return result
    qtype = (qtype or "").lower()

    if qtype == "factoid":
        groups = _coerce_groups(query.get("exact_answer"))
        if groups:
            result["qa_score"] = factoid_mrr(pred, groups)

    elif qtype == "yesno":
        gold = query.get("exact_answer")
        if isinstance(gold, list) and gold:
            gold = gold[0]
        if isinstance(gold, str):
            result["qa_score"] = yesno_accuracy(pred, gold)
            result["pred_label"] = extract_yesno_label(pred)
            result["gold_label"] = gold.strip().lower()

    elif qtype == "list":
        groups = _coerce_groups(query.get("exact_answer"))
        if groups:
            result["qa_score"] = list_f1(pred, groups)

    elif qtype == "summary":
        ideals = _coerce_ideals(query.get("ideal_answer"))
        question = str(query.get("text") or query.get("body") or "")
        if ideals:
            if not skip_judge:
                result["qa_score"] = summary_judge_binary(question, pred, ideals)
            result["rouge_l"] = rouge_l_score(pred, ideals)
            result["bert_score"] = bert_score_f1(pred, ideals)

    return result


def aggregate_qa_scores(
    rows: Iterable[tuple[str, float]],
) -> dict[str, object]:
    """Aggregate per-query (qtype, score) rows into per-type and overall means.

    Returns:
        {
          "by_type": {qtype: mean_score},
          "n_per_type": {qtype: count},
          "overall_macro": mean of by_type values,
          "overall_micro": mean of all scores,
        }
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    total_sum = 0.0
    total_n = 0
    for qtype, score in rows:
        if score is None:
            continue
        sums[qtype] = sums.get(qtype, 0.0) + float(score)
        counts[qtype] = counts.get(qtype, 0) + 1
        total_sum += float(score)
        total_n += 1
    by_type = {t: round(sums[t] / counts[t], 4) for t in sums if counts[t] > 0}
    macro = round(sum(by_type.values()) / len(by_type), 4) if by_type else 0.0
    micro = round(total_sum / total_n, 4) if total_n > 0 else 0.0
    return {
        "by_type": by_type,
        "n_per_type": counts,
        "overall_macro": macro,
        "overall_micro": micro,
    }
