"""Type-specific RAG prompts for biomedical question answering.

Each prompt instructs the model to (a) answer ONLY from the provided abstracts,
(b) say "no relevant documents" when context is insufficient, and (c) follow a
type-specific output format that the evaluation metrics can parse.

The four BioASQ question types use distinct output formats:
  yesno    → single token "Yes" or "No"
  factoid  → up to 5 ranked entity names, one per line
  list     → bulleted list of distinct entities
  summary  → ~2-3 sentence "ideal answer"
"""

SYSTEM_BASE = (
    "You are a biomedical research assistant for the BioASQ task. "
    "Answer ONLY using information explicitly stated in the provided PubMed "
    "abstracts — do not use prior knowledge. If the abstracts do not contain "
    "enough information to answer the question, respond with exactly: "
    "NO RELEVANT DOCUMENTS. Each generation is fully independent — there is no "
    "memory of previous questions."
)


def _format_context(docs: list[dict]) -> str:
    """docs: list of {rank, docid, title, text}."""
    parts = []
    for d in docs:
        title = d.get("title", "").strip()
        text = d.get("text", "").strip()
        parts.append(f"[{d['rank']}] PMID {d['docid']} — {title}\n{text}")
    return "\n\n".join(parts)


PROMPT_BY_TYPE = {
    # Yes/No → graded by exact accuracy on the first yes/no token.
    "yesno": (
        "This is a YES/NO question. Read the abstracts carefully and decide "
        "whether the claim in the question is supported.\n"
        "Output rules:\n"
        "  • Respond with exactly one token: Yes or No.\n"
        "  • No punctuation, no explanation, no citations, no extra text.\n"
        "  • If the abstracts genuinely do not address the question, prefer the "
        "    answer most consistent with the strongest evidence shown."
    ),
    # Factoid → graded by MRR over up to 5 ranked candidates.
    "factoid": (
        "This is a FACTOID question expecting ONE correct entity name "
        "(gene, drug, disease, protein, organism, person, etc.).\n"
        "Output rules:\n"
        "  • Return UP TO 5 candidate answers, one per line.\n"
        "  • Order them from MOST to LEAST confident — the first line should be "
        "    your single best guess.\n"
        "  • Each line is a SHORT noun phrase or entity name only "
        "    (no articles like 'the', no verbs, no full sentences).\n"
        "  • Do NOT add bullets, numbering, prefixes, citations, or commentary.\n"
        "  • Do NOT repeat the same entity across lines.\n"
        "  • If you only have one good candidate, return just that one line."
    ),
    # List → graded by mean F-measure on the set of returned entities.
    "list": (
        "This is a LIST question expecting MULTIPLE correct entities.\n"
        "Output rules:\n"
        "  • Return one entity per line, each prefixed with '- '.\n"
        "  • Each item is a short specific entity name only "
        "    (no descriptions, no parenthetical notes).\n"
        "  • Use the most specific terms found in the abstracts (e.g., gene "
        "    symbols, drug names, disease names).\n"
        "  • Do not repeat entities. Do not add an introduction or summary.\n"
        "  • Precision matters as much as recall — only include items the "
        "    abstracts actually support."
    ),
    # Summary → graded by ROUGE-L / BERTScore / LLM-judge against ideal_answer.
    "summary": (
        "This is a SUMMARY question expecting a short 'ideal answer'.\n"
        "Output rules:\n"
        "  • Write 2 to 3 complete sentences (around 50-100 words).\n"
        "  • Be a direct answer to the question, written as flowing prose.\n"
        "  • Stay strictly grounded in the abstracts.\n"
        "  • No bullet points, no headings, no citations, no PMIDs, "
        "    no meta-commentary like 'Based on the abstracts...'."
    ),
}


def build_messages(
    question: str,
    docs: list[dict],
    qtype: str,
    history: list[dict] | None = None,
) -> list[dict]:
    """Construct chat messages for Ollama."""
    type_instr = PROMPT_BY_TYPE.get(qtype, PROMPT_BY_TYPE["summary"])
    system = f"{SYSTEM_BASE}\n\n{type_instr}"
    context = _format_context(docs)

    user_msg = (
        f"Question: {question}\n\n"
        f"Abstracts:\n{context}\n\n"
        f"Answer:"
    )

    msgs: list[dict] = [{"role": "system", "content": system}]
    if history:
        for turn in history[-6:]:  # last 6 turns max
            msgs.append({"role": turn["role"], "content": turn["content"]})
    msgs.append({"role": "user", "content": user_msg})
    return msgs


CLASSIFY_SYSTEM = (
    "Classify the user's biomedical question into exactly ONE of these four "
    "types and respond with the label only (no quotes, no explanation):\n"
    "- factoid : a single factual answer (e.g., 'What gene encodes X?')\n"
    "- list    : asks for a list of items (e.g., 'List the inhibitors of X')\n"
    "- yesno   : has a yes/no answer (e.g., 'Is X effective for Y?')\n"
    "- summary : asks for an explanation/description (e.g., 'Describe X')"
)


def build_classify_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": CLASSIFY_SYSTEM},
        {"role": "user", "content": question},
    ]
