"""Type-specific RAG prompts for biomedical question answering.

Each prompt instructs the model to (a) answer ONLY from the provided abstracts,
(b) say "no relevant documents" when context is insufficient, and (c) follow a
type-specific output format that the evaluation metrics can parse.
"""

SYSTEM_BASE = (
    "You are a biomedical research assistant. Answer ONLY using information from "
    "the provided PubMed abstracts. If the abstracts do not contain enough "
    "information to answer the question, respond with exactly: NO RELEVANT DOCUMENTS."
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
    "yesno": (
        "Read the abstracts carefully. Answer with exactly one word: Yes or No. "
        "Do not add any explanation, citations, or extra text."
    ),
    "factoid": (
        "Return up to 5 candidate answers, one per line, ranked from "
        "MOST to LEAST confident. Each line must be a single short noun "
        "phrase or entity name (no bullets, no numbering, no full "
        "sentences, no explanations). If you only have one candidate, "
        "return just that one line."
    ),
    "list": (
        "Return ONLY the list. One item per line, prefixed with '- '. "
        "No introduction, no numbering, no explanation. "
        "Use the most specific item names found in the abstracts."
    ),
    "summary": (
        "Provide a concise 2-4 sentence summary that answers the question, "
        "grounded strictly in the abstracts. No bullet points."
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
