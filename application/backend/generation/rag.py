"""End-to-end RAG composition: classify type → build prompt → call LLM."""

from . import ollama_client
from . import prompts


VALID_TYPES = {"factoid", "list", "yesno", "summary"}


def classify_question(question: str) -> str:
    """Return one of factoid/list/yesno/summary. Falls back to 'summary' on error."""
    try:
        out = ollama_client.chat(
            prompts.build_classify_messages(question),
            temperature=0.0,
            max_tokens=10,
        )
        label = out.strip().lower().split()[0] if out else "summary"
        # Strip punctuation
        label = label.strip(".,'\"")
        if label in VALID_TYPES:
            return label
    except Exception:
        pass
    return "summary"


def generate_answer(
    question: str,
    docs: list[dict],
    qtype: str,
    history: list[dict] | None = None,
) -> str:
    msgs = prompts.build_messages(question, docs, qtype, history)
    return ollama_client.chat(msgs, temperature=0.2, max_tokens=512)
