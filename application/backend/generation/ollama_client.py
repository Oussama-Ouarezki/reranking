"""Thin wrapper around the Ollama OpenAI-compatible API."""

from openai import OpenAI

from .. import config


_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key="ollama", base_url=config.OLLAMA_BASE_URL)
    return _client


def chat(
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
    num_ctx: int | None = None,
) -> str:
    """Run a chat completion against Ollama.

    Pass ``num_ctx`` to override Ollama's context window (default 2048).
    Generation runs that feed many full-text PubMed abstracts should set this
    high enough to fit the prompt — Llama 3.1 supports up to 128k.
    """
    extra: dict = {}
    if num_ctx is not None:
        extra["extra_body"] = {"options": {"num_ctx": num_ctx}}
    resp = get_client().chat.completions.create(
        model=model or config.OLLAMA_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **extra,
    )
    return (resp.choices[0].message.content or "").strip()
