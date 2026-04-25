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
) -> str:
    resp = get_client().chat.completions.create(
        model=model or config.OLLAMA_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()
