"""RankZephyr 7B (Q4_K_M GGUF) — listwise reranker via llama-cpp-python.

Sliding-window listwise prompting (RankGPT / RankZephyr recipe):

* Window of 20 passages, step 10, walked back-to-front so the head of the
  list is reranked last and benefits from prior context.
* Each window: prompt the LLM with the query + numbered passages; parse the
  returned permutation ``[3] > [1] > [4] > ...`` and reorder.

Final scores are synthetic descending integers so the global ordering is
preserved for downstream IR metrics.

GGUF location is read from ``RANK_ZEPHYR_GGUF`` env var, otherwise defaults
to ``checkpoints/rank_zephyr_7b_v1_full.Q4_K_M.gguf``.
"""

import os
import re
from pathlib import Path

from .. import config


WINDOW_SIZE = 20
STEP = 10
SYSTEM_PROMPT = (
    "You are RankLLM, an intelligent assistant that can rank passages based "
    "on their relevancy to the query."
)
DEFAULT_GGUF = config.ROOT / "checkpoints/rank_zephyr_7b_v1_full.Q4_K_M.gguf"
PERM_RE = re.compile(r"\[(\d+)\]")


def _max_passage_chars() -> int:
    """Truncate each passage so 20 fit into a 4k context with headroom."""
    return 800


def _user_prompt(query: str, passages: list[str]) -> str:
    n = len(passages)
    head = (
        f"I will provide you with {n} passages, each indicated by number "
        f"identifier []. Rank the passages based on their relevance to the "
        f"search query: {query}.\n\n"
    )
    body = "\n".join(
        f"[{i + 1}] {p[: _max_passage_chars()]}" for i, p in enumerate(passages)
    )
    tail = (
        f"\n\nSearch Query: {query}.\n"
        f"Rank the {n} passages above based on their relevance to the search "
        f"query. The passages should be listed in descending order using "
        f"identifiers, the most relevant first. Output format: [] > [] > [] "
        f"(e.g. [2] > [3] > [1]). Only respond with the ranking, no other text."
    )
    return head + body + tail


def _parse_permutation(text: str, n: int) -> list[int]:
    """Return 0-indexed permutation. Missing/invalid entries appended in order."""
    seen: list[int] = []
    used: set[int] = set()
    for m in PERM_RE.finditer(text):
        idx = int(m.group(1)) - 1
        if 0 <= idx < n and idx not in used:
            seen.append(idx)
            used.add(idx)
    for i in range(n):
        if i not in used:
            seen.append(i)
    return seen


class RankZephyrReranker:
    name = "rank_zephyr"

    def __init__(self, gguf_path: str | None = None, n_ctx: int = 4096):
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is required for RankZephyr; install with "
                "`pip install llama-cpp-python`"
            ) from exc

        path = gguf_path or os.environ.get("RANK_ZEPHYR_GGUF") or str(DEFAULT_GGUF)
        if not Path(path).exists():
            raise FileNotFoundError(
                f"RankZephyr GGUF not found at {path}. Download "
                "castorini/rank_zephyr_7b_v1_full Q4_K_M from HF and place it there, "
                "or set RANK_ZEPHYR_GGUF."
            )

        self.llm = Llama(
            model_path=path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            verbose=False,
        )

    def _rank_window(self, query: str, passages: list[str]) -> list[int]:
        prompt = _user_prompt(query, passages)
        out = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.0,
        )
        text = out["choices"][0]["message"]["content"]
        return _parse_permutation(text, len(passages))

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []
        order = list(range(len(candidates)))  # current global order, indices into candidates
        texts = [c[1] for c in candidates]

        # Sliding window, back-to-front.
        end = len(order)
        while end > 0:
            start = max(0, end - WINDOW_SIZE)
            window_idxs = order[start:end]
            window_texts = [texts[i] for i in window_idxs]
            try:
                perm = self._rank_window(query, window_texts)
            except Exception:
                perm = list(range(len(window_idxs)))  # leave the window order intact on failure
            new_window = [window_idxs[p] for p in perm]
            order[start:end] = new_window
            if start == 0:
                break
            end -= STEP

        n = len(order)
        ranked = [(candidates[i][0], float(n - rank)) for rank, i in enumerate(order)]
        return ranked
