"""In-memory corpus loader. Maps docid -> {title, text, corpus_type}."""

import json
from .. import config


class Corpus:
    def __init__(self, path=None):
        self.path = path or config.CORPUS_PATH
        self._docs: dict[str, dict] = {}

    def load(self) -> None:
        if self._docs:
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                self._docs[str(d["_id"])] = {
                    "title": d.get("title", ""),
                    "text": d.get("text", ""),
                    "corpus_type": d.get("corpus_type"),
                }

    def get(self, docid: str) -> dict | None:
        return self._docs.get(str(docid))

    def get_text(self, docid: str) -> str:
        d = self._docs.get(str(docid))
        if not d:
            return ""
        return f"{d['title']} {d['text']}".strip()

    def __len__(self) -> int:
        return len(self._docs)
