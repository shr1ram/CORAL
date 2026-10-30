"""Team memory for one-shot runs over <island_root>/memory/ (agents.memory).

Written only by the manager's memory service (coral/agent/memory_service.py),
never by agents:

    memory/items/<attempt hash>.json   one item per scored attempt
    memory/state.json                  periodic state-of-play summary
    memory/briefs/<agent>.json         the librarian's brief for an agent's next attempt

Agents only ever see their brief. Retrieval for the librarian uses BM25 over
item text and tags once the store is too big to show it whole.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from coral.hub._island import island_root
from coral.hub.steering import _atomic_write_json

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _dir(coral_dir: str | Path, island_id: str | int | None) -> Path:
    return island_root(coral_dir, island_id) / "memory"


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _save(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, data)


def write_item(
    coral_dir: str | Path, item: dict[str, Any], island_id: str | int | None = None
) -> None:
    _save(_dir(coral_dir, island_id) / "items" / f"{item['id']}.json", item)


def read_items(coral_dir: str | Path, island_id: str | int | None = None) -> list[dict[str, Any]]:
    items = [_load(p) for p in (_dir(coral_dir, island_id) / "items").glob("*.json")]
    return sorted((i for i in items if i), key=lambda i: i.get("created_at", ""))


def item_text(item: dict[str, Any]) -> str:
    """One-line rendering used for search, the librarian's index and summaries."""
    tags = ", ".join(item.get("tags") or [])
    return (
        f"[{item['id'][:8]}] {item.get('agent', '?')} score {item.get('score')} "
        f"({item.get('outcome', '')}): {item.get('approach', '')}. Change: {item.get('change', '')}. "
        f"Result: {item.get('result', '')}. Lesson: {item.get('lesson', '')}. Tags: {tags}"
    )


def known_tags(items: list[dict[str, Any]]) -> list[str]:
    counts = Counter(t for i in items for t in (i.get("tags") or []))
    return [t for t, _ in counts.most_common()]


def write_state(
    coral_dir: str | Path, state: dict[str, Any], island_id: str | int | None = None
) -> None:
    _save(_dir(coral_dir, island_id) / "state.json", state)


def read_state(coral_dir: str | Path, island_id: str | int | None = None) -> dict[str, Any] | None:
    return _load(_dir(coral_dir, island_id) / "state.json")


def write_brief(
    coral_dir: str | Path, agent_id: str, brief: dict[str, Any], island_id: str | int | None = None
) -> None:
    _save(_dir(coral_dir, island_id) / "briefs" / f"{agent_id}.json", brief)


def read_brief(
    coral_dir: str | Path, agent_id: str, island_id: str | int | None = None
) -> dict[str, Any] | None:
    return _load(_dir(coral_dir, island_id) / "briefs" / f"{agent_id}.json")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower().replace("-", " "))


def bm25_rank(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[int]:
    """Indices of ``docs`` ordered by BM25 relevance to ``query`` (best first)."""
    tokenized = [tokenize(d) for d in docs]
    if not tokenized:
        return []
    avg_len = sum(len(t) for t in tokenized) / len(tokenized) or 1.0
    df = Counter(term for t in tokenized for term in set(t))
    n = len(tokenized)
    terms = set(tokenize(query))
    scores = []
    for i, tokens in enumerate(tokenized):
        tf = Counter(tokens)
        score = 0.0
        for term in terms:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * len(tokens) / avg_len)
            score += idf * tf[term] * (k1 + 1) / denom
        scores.append((score, i))
    return [i for _, i in sorted(scores, key=lambda x: (-x[0], x[1]))]
