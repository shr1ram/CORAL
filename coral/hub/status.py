"""Per-agent STATUS lines over <island_root>/status/<agent>.json (one-shot board).

Each one-shot reply may carry `STATUS: <what I'm working on>`; the worker
stores it here with the attempt's direction and any RECALL question, and the
prompts of teammates show everyone's latest status. The memory service also
reads these to tailor each agent's brief.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from coral.hub._island import island_root
from coral.hub.steering import _atomic_write_json


def _path(coral_dir: str | Path, agent_id: str, island_id: str | int | None) -> Path:
    return island_root(coral_dir, island_id) / "status" / f"{agent_id}.json"


def write_status(
    coral_dir: str | Path,
    agent_id: str,
    *,
    status: str | None,
    direction: str = "",
    recall: str = "",
    island_id: str | int | None = None,
) -> dict[str, Any]:
    """Record the agent's latest status; keeps the previous status if none given."""
    previous = read_status(coral_dir, agent_id, island_id) or {}
    record = {
        "agent_id": agent_id,
        "status": (status or previous.get("status", "")).strip()[:300],
        "direction": direction,
        "recall": recall.strip()[:500],
        "updated_at": datetime.now(UTC).isoformat(),
    }
    path = _path(coral_dir, agent_id, island_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, record)
    return record


def read_status(
    coral_dir: str | Path, agent_id: str, island_id: str | int | None = None
) -> dict[str, Any] | None:
    try:
        return json.loads(_path(coral_dir, agent_id, island_id).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_board(
    coral_dir: str | Path, island_id: str | int | None = None
) -> dict[str, dict[str, Any]]:
    """Every agent's latest status record, by agent id."""
    out = {}
    for path in sorted((island_root(coral_dir, island_id) / "status").glob("*.json")):
        try:
            out[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return out
