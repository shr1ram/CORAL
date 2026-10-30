"""Subagent requests and results over <island_root>/subagents/ (agents.subagents).

An agent (sandboxed, often without internet) cannot run a Claude subagent
itself, so it writes a request file; the manager's broker runs the subagent
and writes the result next to it:

    subagents/requests/<id>.json   written by `coral subagent run` in the agent
    subagents/results/<id>.json    written by the broker in the manager

Subagent types are the markdown definitions in <island_root>/agents/ (seeded
from coral/template/agents/) plus the built-in "general".
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from coral.hub._island import island_root
from coral.hub.steering import _atomic_write_json

GENERAL = "general"
MAX_TASK_CHARS = 8000


@dataclass
class SubagentRequest:
    agent_id: str
    kind: str
    task: str
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class SubagentDefinition:
    name: str
    description: str
    tools: list[str]
    prompt: str


def _dir(coral_dir: str | Path, island_id: str | int | None, sub: str) -> Path:
    return island_root(coral_dir, island_id) / "subagents" / sub


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def submit(
    coral_dir: str | Path,
    agent_id: str,
    kind: str,
    task: str,
    island_id: str | int | None = None,
) -> SubagentRequest:
    task = task.strip()
    if not task:
        raise ValueError("subagent task is empty")
    if len(task) > MAX_TASK_CHARS:
        raise ValueError(f"subagent task is {len(task)} characters; keep it under {MAX_TASK_CHARS}")
    request = SubagentRequest(agent_id=agent_id, kind=kind, task=task)
    path = _dir(coral_dir, island_id, "requests") / f"{request.id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, asdict(request))
    return request


def get_result(
    coral_dir: str | Path, request_id: str, island_id: str | int | None = None
) -> dict[str, Any] | None:
    """The result for a request (id or unique prefix), or None if still running."""
    results = _dir(coral_dir, island_id, "results")
    matches = list(results.glob(f"{request_id}*.json"))
    return _load(matches[0]) if len(matches) == 1 else None


def wait_result(
    coral_dir: str | Path,
    request_id: str,
    island_id: str | int | None = None,
    timeout: float = 900,
    poll: float = 1.0,
) -> dict[str, Any] | None:
    """Block until the broker writes the result, or return None on timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = get_result(coral_dir, request_id, island_id)
        if result is not None:
            return result
        time.sleep(poll)
    return None


def pending_requests(
    coral_dir: str | Path, island_id: str | int | None = None
) -> list[SubagentRequest]:
    """Requests without a result yet (queued or running), oldest first."""
    done = {p.stem for p in _dir(coral_dir, island_id, "results").glob("*.json")}
    out = []
    for path in sorted(_dir(coral_dir, island_id, "requests").glob("*.json")):
        data = _load(path)
        if path.stem in done or data is None:
            continue
        try:
            out.append(SubagentRequest(**data))
        except TypeError:
            continue
    out.sort(key=lambda r: r.created_at)
    return out


def running_count(coral_dir: str | Path, agent_id: str, island_id: str | int | None = None) -> int:
    """How many of this agent's subagents are queued or running."""
    return sum(1 for r in pending_requests(coral_dir, island_id) if r.agent_id == agent_id)


def list_results(coral_dir: str | Path, island_id: str | int | None = None) -> list[dict[str, Any]]:
    out = [_load(p) for p in sorted(_dir(coral_dir, island_id, "results").glob("*.json"))]
    return sorted((r for r in out if r), key=lambda r: r.get("created_at", ""))


def spent_usd(coral_dir: str | Path, island_ids: list[str | int | None]) -> float:
    """Estimated cost of all finished subagent calls in the run."""
    return sum(
        float(r.get("cost_usd") or 0) for i in island_ids for r in list_results(coral_dir, i)
    )


def write_result(
    coral_dir: str | Path,
    request: SubagentRequest,
    result: dict[str, Any],
    island_id: str | int | None = None,
) -> None:
    path = _dir(coral_dir, island_id, "results") / f"{request.id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, {**asdict(request), **result})


def read_definitions(
    coral_dir: str | Path, island_id: str | int | None = None
) -> dict[str, SubagentDefinition]:
    """Subagent types: markdown definitions in agents/ plus the built-in general helper."""
    defs = {
        GENERAL: SubagentDefinition(
            name=GENERAL,
            description="General worker: write and test code, analyse results, write notes.",
            tools=["Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebSearch", "WebFetch"],
            prompt="You are a general-purpose worker. Do the task you are given thoroughly.",
        )
    }
    for path in sorted((island_root(coral_dir, island_id) / "agents").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        meta: dict[str, Any] = {}
        body = text
        if text.startswith("---"):
            _, front, body = text.split("---", 2)
            meta = yaml.safe_load(front) or {}
        tools = meta.get("tools") or {}
        if isinstance(tools, dict):
            tools = [name for name, on in tools.items() if on]
        name = meta.get("name") or path.stem
        defs[name] = SubagentDefinition(
            name=name,
            description=str(meta.get("description", "")),
            tools=list(tools),
            prompt=body.strip(),
        )
    return defs
