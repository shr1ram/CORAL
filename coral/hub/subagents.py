"""Subagent requests and results (agents.subagents).

An agent (sandboxed, often without internet) cannot run a Claude subagent
itself, so it writes a request file; the manager's broker runs the subagent
and writes the result:

    <island_root>/subagents/requests/<agent>/<id>.json   written by `coral subagent run`
    .coral/private/subagents/results/<id>.json           authoritative, written by the broker
    <island_root>/subagents/results/<id>.json            read-only mirror for the CLI

Everything under <island_root> is agent-writable, so nothing there is trusted:
the broker takes the request id from the file name and the agent from the
directory name (the agent sandbox denies writes to other agents' request
directories and to the results mirror), and keeps results, the spend ledger
and a snapshot of the subagent definitions under ``.coral/private``.

Subagent types are the markdown definitions in <island_root>/agents/ (seeded
from coral/template/agents/) plus the built-in "general".
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from coral.hub._island import island_root
from coral.hub.steering import _atomic_write_json

logger = logging.getLogger(__name__)

GENERAL = "general"
MAX_TASK_CHARS = 8000
REQUEST_ID_RE = re.compile(r"^[0-9a-f]{32}$")
AGENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
KIND_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


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


def private_dir(coral_dir: str | Path) -> Path:
    """Broker-only state: authoritative results, spend ledger, definition snapshot."""
    return Path(coral_dir) / "private" / "subagents"


def valid_agent_id(agent_id: str) -> bool:
    return bool(AGENT_ID_RE.match(agent_id)) and agent_id not in (".", "..")


def _load(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def submit(
    coral_dir: str | Path,
    agent_id: str,
    kind: str,
    task: str,
    island_id: str | int | None = None,
) -> SubagentRequest:
    if not valid_agent_id(agent_id):
        raise ValueError(f"invalid agent id {agent_id!r}")
    task = task.strip()
    if not task:
        raise ValueError("subagent task is empty")
    if len(task) > MAX_TASK_CHARS:
        raise ValueError(f"subagent task is {len(task)} characters; keep it under {MAX_TASK_CHARS}")
    request = SubagentRequest(agent_id=agent_id, kind=kind, task=task)
    path = _dir(coral_dir, island_id, "requests") / agent_id / f"{request.id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, asdict(request))
    return request


def get_result(
    coral_dir: str | Path, request_id: str, island_id: str | int | None = None
) -> dict[str, Any] | None:
    """The mirrored result for a request (id or unique prefix), or None if still running."""
    if not re.fullmatch(r"[0-9a-f]{1,32}", request_id):
        return None
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
    coral_dir: str | Path,
    island_id: str | int | None = None,
    done: set[str] | None = None,
) -> list[SubagentRequest]:
    """Well-formed requests without a result yet, oldest first.

    The id comes from the file name and the agent from its directory, never
    from the JSON. ``done`` defaults to the ids in the results mirror (good
    enough for the CLI); the broker passes its own private set.
    """
    if done is None:
        done = {p.stem for p in _dir(coral_dir, island_id, "results").glob("*.json")}
    out = []
    for path in sorted(_dir(coral_dir, island_id, "requests").glob("*/*.json")):
        request_id, agent_id = path.stem, path.parent.name
        if request_id in done or not REQUEST_ID_RE.match(request_id):
            continue
        if not valid_agent_id(agent_id):
            continue
        data = _load(path)
        if data is None:
            continue
        kind, task = data.get("kind"), data.get("task")
        if not isinstance(kind, str) or not isinstance(task, str):
            continue
        created = data.get("created_at")
        out.append(
            SubagentRequest(
                agent_id=agent_id,
                kind=kind,
                task=task,
                id=request_id,
                created_at=created if isinstance(created, str) else "",
            )
        )
    out.sort(key=lambda r: r.created_at)
    return out


def running_count(coral_dir: str | Path, agent_id: str, island_id: str | int | None = None) -> int:
    """How many of this agent's subagents are queued or running (per the mirror)."""
    return sum(1 for r in pending_requests(coral_dir, island_id) if r.agent_id == agent_id)


def list_results(coral_dir: str | Path, island_id: str | int | None = None) -> list[dict[str, Any]]:
    """Mirrored results, for display only (agent-writable, never trusted)."""
    out = [_load(p) for p in sorted(_dir(coral_dir, island_id, "results").glob("*.json"))]
    return sorted((r for r in out if r), key=lambda r: str(r.get("created_at", "")))


def private_result_ids(coral_dir: str | Path) -> set[str]:
    return {p.stem for p in (private_dir(coral_dir) / "results").glob("*.json")}


def write_result(
    coral_dir: str | Path,
    request: SubagentRequest,
    result: dict[str, Any],
    island_id: str | int | None = None,
) -> None:
    """Write the authoritative result privately, then mirror it for the CLI."""
    if not REQUEST_ID_RE.match(request.id):
        raise ValueError(f"invalid request id {request.id!r}")
    record = {**asdict(request), **result}
    private = private_dir(coral_dir) / "results" / f"{request.id}.json"
    private.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(private, record)
    mirror = _dir(coral_dir, island_id, "results") / f"{request.id}.json"
    mirror.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(mirror, record)


def _general() -> SubagentDefinition:
    return SubagentDefinition(
        name=GENERAL,
        description="General worker: write and test code, analyse results, write notes.",
        tools=["Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebSearch", "WebFetch"],
        prompt="You are a general-purpose worker. Do the task you are given thoroughly.",
    )


def _parse_definition(path: Path) -> SubagentDefinition | None:
    try:
        text = path.read_text(encoding="utf-8")
        meta: dict[str, Any] = {}
        body = text
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) < 3:
                return None
            _, front, body = parts
            loaded = yaml.safe_load(front) or {}
            meta = loaded if isinstance(loaded, dict) else {}
        tools = meta.get("tools") or {}
        if isinstance(tools, dict):
            tools = [name for name, on in tools.items() if on]
        if not isinstance(tools, list):
            tools = []
        name = str(meta.get("name") or path.stem)
        return SubagentDefinition(
            name=name,
            description=str(meta.get("description", "")),
            tools=[str(t) for t in tools],
            prompt=body.strip(),
        )
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ValueError):
        logger.warning(f"Ignoring malformed subagent definition {path}")
        return None


def read_definitions_dir(directory: Path) -> dict[str, SubagentDefinition]:
    """Definitions in ``directory`` plus the built-in general helper; bad files are skipped."""
    defs = {GENERAL: _general()}
    for path in sorted(Path(directory).glob("*.md")):
        definition = _parse_definition(path)
        if definition is not None and KIND_RE.match(definition.name):
            defs[definition.name] = definition
    return defs


def read_definitions(
    coral_dir: str | Path, island_id: str | int | None = None
) -> dict[str, SubagentDefinition]:
    """Definitions from the shared agents/ dir (for the CLI; the broker uses its snapshot)."""
    return read_definitions_dir(island_root(coral_dir, island_id) / "agents")


def snapshot_definitions(coral_dir: str | Path, island_id: str | int | None = None) -> Path:
    """Copy the shared definitions to a broker-only dir before agents can edit them."""
    target = private_dir(coral_dir) / "agents" / (str(island_id) if island_id is not None else "_")
    target.mkdir(parents=True, exist_ok=True)
    source = island_root(coral_dir, island_id) / "agents"
    for path in sorted(source.glob("*.md")):
        dest = target / path.name
        if not dest.exists():
            shutil.copyfile(path, dest)
    return target
