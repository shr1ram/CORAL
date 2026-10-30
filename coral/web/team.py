"""Team view: who influenced whom, and how the shared memory grew.

Every interaction is a timestamped ``(from, to, kind)`` event so the dashboard
can replay the run:
  - ``message`` / ``broadcast``: an explicit `coral msg` (cooperate mode only).
  - ``built_on``: an agent evaluated an attempt whose parent commit was another
    agent's attempt (it started from that agent's code).
  - ``note_ref``: an agent's note links to a note written by another agent.

Direction is the direction of influence: from the agent whose work was used
to the agent who used it.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from coral.hub._island import all_view_roots

BROADCAST = "all"


def _epoch(stamp: str | None) -> float | None:
    """ISO timestamp → epoch seconds. Naive stamps are local time, as CORAL writes them."""
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _born(path: Path) -> float:
    """When a file or dir first appeared (birth time where the OS records it)."""
    stat = path.stat()
    return getattr(stat, "st_birthtime", stat.st_mtime)


def _run_mode(coral_dir: Path) -> str:
    try:
        config = yaml.safe_load((coral_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    except OSError:
        return "compete"
    return (config.get("agents") or {}).get("mode", "compete")


def _run_start(coral_dir: Path) -> float | None:
    config = coral_dir / "config.yaml"
    return _born(config) if config.exists() else None


def team_view(coral_dir: str | Path) -> dict[str, Any]:
    from coral.hub.attempts import _read_all_island_attempts
    from coral.hub.messages import list_messages
    from coral.hub.notes import notes_graph

    coral_dir = Path(coral_dir)
    roots = all_view_roots(coral_dir)
    start = _run_start(coral_dir)
    agents: set[str] = set()
    for root in roots:
        agents.update(p.stem for p in (root / "roles").glob("*.md"))

    memory: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []

    # Attempts, plus "built on" links through parent commits.
    attempts = _read_all_island_attempts(coral_dir)
    author = {a.commit_hash: a.agent_id for a in attempts}
    for a in attempts:
        agents.add(a.agent_id)
        t = _epoch(a.timestamp)
        memory.append(
            {
                "t": t,
                "kind": "attempt",
                "agent": a.agent_id,
                "label": a.title,
                "score": a.score,
                "status": a.status,
            }
        )
        parent_author = author.get(a.parent_hash or "")
        if parent_author and parent_author != a.agent_id:
            interactions.append(
                {
                    "t": t,
                    "from": parent_author,
                    "to": a.agent_id,
                    "kind": "built_on",
                    "text": f"{a.title} (built on {a.parent_hash[:8]})",
                }
            )

    # Notes, plus cross-agent note references.
    note_time: dict[str, float] = {}
    note_creator: dict[str, str] = {}
    for root in roots:
        graph = notes_graph(str(coral_dir), island_id=root.name if len(roots) > 1 else None)
        for node in graph["nodes"]:
            path = root / "notes" / node["id"]
            t = _born(path) if path.exists() else _epoch(node.get("date"))
            if start is not None and t is not None and t < start:
                continue  # seeded before the run
            note_time[node["id"]] = t
            note_creator[node["id"]] = node.get("creator") or "unknown"
            memory.append(
                {
                    "t": t,
                    "kind": "note",
                    "agent": note_creator[node["id"]],
                    "label": node.get("title") or node["id"],
                }
            )
        for edge in graph["edges"]:
            src, dst = note_creator.get(edge["to"]), note_creator.get(edge["from"])
            if src and dst and src != dst:
                interactions.append(
                    {
                        "t": note_time.get(edge["from"]),
                        "from": src,
                        "to": dst,
                        "kind": "note_ref",
                        "text": f"{edge['from']} {edge['kind']} {edge['to']}",
                    }
                )

    # Skills written during the run (seeded skills predate the run start).
    for root in roots:
        skills_dir = root / "skills"
        if not skills_dir.is_dir():
            continue
        for skill in sorted(p for p in skills_dir.iterdir() if p.is_dir()):
            t = _born(skill)
            if start is not None and t < start + 5:
                continue
            memory.append({"t": t, "kind": "skill", "agent": None, "label": skill.name})

    # Team memory items written by the one-shot memory service.
    from coral.hub.memory import read_items

    for root in roots:
        island = root.name if len(roots) > 1 else None
        for item in read_items(coral_dir, island):
            memory.append(
                {
                    "t": _epoch(item.get("created_at")),
                    "kind": "memory",
                    "agent": item.get("agent"),
                    "label": f"{item.get('approach', '')}: {item.get('lesson', '')}"[:160],
                }
            )

    # Messages.
    messages = []
    for root in roots:
        island = root.name if len(roots) > 1 else None
        for m in list_messages(coral_dir, island_id=island):
            t = _epoch(m.created_at)
            messages.append(
                {"t": t, "id": m.id, "from": m.sender, "to": m.to, "body": m.body, "kind": m.kind}
            )
            memory.append({"t": t, "kind": "message", "agent": m.sender, "label": m.body[:80]})
            agents.add(m.sender)
            recipients = sorted(agents - {m.sender}) if m.to == BROADCAST else [m.to]
            for to in recipients:
                interactions.append(
                    {
                        "t": t,
                        "from": m.sender,
                        "to": to,
                        "kind": "broadcast" if m.to == BROADCAST else "message",
                        "text": m.body if m.kind == "message" else f"[{m.kind.upper()}] {m.body}",
                    }
                )

    def _sorted(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted((i for i in items if i["t"] is not None), key=lambda i: i["t"])

    return {
        "mode": _run_mode(coral_dir),
        "run_start": start,
        "now": time.time(),
        "agents": sorted(agents),
        "interactions": _sorted(interactions),
        "memory": _sorted(memory),
        "messages": _sorted(messages),
    }
