"""Agent-to-agent messages over <island_root>/messages/ (cooperate mode only).

Each message is one JSON file named ``<created_at>-<id>.json`` so a directory
listing is already in send order. ``to`` is an agent id or ``"all"`` for a
broadcast. Read state is tracked per recipient in ``messages/.read/<agent>.json``
so a broadcast is delivered once to every agent except its sender.

Messages carry a ``kind``: "message" (plain, from `coral msg`), or one of the
one-shot types "finding", "warning", "request", "reply". A request stays open
for a recipient until that recipient sends a reply to the requester.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from coral.hub._island import island_root
from coral.hub.steering import _atomic_write_json

BROADCAST = "all"
MAX_BODY_CHARS = 2000
KINDS = ("message", "finding", "warning", "request", "reply")


@dataclass
class Message:
    sender: str
    to: str
    body: str
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    kind: str = "message"

    @classmethod
    def from_dict(cls, data: dict) -> Message:
        return cls(
            sender=data["sender"],
            to=data["to"],
            body=data["body"],
            id=data["id"],
            created_at=data["created_at"],
            kind=data.get("kind", "message"),
        )

    def is_for(self, agent_id: str) -> bool:
        return self.sender != agent_id and self.to in (agent_id, BROADCAST)


def send(
    coral_dir: str | Path,
    sender: str,
    to: str,
    body: str,
    island_id: str | int | None = None,
    kind: str = "message",
) -> Message:
    """Write one message. Raises ValueError for an empty or oversized body or bad kind."""
    if kind not in KINDS:
        raise ValueError(f"message kind must be one of {KINDS}, got {kind!r}")
    body = body.strip()
    if not body:
        raise ValueError("message body is empty")
    if len(body) > MAX_BODY_CHARS:
        raise ValueError(
            f"message is {len(body)} characters; keep it under {MAX_BODY_CHARS} "
            "(put longer findings in a note and send its path)"
        )
    message = Message(sender=sender, to=to, body=body, kind=kind)
    directory = _messages_dir(coral_dir, island_id)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(message.created_at).strftime("%Y%m%dT%H%M%S%f")
    _atomic_write_json(directory / f"{stamp}-{message.id[:8]}.json", asdict(message))
    return message


def list_messages(coral_dir: str | Path, island_id: str | int | None = None) -> list[Message]:
    """All messages in send order, skipping unreadable files."""
    messages = []
    for path in sorted(_messages_dir(coral_dir, island_id).glob("*.json")):
        try:
            messages.append(Message.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
    messages.sort(key=lambda m: m.created_at)
    return messages


def read_unread(
    coral_dir: str | Path, agent_id: str, island_id: str | int | None = None
) -> list[Message]:
    """Messages addressed to ``agent_id`` (directly or broadcast) it has not read."""
    seen = _read_ids(coral_dir, agent_id, island_id)
    return [
        m for m in list_messages(coral_dir, island_id) if m.is_for(agent_id) and m.id not in seen
    ]


def mark_read(
    coral_dir: str | Path,
    agent_id: str,
    message_ids: list[str],
    island_id: str | int | None = None,
) -> None:
    if not message_ids:
        return
    seen = _read_ids(coral_dir, agent_id, island_id) | set(message_ids)
    path = _read_path(coral_dir, agent_id, island_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, {"read": sorted(seen)})


def format_messages(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        scope = " (to all)" if m.to == BROADCAST else ""
        kind = "" if m.kind == "message" else f" [{m.kind.upper()}]"
        lines.append(f"- From {m.sender}{scope}{kind} at {m.created_at[:19]}Z:\n  {m.body}")
    return "\n".join(lines)


def recent_for(
    coral_dir: str | Path, agent_id: str, n: int = 10, island_id: str | int | None = None
) -> list[Message]:
    """The last ``n`` messages this agent can see: to it, to all, or sent by it."""
    visible = [
        m
        for m in list_messages(coral_dir, island_id)
        if m.to in (agent_id, BROADCAST) or m.sender == agent_id
    ]
    return visible[-n:]


def open_requests(
    coral_dir: str | Path, agent_id: str, island_id: str | int | None = None
) -> list[Message]:
    """Requests addressed to ``agent_id`` (or all) it has not replied to yet."""
    messages = list_messages(coral_dir, island_id)
    replies = [m for m in messages if m.kind == "reply" and m.sender == agent_id]
    return [
        r
        for r in messages
        if r.kind == "request"
        and r.is_for(agent_id)
        and not any(p.to == r.sender and p.created_at > r.created_at for p in replies)
    ]


def _messages_dir(coral_dir: str | Path, island_id: str | int | None) -> Path:
    return island_root(coral_dir, island_id) / "messages"


def _read_path(coral_dir: str | Path, agent_id: str, island_id: str | int | None) -> Path:
    return _messages_dir(coral_dir, island_id) / ".read" / f"{agent_id}.json"


def _read_ids(coral_dir: str | Path, agent_id: str, island_id: str | int | None) -> set[str]:
    path = _read_path(coral_dir, agent_id, island_id)
    try:
        return set(json.loads(path.read_text(encoding="utf-8"))["read"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return set()
