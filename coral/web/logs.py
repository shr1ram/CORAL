"""Parse agent NDJSON logs (Claude Code, Pi) into structured turns for the web UI."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class LogEntry:
    """A single meaningful event extracted from a log."""

    # Types: thinking, tool_call, tool_result, text, system, error,
    #        coral_prompt, subagent_start, subagent_progress, subagent_done,
    #        compact, result
    type: str
    content: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    turn_index: int = 0


@dataclass
class LogTurn:
    """A conversation turn (assistant message + tool results)."""

    index: int
    entries: list[LogEntry] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "entries": [
                {
                    "type": e.type,
                    "content": e.content,
                    "details": e.details,
                    "timestamp": e.timestamp,
                }
                for e in self.entries
            ],
            "usage": self.usage,
            "timestamp": self.timestamp,
        }


# Pi streams one line per token delta, each repeating the whole partial message.
# Those lines are most of a Pi log; skip them without decoding (message_end has
# the final content).
_PI_STREAM_PREFIXES = ('{"type":"message_update"', '{"type":"tool_execution_update"')


def _ms_to_iso(ms: Any) -> str:
    try:
        return datetime.fromtimestamp(float(ms) / 1000).isoformat(timespec="seconds")
    except (TypeError, ValueError, OSError):
        return ""


def _tool_input_text(tool_input: Any) -> str:
    """Full tool input for display: the command for bash, else pretty JSON."""
    if isinstance(tool_input, dict) and set(tool_input) == {"command"}:
        return str(tool_input["command"])
    try:
        return json.dumps(tool_input, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(tool_input)


def _extract_pi_blocks(content_list: list[dict]) -> list[LogEntry]:
    """Entries from a Pi assistant message (thinking / text / toolCall blocks)."""
    entries = []
    for block in content_list:
        block_type = block.get("type", "")
        if block_type == "thinking" and block.get("thinking"):
            entries.append(LogEntry(type="thinking", content=block["thinking"]))
        elif block_type == "text" and block.get("text"):
            entries.append(LogEntry(type="text", content=block["text"]))
        elif block_type == "toolCall":
            name = block.get("name", "unknown")
            args = block.get("arguments", {})
            entries.append(
                LogEntry(
                    type="tool_call",
                    content=name,
                    details={
                        "input_summary": _summarize_tool_input(name, args),
                        "input": _tool_input_text(args),
                        "tool_use_id": block.get("id", ""),
                    },
                )
            )
    return entries


def _extract_content_blocks(content_list: list[dict]) -> list[LogEntry]:
    """Extract structured entries from an assistant message's content array."""
    entries = []
    for block in content_list:
        block_type = block.get("type", "")

        if block_type == "thinking":
            text = block.get("thinking", "")
            if text:
                entries.append(
                    LogEntry(
                        type="thinking",
                        content=text,
                    )
                )

        elif block_type == "text":
            text = block.get("text", "")
            if text:
                entries.append(LogEntry(type="text", content=text))

        elif block_type == "tool_use":
            name = block.get("name", "unknown")
            tool_input = block.get("input", {})
            # Summarize tool input
            summary = _summarize_tool_input(name, tool_input)
            entries.append(
                LogEntry(
                    type="tool_call",
                    content=name,
                    details={
                        "input_summary": summary,
                        "input": _tool_input_text(tool_input),
                        "tool_use_id": block.get("id", ""),
                    },
                )
            )

        elif block_type == "tool_result":
            content = block.get("content", "")
            if isinstance(content, list):
                content = "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
            entries.append(
                LogEntry(
                    type="tool_result",
                    content=str(content),
                    details={"tool_use_id": block.get("tool_use_id", "")},
                )
            )

    return entries


def _summarize_tool_input(tool_name: str, tool_input: dict) -> str:
    """Create a short summary of tool input."""
    if not isinstance(tool_input, dict):
        return str(tool_input)[:200]

    if tool_name in ("Read", "read_file"):
        return tool_input.get("file_path", tool_input.get("path", ""))
    elif tool_name in ("Edit", "edit_file"):
        return tool_input.get("file_path", tool_input.get("path", ""))
    elif tool_name in ("Write", "write_file"):
        return tool_input.get("file_path", tool_input.get("path", ""))
    elif tool_name in ("Bash", "bash"):
        cmd = tool_input.get("command", "")
        return cmd[:200] if cmd else ""
    elif tool_name in ("Grep", "grep"):
        return f"/{tool_input.get('pattern', '')}/ in {tool_input.get('path', '.')}"
    elif tool_name in ("Glob", "glob"):
        return tool_input.get("pattern", "")
    elif tool_name == "Agent":
        return tool_input.get("description", tool_input.get("prompt", ""))[:200]
    else:
        # Generic: show first key-value
        for k, v in tool_input.items():
            return f"{k}: {str(v)[:150]}"
    return ""


@dataclass
class SessionMeta:
    """Metadata from the result message at the end of a session."""

    total_cost_usd: float | None = None
    duration_ms: int | None = None
    duration_api_ms: int | None = None
    num_turns: int | None = None
    stop_reason: str = ""
    session_id: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    model_usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cost_usd": self.total_cost_usd,
            "duration_ms": self.duration_ms,
            "duration_api_ms": self.duration_api_ms,
            "num_turns": self.num_turns,
            "stop_reason": self.stop_reason,
            "session_id": self.session_id,
            "usage": self.usage,
            "model_usage": self.model_usage,
        }


_PI_STREAM_PREFIXES_B = tuple(p.encode() for p in _PI_STREAM_PREFIXES)


class LogParser:
    """Incremental NDJSON log parser that keeps its turn state between reads.

    ``read(path)`` consumes only the bytes added since the last read. Lines are
    filtered on their raw bytes before decoding (Pi's per-token
    ``message_update`` lines are most of a Pi log), and a trailing partial line
    is left for the next read unless it already parses as JSON.
    """

    def __init__(self, offset: int = 0) -> None:
        self.turns: list[LogTurn] = []
        self.current_turn: LogTurn | None = None
        self.turn_index = 0
        self.session_meta: SessionMeta | None = None
        self.offset = offset

    def read(self, path: Path, chunk_size: int = 8 << 20) -> None:
        with open(path, "rb") as f:
            f.seek(self.offset)
            tail = b""  # bytes after the last newline, not yet consumed
            while chunk := f.read(chunk_size):
                data = tail + chunk
                cut = data.rfind(b"\n") + 1
                if cut:
                    self._feed_bytes(data[:cut])
                    self.offset += cut
                tail = data[cut:]
            if tail.strip():
                # Complete JSON without a trailing newline: consume it; otherwise
                # it is still being written, so leave it for the next read.
                try:
                    obj = json.loads(tail)
                except json.JSONDecodeError:
                    return
                if isinstance(obj, dict):
                    self.feed(obj)
                self.offset += len(tail)

    def _feed_bytes(self, data: bytes) -> None:
        for raw in data.split(b"\n"):
            raw = raw.strip()
            if not raw or raw.startswith(_PI_STREAM_PREFIXES_B):
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                self.feed(obj)

    def result(self) -> list[LogTurn]:
        turns = list(self.turns)
        if self.current_turn and self.current_turn.entries:
            # A copy: the cached parser keeps appending to the open turn while
            # another request may still be serialising this one.
            turns.append(
                replace(
                    self.current_turn,
                    entries=list(self.current_turn.entries),
                    usage=dict(self.current_turn.usage),
                )
            )
        return turns

    def feed(self, obj: dict[str, Any]) -> None:
        turns = self.turns
        current_turn = self.current_turn
        turn_index = self.turn_index
        session_meta = self.session_meta
        try:
            msg_type = obj.get("type", "")

            if msg_type == "coral":
                # CORAL prompt injection (initial instruction, heartbeat, restart)
                subtype = obj.get("subtype", "")
                if subtype == "prompt":
                    source = obj.get("source", "")
                    prompt_text = obj.get("prompt", "")
                    timestamp = obj.get("timestamp", "")
                    details: dict[str, Any] = {"source": source}
                    task_name = obj.get("task_name")
                    task_description = obj.get("task_description")
                    if task_name:
                        details["task_name"] = task_name
                    if task_description:
                        details["task_description"] = task_description
                    # For start/restart, show task description instead of bare "Begin."
                    if source in ("start", "restart") and task_description:
                        display_content = task_description
                    else:
                        display_content = prompt_text
                    entry = LogEntry(
                        type="coral_prompt",
                        content=display_content,
                        details=details,
                        timestamp=timestamp,
                    )
                    # Start a new turn for the coral prompt
                    if current_turn and current_turn.entries:
                        turns.append(current_turn)
                    current_turn = LogTurn(index=turn_index, timestamp=timestamp)
                    turn_index += 1
                    current_turn.entries.append(entry)
                return

            if msg_type == "message_end":
                # Pi: one event per finished message. User messages are the
                # prompts CORAL sent, already logged as "coral" lines above.
                message = obj.get("message", {})
                role = message.get("role")
                timestamp = _ms_to_iso(message.get("timestamp"))
                if role == "assistant":
                    if current_turn and current_turn.entries:
                        turns.append(current_turn)
                    current_turn = LogTurn(index=turn_index, timestamp=timestamp)
                    turn_index += 1
                    current_turn.entries.extend(_extract_pi_blocks(message.get("content", [])))
                    usage = message.get("usage") or {}
                    current_turn.usage = {
                        "input_tokens": usage.get("input", 0),
                        "output_tokens": usage.get("output", 0),
                        "cache_creation": usage.get("cacheWrite", 0),
                        "cache_read": usage.get("cacheRead", 0),
                    }
                elif role == "toolResult":
                    text = "\n".join(
                        c.get("text", "")
                        for c in message.get("content", [])
                        if isinstance(c, dict) and c.get("type") == "text"
                    )
                    if current_turn is None:
                        current_turn = LogTurn(index=turn_index, timestamp=timestamp)
                        turn_index += 1
                    current_turn.entries.append(
                        LogEntry(
                            type="tool_result",
                            content=text,
                            details={
                                "tool_use_id": message.get("toolCallId", ""),
                                "is_error": bool(message.get("isError")),
                            },
                            timestamp=timestamp,
                        )
                    )
                return

            if msg_type == "system":
                subtype = obj.get("subtype", "")

                if subtype == "init":
                    model = obj.get("model", "")
                    session_id = obj.get("session_id", "")
                    entry = LogEntry(
                        type="system",
                        content=f"Session started (model: {model})",
                        details={
                            "session_id": session_id,
                            "model": model,
                            "tools": obj.get("tools", []),
                            "skills": obj.get("skills", []),
                            "agents": obj.get("agents", []),
                            "plugins": [
                                p.get("name", "") if isinstance(p, dict) else str(p)
                                for p in obj.get("plugins", [])
                            ],
                            "claude_code_version": obj.get("claude_code_version", ""),
                        },
                    )
                    if current_turn is None:
                        current_turn = LogTurn(index=turn_index)
                        turn_index += 1
                    current_turn.entries.append(entry)

                elif subtype == "task_started":
                    # Subagent launched
                    entry = LogEntry(
                        type="subagent_start",
                        content=obj.get("description", ""),
                        details={
                            "task_id": obj.get("task_id", ""),
                            "prompt": obj.get("prompt", ""),
                            "task_type": obj.get("task_type", ""),
                        },
                    )
                    if current_turn is None:
                        current_turn = LogTurn(index=turn_index)
                        turn_index += 1
                    current_turn.entries.append(entry)

                elif subtype == "task_progress":
                    # Subagent progress — tool calls etc.
                    task_usage = obj.get("usage", {})
                    entry = LogEntry(
                        type="subagent_progress",
                        content=obj.get("last_tool_name", ""),
                        details={
                            "task_id": obj.get("task_id", ""),
                            "description": obj.get("description", ""),
                            "tool_uses": task_usage.get("tool_uses", 0),
                        },
                    )
                    if current_turn is None:
                        current_turn = LogTurn(index=turn_index)
                        turn_index += 1
                    current_turn.entries.append(entry)

                elif subtype == "task_notification":
                    # Subagent completed
                    task_usage = obj.get("usage", {})
                    entry = LogEntry(
                        type="subagent_done",
                        content=obj.get("summary", ""),
                        details={
                            "task_id": obj.get("task_id", ""),
                            "status": obj.get("status", ""),
                            "total_tokens": task_usage.get("total_tokens", 0),
                            "tool_uses": task_usage.get("tool_uses", 0),
                            "duration_ms": task_usage.get("duration_ms", 0),
                        },
                    )
                    if current_turn is None:
                        current_turn = LogTurn(index=turn_index)
                        turn_index += 1
                    current_turn.entries.append(entry)

                elif subtype == "compact_boundary":
                    # Context compaction boundary
                    meta = obj.get("compact_metadata", {})
                    entry = LogEntry(
                        type="compact",
                        content=f"Context compacted ({meta.get('trigger', 'auto')})",
                        details={"pre_tokens": meta.get("pre_tokens", 0)},
                    )
                    if current_turn and current_turn.entries:
                        turns.append(current_turn)
                    current_turn = LogTurn(index=turn_index)
                    turn_index += 1
                    current_turn.entries.append(entry)

                # system:status silently ignored (just "compacting" state changes)

            elif msg_type == "assistant":
                message = obj.get("message", {})
                content = message.get("content", [])
                usage = message.get("usage", {})

                parsed_usage = {}
                if usage:
                    parsed_usage = {
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "cache_creation": usage.get("cache_creation_input_tokens", 0),
                        "cache_read": usage.get("cache_read_input_tokens", 0),
                    }

                # Same usage as current turn = same API response logged as
                # multiple NDJSON lines (one per content block). Merge them.
                if current_turn and current_turn.usage and parsed_usage == current_turn.usage:
                    if isinstance(content, list):
                        current_turn.entries.extend(_extract_content_blocks(content))
                else:
                    # New API call — start a new turn
                    if current_turn and current_turn.entries:
                        turns.append(current_turn)
                    current_turn = LogTurn(index=turn_index)
                    turn_index += 1
                    if isinstance(content, list):
                        current_turn.entries.extend(_extract_content_blocks(content))
                    if parsed_usage:
                        current_turn.usage = parsed_usage

            elif msg_type == "user":
                # Tool results come as user messages
                message = obj.get("message", {})
                content = message.get("content", [])
                if isinstance(content, list):
                    entries = _extract_content_blocks(content)
                    if current_turn:
                        current_turn.entries.extend(entries)

            elif msg_type == "result":
                # Final result — extract full metadata
                result_text = obj.get("result", "")
                result_usage = obj.get("usage", {})
                model_usage = obj.get("modelUsage", {})
                session_meta = SessionMeta(
                    total_cost_usd=obj.get("total_cost_usd"),
                    duration_ms=obj.get("duration_ms"),
                    duration_api_ms=obj.get("duration_api_ms"),
                    num_turns=obj.get("num_turns"),
                    stop_reason=obj.get("stop_reason", ""),
                    session_id=obj.get("session_id", ""),
                    usage=result_usage,
                    model_usage=model_usage,
                )
                # Add result entry to current turn
                entry = LogEntry(
                    type="result",
                    content=str(result_text)[:500] if result_text else "",
                    details={
                        "total_cost_usd": obj.get("total_cost_usd"),
                        "duration_ms": obj.get("duration_ms"),
                        "num_turns": obj.get("num_turns"),
                        "stop_reason": obj.get("stop_reason", ""),
                    },
                )
                if current_turn is None:
                    current_turn = LogTurn(index=turn_index)
                    turn_index += 1
                current_turn.entries.append(entry)
        finally:
            self.current_turn = current_turn
            self.turn_index = turn_index
            self.session_meta = session_meta


def parse_log_file(path: Path, offset: int = 0) -> tuple[list[LogTurn], int, SessionMeta | None]:
    """Parse a Claude Code or Pi NDJSON log file into structured turns.

    Args:
        path: Path to the log file.
        offset: Byte offset to start reading from (for incremental updates).

    Returns:
        (list of LogTurn, new byte offset, session metadata from result message)
    """
    if not path.exists():
        return [], 0, None
    if offset >= path.stat().st_size:
        return [], offset, None
    parser = LogParser(offset)
    parser.read(path)
    return parser.result(), parser.offset, parser.session_meta


_PARSERS: dict[str, tuple[int, LogParser]] = {}
_PARSERS_LOCK = threading.Lock()


def parse_log_cached(path: Path) -> tuple[list[LogTurn], SessionMeta | None]:
    """Like ``parse_log_file`` from offset 0, but reads only bytes added since the last call.

    Parsers are cached per path; a file that shrank or was replaced (new inode)
    is re-parsed from the start. Safe to call from worker threads.
    """
    key = str(path)
    try:
        stat = path.stat()
    except OSError:
        return [], None
    with _PARSERS_LOCK:
        inode, parser = _PARSERS.get(key, (None, None))
        if parser is None or inode != stat.st_ino or stat.st_size < parser.offset:
            parser = LogParser()
            _PARSERS[key] = (stat.st_ino, parser)
        if stat.st_size > parser.offset:
            parser.read(path)
        return parser.result(), parser.session_meta


def list_log_files(coral_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """List all log files grouped by agent ID.

    Returns:
        {agent_id: [{path, index, size_bytes, modified}]}
    """
    from coral.hub._island import all_view_roots

    agents: dict[str, list[dict[str, Any]]] = {}
    multi_island = (coral_dir / "islands").exists()
    for view_root in all_view_roots(coral_dir):
        logs_dir = view_root / "logs"
        if not logs_dir.exists():
            continue
        for log_file in sorted(logs_dir.glob("*.log")):
            parts = log_file.stem.rsplit(".", 1)
            agent_id = parts[0] if len(parts) == 2 else log_file.stem
            index = int(parts[1]) if len(parts) == 2 else 0
            stat = log_file.stat()
            entry = {
                "path": str(log_file),
                "index": index,
                "size_bytes": stat.st_size,
                "modified": stat.st_mtime,
            }
            if multi_island:
                entry["island_id"] = view_root.name
            agents.setdefault(agent_id, []).append(entry)

    for logs in agents.values():
        logs.sort(key=lambda x: (x["modified"], x["index"], x["path"]))

    return agents


_ASSISTANT_END = b'{"type":"message_end","message":{"role":"assistant"'
# path -> (inode, bytes scanned, context tokens of the last finished model call)
_CONTEXT: dict[str, tuple[int, int, int]] = {}
_CONTEXT_LOCK = threading.Lock()


def live_state(path: Path, tail_bytes: int = 256 << 10) -> dict[str, Any] | None:
    """What a Pi agent is doing right now, from the end of its log.

    Returns ``{"state": ..., "since": iso, "context_tokens": n}`` where state is
    ``waiting`` (request sent, no tokens back yet: the model is reading the
    prompt), ``streaming`` (with the partial thinking/text so far), ``tool``
    (running a tool), ``between`` or ``idle``. Reads the last ``tail_bytes``,
    widening the window only while it holds no complete event (a single
    streaming update can be large); the context size comes from a per-file
    cache that only scans bytes added since the previous call.
    """
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            window = tail_bytes
            while True:
                f.seek(max(0, size - window))
                lines = f.read(window).decode("utf-8", errors="replace").splitlines()
                state = _classify(lines)
                if state is not None or window >= size or window >= 64 << 20:
                    break
                window *= 4
    except OSError:
        return None
    state = state or {"state": "idle", "since": ""}
    state["context_tokens"] = _context_tokens(path)
    return state


def _context_tokens(path: Path, chunk_size: int = 8 << 20) -> int:
    """Prompt + output tokens of the last finished assistant call, scanning only new bytes."""
    key = str(path)
    try:
        stat = path.stat()
    except OSError:
        return 0
    with _CONTEXT_LOCK:
        inode, scanned, tokens = _CONTEXT.get(key, (stat.st_ino, 0, 0))
        if inode != stat.st_ino or stat.st_size < scanned:
            scanned, tokens = 0, 0
        with open(path, "rb") as f:
            f.seek(scanned)
            carry = b""
            while chunk := f.read(chunk_size):
                data = carry + chunk
                cut = data.rfind(b"\n") + 1
                complete, carry = data[:cut], data[cut:]
                scanned += cut
                pos = complete.rfind(b"\n" + _ASSISTANT_END)
                if pos < 0 and complete.startswith(_ASSISTANT_END):
                    pos = -1
                elif pos < 0:
                    continue
                line_end = complete.find(b"\n", pos + 1)
                line = complete[pos + 1 : line_end if line_end >= 0 else None]
                try:
                    usage = json.loads(line)["message"].get("usage") or {}
                    tokens = int(
                        usage.get("input", 0) + usage.get("cacheRead", 0) + usage.get("output", 0)
                    )
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    pass
        _CONTEXT[key] = (stat.st_ino, scanned, tokens)
        return tokens


def _classify(lines: list[str]) -> dict[str, Any] | None:
    for line in reversed(lines):
        line = line.strip()
        if not line.startswith('{"type":"'):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue  # the first line of the tail can be cut mid-way
        kind = obj.get("type")
        if kind == "message_update":
            partial = (obj.get("assistantMessageEvent") or {}).get("partial") or {}
            entries = _extract_pi_blocks(partial.get("content", []))
            return {
                "state": "streaming",
                "since": _ms_to_iso(partial.get("timestamp")),
                "entries": [
                    {"type": e.type, "content": e.content, "details": e.details} for e in entries
                ],
            }
        if kind == "message_start" and obj.get("message", {}).get("role") == "assistant":
            # Request sent, no tokens yet: the model is reading the prompt.
            return {"state": "waiting", "since": _ms_to_iso(obj["message"].get("timestamp"))}
        if kind in ("tool_execution_start", "tool_execution_update"):
            return {"state": "tool", "tool": obj.get("toolName", ""), "since": ""}
        if kind == "agent_end":
            return {"state": "idle", "since": ""}
        if kind in ("message_start", "message_end", "tool_execution_end", "turn_start", "turn_end"):
            # Between model calls: tool results or a prompt are about to be sent.
            return {"state": "between", "since": ""}
    return None
