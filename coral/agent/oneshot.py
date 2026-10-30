"""One-shot agent loop (agents.loop = "oneshot").

Each attempt is a single model call with no tools: the prompt holds the task,
the context files, the best program so far and the team's recent results, and
the reply must contain the complete new program file. The worker writes that
file, starts from the best attempt's commit (so lineage stays truthful), and
runs `coral eval`.

Replies may add `STATUS: ...` (what the agent is working on) and, with
agents.memory, `RECALL: ...` (a question for the team memory); the prompt then
carries the librarian's memory brief (coral/agent/memory_service.py). In
cooperate mode the prompt also shows teammates' statuses, the last team
messages grouped by type and open requests to this agent, and replies can send
typed messages: `FINDING|WARNING|REQUEST|REPLY <agent|all>: <text>`.

Runs as a subprocess in the agent's worktree (see builtin/oneshot.py) and logs
Pi-format JSON events to stdout so the dashboard parses it like a Pi agent.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Rotated per attempt so agents explore instead of all making the same tweak.
DIRECTIONS = (
    "Refine the current program with one concrete improvement.",
    "Try a different algorithm or representation.",
    "Simplify the program so more search fits in the time limit.",
    "Question an assumption the current program makes and test a different choice.",
)
# One-shot agents get no CORAL heartbeats; this is their pivot.
PIVOT = (
    "Your last {n} attempts did not beat your best score. Small tweaks are not working: try a "
    "substantially different approach (a new algorithm, representation or search strategy)."
)
PLATEAU_ATTEMPTS = 3


def choose_direction(own_scores: list[float | None], agent_id: str, minimize: bool) -> str:
    """Next attempt's direction, from the agent's own scores in submit order.

    Rotates through DIRECTIONS by attempt count (offset per agent, so agents
    differ), and switches to PIVOT once the last PLATEAU_ATTEMPTS attempts all
    failed to beat the agent's best before them. Derived from attempt history,
    so it survives worker restarts.
    """
    n = len(own_scores)
    if n > PLATEAU_ATTEMPTS:
        before = [s for s in own_scores[:-PLATEAU_ATTEMPTS] if s is not None]
        recent = own_scores[-PLATEAU_ATTEMPTS:]
        if before:
            best = min(before) if minimize else max(before)
            improved = any(s is not None and (s < best if minimize else s > best) for s in recent)
            if not improved:
                return PIVOT.format(n=PLATEAU_ATTEMPTS)
    return DIRECTIONS[(n + sum(map(ord, agent_id))) % len(DIRECTIONS)]


_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_TITLE_RE = re.compile(r"^\s*TITLE:\s*(.+)$", re.MULTILINE)
_STATUS_RE = re.compile(r"^\s*STATUS:\s*(.+)$", re.MULTILINE)
_RECALL_RE = re.compile(r"^\s*RECALL:\s*(.+)$", re.MULTILINE)
# Typed messages: "<TYPE> <recipient>: <text>"
MESSAGE_TYPES = ("FINDING", "WARNING", "REQUEST", "REPLY")
_MESSAGE_RE = re.compile(
    r"^\s*(" + "|".join(MESSAGE_TYPES) + r")\s+([\w.-]+)\s*:\s*(.+)$", re.MULTILINE
)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _emit(event: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(event, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _coral(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "coral.cli", *args], capture_output=True, text=True
    )


@dataclass
class Reply:
    program: str | None
    title: str
    status: str | None = None
    recall: str | None = None
    messages: list[tuple[str, str, str]] = field(default_factory=list)  # (kind, to, body)


def parse_reply(text: str) -> Reply:
    """Program (last fenced block), title, status, recall and typed messages from a reply."""
    blocks = _FENCE_RE.findall(text)
    outside = _FENCE_RE.sub("", text)

    def first(regex: re.Pattern) -> str | None:
        match = regex.search(outside)
        return match.group(1).strip() if match else None

    return Reply(
        program=blocks[-1] if blocks else None,
        title=(first(_TITLE_RE) or "one-shot attempt")[:120],
        status=first(_STATUS_RE),
        recall=first(_RECALL_RE),
        messages=[(k.lower(), to, body.strip()) for k, to, body in _MESSAGE_RE.findall(outside)],
    )


def build_prompt(
    *,
    task: str,
    context_files: dict[str, str],
    target: str,
    language: str,
    program: str,
    program_note: str,
    recent: list[str],
    direction: str,
    cooperate: bool,
    agent_id: str,
    teammates: list[str] | None = None,
    board: list[str] | None = None,
    history: dict[str, list[str]] | None = None,
    requests: list[str] | None = None,
    undelivered: list[str] | None = None,
    brief: str | None = None,
    memory: bool = False,
) -> str:
    parts = [task.strip()]
    for name, text in context_files.items():
        parts.append(f"## {name}\n\n{text.strip()}")
    parts.append(f"## Current program ({target}): {program_note}\n\n```{language}\n{program}\n```")
    if recent:
        parts.append("## Recent attempts by the team (newest last)\n\n" + "\n".join(recent))
    if brief:
        parts.append("## Memory brief from the team librarian\n\n" + brief.strip())
    if board:
        parts.append("## What teammates are working on\n\n" + "\n".join(board))
    if requests:
        parts.append("## Open requests to you (answer with REPLY)\n\n" + "\n".join(requests))
    for kind, lines in (history or {}).items():
        if lines:
            parts.append(f"## Recent {kind.lower()}s from the team\n\n" + "\n".join(lines))
    if undelivered:
        parts.append(
            "## Your messages that were NOT delivered last attempt\n\n" + "\n".join(undelivered)
        )

    lines = [
        f"Reply with a first line `TITLE: <what you changed, in a few words>`, then the complete "
        f"new {target} in a single ```{language} block. The whole file is replaced by your "
        "block, so include all of it.",
        "Outside the code block you may add:",
        "- `STATUS: <what you are working on, one line>`"
        + (" (teammates see it)" if cooperate else ""),
    ]
    if memory:
        lines.append(
            "- `RECALL: <a question for the team memory>`: the librarian answers it in your next "
            "memory brief"
        )
    if cooperate:
        others = [t for t in (teammates or []) if t != agent_id]
        names = f"`all` or one of: {', '.join(others)}" if others else "`all`"
        lines.append(
            "- messages to teammates, one per line as `<TYPE> <recipient>: <text>`, where TYPE is "
            "FINDING (something you learned), WARNING (a pitfall to avoid), REQUEST (ask a "
            f"teammate for something) or REPLY (answer a request), and the recipient is {names}"
        )
    parts.append(f"## Your task now\n\n{direction}\n\n" + "\n".join(lines))
    return "\n\n".join(parts)


def chat_stream(
    base_url: str, model: str, prompt: str, max_tokens: int, temperature: float
) -> tuple[str, str, dict[str, int]]:
    """Stream one chat completion, emitting progress events. Returns (reasoning, text, usage)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
    )
    started = _now_ms()
    _emit({"type": "message_start", "message": {"role": "assistant", "timestamp": started}})
    reasoning, text, usage = [], [], {}
    last_update = 0.0
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                reasoning.append(delta.get("reasoning_content") or "")
                text.append(delta.get("content") or "")
            if time.time() - last_update > 2:  # throttle: each update repeats the partial
                last_update = time.time()
                _emit(
                    {
                        "type": "message_update",
                        "assistantMessageEvent": {
                            "partial": {
                                "content": _content("".join(reasoning), "".join(text)),
                                "timestamp": started,
                            }
                        },
                    }
                )
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    tokens = {
        "input": usage.get("prompt_tokens", 0) - cached,
        "output": usage.get("completion_tokens", 0),
        "cacheRead": cached,
        "cacheWrite": 0,
    }
    return "".join(reasoning), "".join(text), tokens


def _content(reasoning: str, text: str) -> list[dict[str, str]]:
    content = []
    if reasoning:
        content.append({"type": "thinking", "thinking": reasoning})
    if text:
        content.append({"type": "text", "text": text})
    return content


def _tool_result(name: str, output: str, is_error: bool) -> None:
    _emit(
        {
            "type": "message_end",
            "message": {
                "role": "toolResult",
                "toolName": name,
                "content": [{"type": "text", "text": output}],
                "isError": is_error,
                "timestamp": _now_ms(),
            },
        }
    )


def run(args: argparse.Namespace) -> None:
    from coral.cli._helpers import find_coral_dir_and_island, read_agent_id, read_direction
    from coral.hub._island import island_root
    from coral.hub.attempts import get_leaderboard, read_attempts
    from coral.hub.memory import read_brief
    from coral.hub.messages import open_requests, recent_for
    from coral.hub.status import read_board, write_status

    worktree = Path.cwd()
    coral_dir, island_id = find_coral_dir_and_island()
    agent_id = read_agent_id()
    direction_name = read_direction(coral_dir)
    config = _read_yaml(coral_dir / "config.yaml")
    agents_cfg = config.get("agents") or {}
    cooperate = agents_cfg.get("mode") == "cooperate"
    memory = bool(agents_cfg.get("memory", True))
    task = config.get("task") or {}
    task_text = f"# {task.get('name', '')}\n\n{task.get('description', '')}"
    if task.get("tips"):
        task_text += f"\n\n## Tips\n\n{task['tips']}"
    context_files = {
        name: (worktree / name).read_text(encoding="utf-8", errors="replace")
        for name in args.context
        if (worktree / name).exists()
    }
    undelivered: list[str] = []

    _emit({"type": "session", "id": uuid.uuid4().hex})
    while True:
        # Start from the best attempt so far (any agent), else the current file.
        best = get_leaderboard(
            str(coral_dir), top_n=1, direction=direction_name, island_id=island_id
        )
        target_path = worktree / args.target
        if best:
            b = best[0]
            checkout = _coral("checkout", b.commit_hash)
            shown = subprocess.run(
                ["git", "show", f"{b.commit_hash}:{args.target}"], capture_output=True, text=True
            )
            program = shown.stdout if shown.returncode == 0 else target_path.read_text()
            note = f"best so far, score {b.score} by {b.agent_id} ({b.commit_hash[:8]})"
            if checkout.returncode != 0:
                note += " [checkout failed; your edit applies on top of your own last commit]"
        else:
            program = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
            note = "starting program, not yet scored"

        team = sorted(read_attempts(coral_dir, island_id), key=lambda a: a.timestamp)
        own_scores = [a.score for a in team if a.agent_id == agent_id]
        recent = []
        for a in team[-8:]:
            line = f"- {a.agent_id}: score {a.score} ({a.status}): {a.title}"
            if a.agent_id == agent_id and a.feedback:
                line += "\n  feedback: " + a.feedback.strip().replace("\n", "\n  ")[:1200]
            recent.append(line)

        teammates = sorted(
            p.stem for p in (island_root(coral_dir, island_id) / "roles").glob("*.md")
        )
        direction = choose_direction(own_scores, agent_id, direction_name == "minimize")
        board, history, requests = [], {}, []
        if cooperate:
            best_of = {}
            for a in team:
                if a.score is not None:
                    prev = best_of.get(a.agent_id)
                    better = prev is None or (
                        a.score < prev if direction_name == "minimize" else a.score > prev
                    )
                    best_of[a.agent_id] = a.score if better else prev
            for other, record in read_board(coral_dir, island_id).items():
                if other != agent_id:
                    board.append(
                        f"- {other} (best {best_of.get(other)}): {record.get('status') or '-'}"
                    )
            for m in recent_for(coral_dir, agent_id, 10, island_id):
                who = "you" if m.sender == agent_id else m.sender
                history.setdefault(m.kind, []).append(f"- {who} -> {m.to}: {m.body}")
            requests = [
                f"- {r.sender}: {r.body}" for r in open_requests(coral_dir, agent_id, island_id)
            ][-5:]
        brief = None
        if memory:
            brief = (read_brief(coral_dir, agent_id, island_id) or {}).get("text")

        prompt = build_prompt(
            task=task_text,
            context_files=context_files,
            target=args.target,
            language=args.language,
            program=program,
            program_note=note,
            recent=recent,
            direction=direction,
            cooperate=cooperate,
            agent_id=agent_id,
            teammates=teammates,
            board=board,
            history=history,
            requests=requests,
            undelivered=undelivered,
            brief=brief,
            memory=memory,
        )
        _emit({"type": "turn_start"})
        try:
            reasoning, text, usage = chat_stream(
                args.base_url, args.model, prompt, args.max_tokens, args.temperature
            )
        except Exception as e:  # server restarting, connection reset, ...
            _tool_result("model", f"model call failed: {e}", True)
            time.sleep(10)
            continue

        reply = parse_reply(text)
        _emit(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": _content(reasoning, text),
                    "usage": usage,
                    "timestamp": _now_ms(),
                },
            }
        )
        # Status (and any recall question) feed the board and the librarian.
        write_status(
            coral_dir,
            agent_id,
            status=reply.status,
            direction=direction,
            recall=reply.recall or "",
            island_id=island_id,
        )
        undelivered = []  # shown in the next prompt so the agent can fix the format
        if cooperate:
            for kind, to, body in reply.messages:
                sent = _coral("msg", "--type", kind, to, body)
                output = (sent.stdout + sent.stderr).strip()
                _tool_result(f"coral msg {kind} {to}", output, sent.returncode != 0)
                if sent.returncode != 0:
                    reason = output.removeprefix("Error: ").splitlines()[0] if output else "failed"
                    undelivered.append(f"- `{kind.upper()} {to}: {body[:100]}` -> {reason}")
        if not reply.program or not reply.program.strip():
            _tool_result("write", "No fenced code block in the reply; nothing to evaluate.", True)
            continue

        target_path.write_text(reply.program, encoding="utf-8")
        title = reply.title
        _emit({"type": "tool_execution_start", "toolName": "coral eval"})
        result = _coral("eval", "-m", title)
        _tool_result("coral eval", (result.stdout + result.stderr).strip(), result.returncode != 0)


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL (…/v1)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--target", required=True, help="Program file the model rewrites")
    parser.add_argument("--language", default="", help="Code fence language, e.g. cpp")
    parser.add_argument("--context", action="append", default=[], help="Extra file to include")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--prompt", default="", help="Prompt injected by the manager")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
