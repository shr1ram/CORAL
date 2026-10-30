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
typed messages: `FINDING|WARNING|REQUEST|REPLY <agent|all>: <text>`. Parsing is
strict; lines that look like one of these but do not match are reported back
in the next prompt, as are failed attempts (no code, cut-off reply, eval error).

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
    "Your last {n} attempts did not beat the team's best score. Small tweaks are not working: "
    "try a substantially different approach (a new algorithm, representation or search strategy)."
)
PLATEAU_ATTEMPTS = 3
PARTIAL_TAIL_CHARS = 2000  # per content block in progress events
PROGRESS_EVERY_S = 5.0


def direction_history(attempts: list, agent_id: str, minimize: bool) -> list[tuple]:
    """(score, team best before it) for each of the agent's attempts, in order.

    ``attempts`` must be sorted by time. Agents start from the team's best, so
    progress is measured against it rather than against the agent's own best.
    """
    out, team_best = [], None
    for a in attempts:
        if a.agent_id == agent_id:
            out.append((a.score, team_best))
        if a.score is not None and (
            team_best is None or (a.score < team_best if minimize else a.score > team_best)
        ):
            team_best = a.score
    return out


def choose_direction(history: list[tuple], agent_id: str, minimize: bool) -> str:
    """Next attempt's direction from ``direction_history``.

    Rotates through DIRECTIONS by attempt count (offset per agent, so agents
    differ). After every PLATEAU_ATTEMPTS consecutive attempts that failed to
    beat the team's best at the time, the next attempt pivots once, then the
    rotation resumes. Stateless, so it survives worker restarts.
    """
    streak = 0
    for score, best_before in reversed(history):
        beat = score is not None and (
            best_before is None or (score < best_before if minimize else score > best_before)
        )
        if beat:
            break
        streak += 1
    if streak and streak % PLATEAU_ATTEMPTS == 0:
        return PIVOT.format(n=PLATEAU_ATTEMPTS)
    return DIRECTIONS[(len(history) + sum(map(ord, agent_id))) % len(DIRECTIONS)]


_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_TITLE_RE = re.compile(r"^\s*TITLE:\s*(.+)$", re.MULTILINE)
_STATUS_RE = re.compile(r"^\s*STATUS:\s*(.+)$", re.MULTILINE)
_RECALL_RE = re.compile(r"^\s*RECALL:\s*(.+)$", re.MULTILINE)
# Typed messages: "<TYPE> <recipient>: <text>"
MESSAGE_TYPES = ("FINDING", "WARNING", "REQUEST", "REPLY")
_MESSAGE_RE = re.compile(
    r"^\s*(" + "|".join(MESSAGE_TYPES) + r")\s+([\w.-]+)\s*:\s*(.+)$", re.MULTILINE
)
_KEYWORDS = ("TITLE", "STATUS", "RECALL", *MESSAGE_TYPES)
# A line that starts (after bullets, quotes or emphasis) with one of the keywords.
_LOOSE_RE = re.compile(r"^[\s>*`_\-\d.]*(" + "|".join(_KEYWORDS) + r")\b")
_STRICT = (_TITLE_RE, _STATUS_RE, _RECALL_RE, _MESSAGE_RE)


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
    malformed: list[str] = field(default_factory=list)  # keyword lines that did not parse


def parse_reply(text: str) -> Reply:
    """Program (last fenced block), title, status, recall and typed messages from a reply."""
    blocks = _FENCE_RE.findall(text)
    outside = _FENCE_RE.sub("", text)

    def first(regex: re.Pattern) -> str | None:
        match = regex.search(outside)
        return match.group(1).strip() if match else None

    malformed = [
        line.strip()
        for line in outside.splitlines()
        if _LOOSE_RE.match(line) and not any(r.match(line) for r in _STRICT)
    ]
    return Reply(
        program=blocks[-1] if blocks else None,
        title=(first(_TITLE_RE) or "one-shot attempt")[:120],
        status=first(_STATUS_RE),
        recall=first(_RECALL_RE),
        messages=[(k.lower(), to, body.strip()) for k, to, body in _MESSAGE_RE.findall(outside)],
        malformed=malformed,
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
    last_failure: str | None = None,
) -> str:
    parts = [task.strip()]
    for name, text in context_files.items():
        parts.append(f"## {name}\n\n{text.strip()}")
    parts.append(f"## Current program ({target}): {program_note}\n\n```{language}\n{program}\n```")
    if recent:
        parts.append("## Recent attempts by the team (newest last)\n\n" + "\n".join(recent))
    if last_failure:
        parts.append("## Your last attempt failed\n\n" + last_failure)
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
            "## Lines from your last reply that were NOT delivered\n\n" + "\n".join(undelivered)
        )

    lines = [
        "Reply in this format. Write each line exactly as shown, at the start of its own line, "
        "with no bullet, quotes or bold:",
        "",
        "TITLE: <what you changed, in a few words>",
        "STATUS: <what you are working on, one line>"
        + ("   (optional; teammates see it)" if cooperate else "   (optional)"),
    ]
    if memory:
        lines.append(
            "RECALL: <a question for the team memory>   (optional; the librarian answers it in "
            "your next memory brief)"
        )
    if cooperate:
        others = [t for t in (teammates or []) if t != agent_id]
        names = f"`all` or one of: {', '.join(others)}" if others else "`all`"
        lines += [
            "FINDING all: <something you learned>",
            "WARNING <recipient>: <a pitfall to avoid>",
            "REQUEST <recipient>: <ask a teammate for something>",
            "REPLY <recipient>: <answer a request>",
            "",
            "Message lines are optional, one per line: FINDING (something you learned), "
            "WARNING (a pitfall to avoid), REQUEST (ask a teammate for something) or REPLY "
            f"(answer a request); the recipient is {names}.",
        ]
    lines += [
        "",
        f"Then the complete new {target} in a single ```{language} block. The whole file is "
        "replaced by your block, so include all of it.",
    ]
    parts.append(f"## Your task now\n\n{direction}\n\n" + "\n".join(lines))
    return "\n\n".join(parts)


def _tail(text: str) -> str:
    return text[-PARTIAL_TAIL_CHARS:]


def chat_stream(
    base_url: str, model: str, prompt: str, max_tokens: int, temperature: float
) -> tuple[str, str, dict[str, int], str | None]:
    """Stream one chat completion, emitting progress events.

    Returns (reasoning, text, usage, finish_reason). Progress events carry
    character counts and only the tail of each block, so the log grows
    linearly with the reply rather than quadratically.
    """
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
    reasoning, text, usage, finish = [], [], {}, None
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
                finish = choice.get("finish_reason") or finish
            if time.time() - last_update > PROGRESS_EVERY_S:
                last_update = time.time()
                r, t = "".join(reasoning), "".join(text)
                _emit(
                    {
                        "type": "message_update",
                        "progress": {"reasoning_chars": len(r), "text_chars": len(t)},
                        "assistantMessageEvent": {
                            "partial": {
                                "content": _content(_tail(r), _tail(t)),
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
    return "".join(reasoning), "".join(text), tokens, finish


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


def send_messages(
    coral_dir: Path,
    agent_id: str,
    teammates: list[str],
    messages: list[tuple[str, str, str]],
    island_id: str | None,
) -> list[str]:
    """Send typed messages in-process, with `coral msg`'s checks; returns undelivered lines."""
    from coral.hub.messages import BROADCAST, send

    undelivered = []
    others = [t for t in teammates if t != agent_id]
    for kind, to, body in messages:
        if to == agent_id:
            reason = "you can't message yourself."
        elif to != BROADCAST and to not in others:
            reason = f"unknown agent '{to}'. Known: {', '.join(others)}, or 'all'."
        else:
            try:
                send(coral_dir, agent_id, to, body, island_id=island_id, kind=kind)
                _tool_result(f"message {kind} {to}", f"Sent to {to}.", False)
                continue
            except ValueError as e:
                reason = str(e)
        _tool_result(f"message {kind} {to}", reason, True)
        undelivered.append(f"{kind.upper()} {to}: {body[:100]}  -> not delivered: {reason}")
    return undelivered


def _seed_program(target: str) -> str | None:
    """The target file at the run's first commit (the task's starting program)."""
    roots = subprocess.run(
        ["git", "rev-list", "--max-parents=0", "HEAD"], capture_output=True, text=True
    )
    if roots.returncode != 0 or not roots.stdout.strip():
        return None
    shown = subprocess.run(
        ["git", "show", f"{roots.stdout.split()[0]}:{target}"], capture_output=True, text=True
    )
    return shown.stdout if shown.returncode == 0 else None


def _eval_failure(output: str) -> str:
    lowered = output.lower()
    if "nothing to commit" in lowered or "no changes" in lowered:
        return (
            "`coral eval` found no change: your program was identical to the current one. "
            "Make an actual change."
        )
    if "pending attempt" in lowered or "still pending" in lowered:
        return (
            "`coral eval` did not score your program: your previous attempt was still being "
            "graded, so this one was not submitted. Resubmit it if it is still worth trying."
        )
    first = next((ln for ln in output.splitlines() if ln.strip()), "eval failed")
    return f"`coral eval` failed: {first.removeprefix('Error: ')[:300]}"


def run(args: argparse.Namespace) -> None:
    from coral.cli._helpers import find_coral_dir_and_island, read_agent_id, read_direction

    worktree = Path.cwd()
    coral_dir, island_id = find_coral_dir_and_island()
    agent_id = read_agent_id()
    direction_name = read_direction(coral_dir)
    minimize = direction_name == "minimize"
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
    target_path = worktree / args.target
    undelivered: list[str] = []
    last_failure: str | None = None

    _emit({"type": "session", "id": uuid.uuid4().hex})
    while True:
        try:
            undelivered, last_failure = _attempt(
                args=args,
                coral_dir=coral_dir,
                island_id=island_id,
                agent_id=agent_id,
                minimize=minimize,
                direction_name=direction_name,
                cooperate=cooperate,
                memory=memory,
                task_text=task_text,
                context_files=context_files,
                target_path=target_path,
                undelivered=undelivered,
                last_failure=last_failure,
            )
        except Exception as e:  # never let one bad attempt kill the loop
            _tool_result("worker", f"attempt failed: {e!r}", True)
            time.sleep(10)


def _attempt(
    *,
    args,
    coral_dir,
    island_id,
    agent_id,
    minimize,
    direction_name,
    cooperate,
    memory,
    task_text,
    context_files,
    target_path,
    undelivered,
    last_failure,
) -> tuple[list[str], str | None]:
    """One attempt; returns (undelivered lines, failure note) for the next prompt."""
    from coral.hub._island import island_root
    from coral.hub.attempts import get_leaderboard, read_attempts
    from coral.hub.memory import read_brief
    from coral.hub.messages import open_requests, recent_for
    from coral.hub.status import read_board, read_status, status_changed, write_status

    # Start from the best attempt so far (any agent), else the task's seed program.
    best = get_leaderboard(str(coral_dir), top_n=1, direction=direction_name, island_id=island_id)
    if best:
        b = best[0]
        checkout = _coral("checkout", b.commit_hash)
        note = f"best so far, score {b.score} by {b.agent_id} ({b.commit_hash[:8]})"
        if checkout.returncode != 0:
            note += " [checkout failed; this is your own last program, not the best]"
        program = (
            target_path.read_text(encoding="utf-8", errors="replace")
            if target_path.exists()
            else ""
        )
    else:
        seed = _seed_program(args.target)
        if seed is not None:
            program, note = seed, "the task's starting program (no attempt has scored yet)"
        else:
            program = (
                target_path.read_text(encoding="utf-8", errors="replace")
                if target_path.exists()
                else ""
            )
            note = "your last program (no attempt has scored yet)"

    team = sorted(read_attempts(coral_dir, island_id), key=lambda a: a.timestamp)
    recent = []
    for a in team[-8:]:
        line = f"- {a.agent_id}: score {a.score} ({a.status}): {a.title}"
        if a.agent_id == agent_id and a.feedback:
            line += "\n  feedback: " + a.feedback.strip().replace("\n", "\n  ")[:1200]
        recent.append(line)

    teammates = sorted(p.stem for p in (island_root(coral_dir, island_id) / "roles").glob("*.md"))
    direction = choose_direction(direction_history(team, agent_id, minimize), agent_id, minimize)
    board, history, requests = [], {}, []
    if cooperate:
        best_of: dict[str, float] = {}
        for a in team:
            if a.score is not None:
                prev = best_of.get(a.agent_id)
                if prev is None or (a.score < prev if minimize else a.score > prev):
                    best_of[a.agent_id] = a.score
        for other, record in read_board(coral_dir, island_id).items():
            if other != agent_id:
                board.append(
                    f"- {other} (best {best_of.get(other)}): {record.get('status') or '-'}"
                )
        open_reqs = open_requests(coral_dir, agent_id, island_id)[-5:]
        requests = [f"- {r.sender}: {r.body}" for r in open_reqs]
        listed = {r.id for r in open_reqs}
        for m in recent_for(coral_dir, agent_id, 10, island_id):
            if m.id in listed:
                continue  # already under open requests
            who = "you" if m.sender == agent_id else m.sender
            history.setdefault(m.kind, []).append(f"- {who} -> {m.to}: {m.body}")
    brief = (read_brief(coral_dir, agent_id, island_id) or {}).get("text") if memory else None

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
        last_failure=last_failure,
    )
    _emit({"type": "turn_start"})
    try:
        reasoning, text, usage, finish = chat_stream(
            args.base_url, args.model, prompt, args.max_tokens, args.temperature
        )
    except Exception as e:  # server restarting, connection reset, ...
        _tool_result("model", f"model call failed: {e}", True)
        time.sleep(10)
        return undelivered, last_failure

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
    undelivered = [
        f"{line[:120]}  -> not understood: write it exactly as "
        f"`{_LOOSE_RE.match(line).group(1)} ...` at the start of a line"
        for line in reply.malformed
    ]
    if cooperate:
        undelivered += send_messages(coral_dir, agent_id, teammates, reply.messages, island_id)

    failure = None
    if not reply.program or not reply.program.strip():
        failure = (
            "Your reply had no ```"
            + (args.language or "")
            + " code block, so nothing was evaluated."
        )
        if finish == "length":
            failure += (
                f" It was cut off at the {args.max_tokens}-token limit (finish_reason=length): "
                "think less and write the code sooner."
            )
        _tool_result("write", failure, True)
    else:
        target_path.write_text(reply.program, encoding="utf-8")
        _emit({"type": "tool_execution_start", "toolName": "coral eval"})
        result = _coral("eval", "-m", reply.title)
        output = (result.stdout + result.stderr).strip()
        _tool_result("coral eval", output, result.returncode != 0)
        if result.returncode != 0:
            failure = _eval_failure(output)

    # After the eval, so the board and the librarian's next brief see this attempt.
    previous = read_status(coral_dir, agent_id, island_id)
    if status_changed(previous, reply.status, reply.recall):
        write_status(
            coral_dir,
            agent_id,
            status=reply.status,
            direction=direction,
            recall=reply.recall or "",
            island_id=island_id,
        )
    return undelivered, failure


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
