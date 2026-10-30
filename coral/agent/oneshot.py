"""One-shot agent loop (agents.loop = "oneshot").

Each attempt is a single model call with no tools: the prompt holds the task,
the context files, the best program so far and the team's recent results, and
the reply must contain the complete new program file. The worker writes that
file, starts from the best attempt's commit (so lineage stays truthful), and
runs `coral eval`. In cooperate mode the prompt also carries unread messages,
and the reply can send messages with `MESSAGE <agent|all>: <text>` lines.

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
_MESSAGE_RE = re.compile(r"^\s*MESSAGE\s+([\w.-]+)\s*:\s*(.+)$", re.MULTILINE)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _emit(event: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(event, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _coral(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "coral.cli", *args], capture_output=True, text=True
    )


def parse_reply(text: str) -> tuple[str | None, str, list[tuple[str, str]]]:
    """(program from the last fenced block, title, messages) from a model reply."""
    blocks = _FENCE_RE.findall(text)
    program = blocks[-1] if blocks else None
    outside = _FENCE_RE.sub("", text)
    title_match = _TITLE_RE.search(outside)
    title = title_match.group(1).strip()[:120] if title_match else "one-shot attempt"
    messages = [(to, body.strip()) for to, body in _MESSAGE_RE.findall(outside)]
    return program, title, messages


def build_prompt(
    *,
    task: str,
    context_files: dict[str, str],
    target: str,
    language: str,
    program: str,
    program_note: str,
    recent: list[str],
    messages: list[str],
    direction: str,
    cooperate: bool,
    agent_id: str,
) -> str:
    parts = [task.strip()]
    for name, text in context_files.items():
        parts.append(f"## {name}\n\n{text.strip()}")
    parts.append(f"## Current program ({target}): {program_note}\n\n```{language}\n{program}\n```")
    if recent:
        parts.append("## Recent attempts by the team (newest last)\n\n" + "\n".join(recent))
    if messages:
        parts.append("## Messages from other agents\n\n" + "\n".join(messages))
    reply_format = (
        f"Reply with a first line `TITLE: <what you changed, in a few words>`, then the "
        f"complete new {target} in a single ```{language} block. The whole file is replaced "
        "by your block, so include all of it."
    )
    if cooperate:
        reply_format += (
            f" You are agent `{agent_id}`. To tell teammates something useful (a finding, a "
            "warning, what you are trying), add lines `MESSAGE <agent-id or all>: <text>` "
            "outside the code block."
        )
    parts.append(f"## Your task now\n\n{direction}\n\n{reply_format}")
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


def _injected_messages(prompt: str) -> list[str]:
    """Message lines from a prompt the manager injected (cooperate mode)."""
    marker = "## Messages from other agents"
    if marker not in prompt:
        return []
    section = prompt.split(marker, 1)[1].split("\n## ", 1)[0]
    return [line for line in section.splitlines()[1:] if line.strip()]


def run(args: argparse.Namespace) -> None:
    from coral.cli._helpers import find_coral_dir_and_island, read_agent_id, read_direction
    from coral.hub.attempts import get_leaderboard, read_attempts
    from coral.hub.messages import format_messages, mark_read, read_unread

    worktree = Path.cwd()
    coral_dir, island_id = find_coral_dir_and_island()
    agent_id = read_agent_id()
    direction_name = read_direction(coral_dir)
    config = _read_yaml(coral_dir / "config.yaml")
    cooperate = (config.get("agents") or {}).get("mode") == "cooperate"
    task = config.get("task") or {}
    task_text = f"# {task.get('name', '')}\n\n{task.get('description', '')}"
    if task.get("tips"):
        task_text += f"\n\n## Tips\n\n{task['tips']}"
    context_files = {
        name: (worktree / name).read_text(encoding="utf-8", errors="replace")
        for name in args.context
        if (worktree / name).exists()
    }
    injected = _injected_messages(args.prompt or "")

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

        messages = list(injected)
        injected = []
        if cooperate:
            unread = read_unread(coral_dir, agent_id, island_id)
            if unread:
                messages += format_messages(unread).splitlines()
                mark_read(coral_dir, agent_id, [m.id for m in unread], island_id)

        prompt = build_prompt(
            task=task_text,
            context_files=context_files,
            target=args.target,
            language=args.language,
            program=program,
            program_note=note,
            recent=recent,
            messages=messages,
            direction=choose_direction(own_scores, agent_id, direction_name == "minimize"),
            cooperate=cooperate,
            agent_id=agent_id,
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

        program_new, title, outgoing = parse_reply(text)
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
        if cooperate:
            for to, body in outgoing:
                sent = _coral("msg", to, body)
                _tool_result(
                    f"coral msg {to}", (sent.stdout + sent.stderr).strip(), sent.returncode != 0
                )
        if not program_new or not program_new.strip():
            _tool_result("write", "No fenced code block in the reply; nothing to evaluate.", True)
            continue

        target_path.write_text(program_new, encoding="utf-8")
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
