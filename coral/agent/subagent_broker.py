"""Run agents' subagent requests with `claude -p` (agents.subagents).

Lives in the manager process, outside the agents' sandbox: that sandbox would
block the Claude API and the user's subscription login. The subagent is
contained by Claude Code instead:

- tools: Bash, Read, Write, Edit, Glob, Grep (+ WebSearch/WebFetch only when
  agents.internet is on), further limited by the subagent type's definition;
- Bash runs in Claude Code's sandbox: reads confined to the run directory
  (home and .coral/private denied), writes to the requesting agent's worktree,
  the shared state and temp dirs, no network, no git writes;
- file tools are confined to the worktree plus the shared state;
- the user's settings, CLAUDE.md and MCP servers are not loaded, and ANTHROPIC_*
  key variables are dropped so the logged-in Claude subscription is used.

Limits: subagent_max_running per agent at once (extra requests wait),
subagent_call_budget_usd per call (--max-budget-usd) and
subagent_run_budget_usd for the whole run (estimated cost, as reported by
Claude Code), plus subagent_timeout per call.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from coral.hub.subagents import (
    SubagentRequest,
    pending_requests,
    read_definitions,
    spent_usd,
    write_result,
)

logger = logging.getLogger(__name__)

_BASE_TOOLS = ("Bash", "Read", "Write", "Edit", "Glob", "Grep")
_WEB_TOOLS = ("WebSearch", "WebFetch")
_KEY_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")


def subagent_tools(declared: list[str], internet: bool) -> list[str]:
    allowed = set(_BASE_TOOLS) | (set(_WEB_TOOLS) if internet else set())
    return [t for t in declared if t in allowed]


def sandbox_settings(*, run_dir: Path, coral_dir: Path, worktree: Path, shared: Path) -> dict:
    """Claude Code settings that sandbox the subagent's Bash commands."""
    tmp = os.environ.get("TMPDIR", "/tmp")
    return {
        "sandbox": {
            "enabled": True,
            "autoAllowBashIfSandboxed": True,
            "allowUnsandboxedCommands": False,
            "filesystem": {
                "denyRead": [str(Path.home()), str((coral_dir / "private").resolve())],
                "allowRead": [str(run_dir.resolve())],
                "allowWrite": [
                    str(worktree.resolve()),
                    str(shared.resolve()),
                    tmp,
                    "/tmp",
                    "/private/tmp",
                ],
            },
        }
    }


def build_command(
    *,
    command: list[str],
    model: str,
    task: str,
    tools: list[str],
    system_prompt: str,
    shared_dir: Path,
    settings: dict,
    max_budget_usd: float,
) -> list[str]:
    return [
        *command,
        "-p",
        task,
        "--model",
        model,
        "--output-format",
        "json",
        "--tools",
        ",".join(tools),
        "--permission-mode",
        "acceptEdits",
        "--add-dir",
        str(shared_dir),
        "--append-system-prompt",
        system_prompt,
        "--settings",
        json.dumps(settings),
        "--setting-sources",
        "project",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--max-budget-usd",
        f"{max_budget_usd:.2f}",
    ]


class SubagentBroker:
    """Poll for subagent requests and run them with `claude -p`."""

    def __init__(
        self,
        config,
        coral_dir: Path,
        worktree_for: Callable[[str], Path],
        island_ids: list[str | None],
        poll: float = 1.0,
    ) -> None:
        self.config = config
        self.coral_dir = Path(coral_dir)
        self.worktree_for = worktree_for
        self.island_ids = island_ids or [None]
        self.poll = poll
        agents = config.agents
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, agents.count) * agents.subagent_max_running
        )
        self._running: dict[str, str] = {}  # request id -> agent id
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="coral-subagents", daemon=True)
        self._thread.start()
        a = self.config.agents
        logger.info(
            f"Subagent broker: {a.subagent_model}, {a.subagent_max_running} per agent at once, "
            f"${a.subagent_call_budget_usd}/call, ${a.subagent_run_budget_usd}/run"
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            for island_id in self.island_ids:
                try:
                    requests = pending_requests(self.coral_dir, island_id)
                except OSError:
                    continue
                for request in requests:
                    with self._lock:
                        if request.id in self._running:
                            continue
                        per_agent = sum(1 for a in self._running.values() if a == request.agent_id)
                        if per_agent >= self.config.agents.subagent_max_running:
                            continue  # waits until one of this agent's subagents finishes
                        self._running[request.id] = request.agent_id
                    self._pool.submit(self._handle, request, island_id)
            self._stop.wait(self.poll)

    def _handle(self, request: SubagentRequest, island_id: str | None) -> None:
        try:
            result = self._run(request, island_id)
        except Exception as e:  # never leave an agent waiting on a crashed handler
            logger.exception(f"Subagent request {request.id} failed")
            result = {"ok": False, "error": f"broker error: {e}"}
        write_result(self.coral_dir, request, result, island_id)
        with self._lock:
            self._running.pop(request.id, None)

    def _run(self, request: SubagentRequest, island_id: str | None) -> dict[str, Any]:
        agents = self.config.agents
        definitions = read_definitions(self.coral_dir, island_id)
        definition = definitions.get(request.kind)
        if definition is None:
            return {
                "ok": False,
                "error": f"unknown subagent type {request.kind!r}; "
                f"available: {', '.join(sorted(definitions))}",
            }
        call_budget = agents.subagent_call_budget_usd
        if agents.subagent_run_budget_usd:
            left = agents.subagent_run_budget_usd - spent_usd(self.coral_dir, self.island_ids)
            if left <= 0.01:
                return {
                    "ok": False,
                    "error": f"the run's subagent budget (${agents.subagent_run_budget_usd}) is used up",
                }
            call_budget = min(call_budget, left)

        worktree = self.worktree_for(request.agent_id)
        shared = (
            self.coral_dir / "public"
            if island_id is None
            else self.coral_dir / "islands" / str(island_id)
        )
        tools = subagent_tools(definition.tools, agents.internet)
        system_prompt = (
            f"{definition.prompt}\n\n"
            f"You are a worker for `{request.agent_id}`, a manager agent in a CORAL run "
            f"(objective: {self.config.task.name}). Your working directory is that agent's "
            f"code; the team's shared notes, skills and attempts are in {shared}. "
            "Your shell is sandboxed: no network, no access outside the run, and never use "
            "git (the manager scores your work with `coral eval`)."
            + ("" if agents.internet else " You have no internet access.")
            + " Finish with a concise report the manager can act on: what you did, which "
            "files changed, and what you measured."
        )
        cmd = build_command(
            command=list(agents.subagent_command),
            model=agents.subagent_model,
            task=request.task,
            tools=tools,
            system_prompt=system_prompt,
            shared_dir=shared.resolve(),
            settings=sandbox_settings(
                run_dir=self.coral_dir.parent,
                coral_dir=self.coral_dir,
                worktree=worktree,
                shared=shared,
            ),
            max_budget_usd=call_budget,
        )
        env = {k: v for k, v in os.environ.items() if k not in _KEY_VARS}
        logger.info(f"Subagent {request.kind} for {request.agent_id} ({request.id[:8]})")
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(worktree),
                env=env,
                capture_output=True,
                text=True,
                timeout=agents.subagent_timeout,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timed out after {agents.subagent_timeout}s"}
        try:
            out = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return {"ok": False, "error": (proc.stderr or proc.stdout)[-2000:]}
        return {
            "ok": not out.get("is_error", False),
            "result": out.get("result", ""),
            "tools": tools,
            "num_turns": out.get("num_turns"),
            "duration_ms": out.get("duration_ms"),
            "cost_usd": out.get("total_cost_usd"),
        }
