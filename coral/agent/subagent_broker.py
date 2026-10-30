"""Run agents' subagent requests with `claude -p` (agents.subagents).

Lives in the manager process, outside the agents' sandbox: that sandbox would
block the Claude API and the user's subscription login. The subagent is
contained by Claude Code instead:

- tools: Bash, Read, Write, Edit, Glob, Grep (+ WebSearch/WebFetch only when
  agents.internet is on), further limited by the subagent type's definition;
- Bash runs in Claude Code's sandbox: reads confined to the requesting
  agent's worktree, the shared state, the run repo and the grader source
  (home, .coral/private and other agents' worktrees denied); writes to the
  worktree, shared notes and skills, and temp dirs; no network, no git writes;
- file tools are confined to the same writable dirs (--restricted, --add-dir);
- it runs from a fresh broker-owned scratch dir that agents cannot write, with
  user/local settings and MCP servers off, so no agent-written CLAUDE.md,
  .claude/settings*.json or .mcp.json is loaded;
- the environment is an allowlist (no API keys, cloud credentials or provider
  switches), so the logged-in Claude subscription is used;
- the task goes in on stdin, never as an argument.

Requests are untrusted (agents write them): the id comes from the file name,
the agent from its directory, and the type, task and worktree are checked here.
Results, the spend ledger and the subagent definitions the broker uses live
under .coral/private.

Limits: subagent_max_running per agent at once (extra requests wait),
subagent_call_budget_usd per call (--max-budget-usd) and
subagent_run_budget_usd for the whole run (reserved before each call and
settled from Claude Code's reported cost; a call that ends without one is
charged its whole reservation), plus subagent_timeout per call.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from coral.hub._island import island_root
from coral.hub.steering import _atomic_write_json
from coral.hub.subagents import (
    MAX_TASK_CHARS,
    SubagentRequest,
    pending_requests,
    private_dir,
    private_result_ids,
    read_definitions_dir,
    snapshot_definitions,
    valid_agent_id,
    write_result,
)

logger = logging.getLogger(__name__)

_BASE_TOOLS = ("Bash", "Read", "Write", "Edit", "Glob", "Grep")
_WEB_TOOLS = ("WebSearch", "WebFetch")
_ENV_KEEP = ("PATH", "HOME", "TMPDIR", "LANG", "TERM", "USER", "LOGNAME", "SHELL")
_MIN_CALL_USD = 0.01
_KILL_GRACE = 5.0


def subagent_tools(declared: list[str], internet: bool) -> list[str]:
    allowed = set(_BASE_TOOLS) | (set(_WEB_TOOLS) if internet else set())
    return [t for t in declared if t in allowed]


def subagent_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """The environment for `claude -p`: an allowlist, never keys or provider switches."""
    environ = os.environ if environ is None else environ
    return {k: v for k, v in environ.items() if k in _ENV_KEEP or k.startswith("LC_")}


def writable_dirs(worktree: Path, shared: Path) -> list[Path]:
    """Where a subagent may write: the agent's code and the team's notes and skills."""
    return [worktree, shared / "notes", shared / "skills"]


def sandbox_settings(
    *,
    coral_dir: Path,
    worktree: Path,
    shared: Path,
    repo_dir: Path,
    scratch: Path,
) -> dict:
    """Claude Code settings that sandbox the subagent's Bash commands."""
    tmp = os.environ.get("TMPDIR", "/tmp")
    reads = [worktree, shared, repo_dir, scratch]
    grader = shared / "grader"
    if grader.exists():
        reads.append(grader.resolve())
    return {
        "sandbox": {
            "enabled": True,
            "autoAllowBashIfSandboxed": True,
            "allowUnsandboxedCommands": False,
            "filesystem": {
                "denyRead": [str(Path.home()), str((coral_dir / "private").resolve())],
                "allowRead": [str(p.resolve()) for p in reads],
                "allowWrite": [
                    *(str(p.resolve()) for p in writable_dirs(worktree, shared)),
                    str(scratch.resolve()),
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
    tools: list[str],
    system_prompt: str,
    add_dirs: list[Path],
    settings: dict,
    max_budget_usd: float,
) -> list[str]:
    """argv for `claude -p`; the task itself goes on stdin."""
    cmd = [
        *command,
        "-p",
        "--model",
        model,
        "--output-format",
        "json",
        "--tools",
        ",".join(tools),
        "--permission-mode",
        "acceptEdits",
        "--restricted",
    ]
    for d in add_dirs:
        cmd += ["--add-dir", str(d)]
    return cmd + [
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


def _killpg(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


class SubagentBroker:
    """Poll for subagent requests and run them with `claude -p`."""

    def __init__(
        self,
        config,
        coral_dir: Path,
        worktree_for: Callable[[str], Path],
        island_ids: list[str | None],
        poll: float = 1.0,
        agents_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.coral_dir = Path(coral_dir)
        self.worktree_for = worktree_for
        self.agents_dir = Path(agents_dir or self.coral_dir.parent / "agents")
        self.island_ids = island_ids or [None]
        self.poll = poll
        agents = config.agents
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, agents.count) * agents.subagent_max_running
        )
        self._private = private_dir(self.coral_dir)
        # Outside every agent-writable tree, so Claude Code's walk up from its
        # cwd finds no agent-written CLAUDE.md or .claude/ settings.
        self._scratch_root = self.coral_dir.parent / ".subagent-scratch"
        self._ledger_path = self._private / "ledger.json"
        self._running: dict[str, str] = {}  # request id -> agent id
        self._procs: dict[str, subprocess.Popen] = {}
        self._done: set[str] = set()
        self._spent = 0.0
        self._reserved = 0.0
        self._definitions: dict[str | None, Path] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._private.mkdir(parents=True, exist_ok=True)
        self._done = private_result_ids(self.coral_dir)
        self._spent = float(self._load_ledger().get("spent_usd", 0.0))
        for island_id in self.island_ids:
            self._definitions[island_id] = snapshot_definitions(self.coral_dir, island_id)
        self._thread = threading.Thread(target=self._loop, name="coral-subagents", daemon=True)
        self._thread.start()
        a = self.config.agents
        logger.info(
            f"Subagent broker: {a.subagent_model}, {a.subagent_max_running} per agent at once, "
            f"${a.subagent_call_budget_usd}/call, ${a.subagent_run_budget_usd}/run "
            f"(${self._spent:.2f} spent so far)"
        )

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            procs = list(self._procs.values())
        for proc in procs:
            _killpg(proc, signal.SIGKILL)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._pool.shutdown(wait=False, cancel_futures=True)

    # --- spend ledger ----------------------------------------------------------

    def _load_ledger(self) -> dict[str, Any]:
        try:
            data = json.loads(self._ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = None
        return data if isinstance(data, dict) else {"spent_usd": 0.0, "calls": {}}

    def _reserve(self) -> float | None:
        """Hold this call's budget; None when the run budget is used up."""
        agents = self.config.agents
        with self._lock:
            amount = agents.subagent_call_budget_usd
            if agents.subagent_run_budget_usd:
                left = agents.subagent_run_budget_usd - self._spent - self._reserved
                amount = min(amount, left)
            if amount < _MIN_CALL_USD:
                return None
            self._reserved += amount
            return amount

    def _settle(self, request_id: str, reserved: float, cost: float) -> None:
        with self._lock:
            self._reserved -= reserved
            self._spent += cost
            ledger = self._load_ledger()
            ledger["spent_usd"] = self._spent
            ledger.setdefault("calls", {})[request_id] = cost
            _atomic_write_json(self._ledger_path, ledger)

    # --- polling -----------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            for island_id in self.island_ids:
                with self._lock:
                    skip = self._done | set(self._running)
                try:
                    requests = pending_requests(self.coral_dir, island_id, done=skip)
                except (OSError, ValueError):
                    continue
                for request in requests:
                    with self._lock:
                        if request.id in self._running or request.id in self._done:
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
        try:
            write_result(self.coral_dir, request, result, island_id)
        finally:
            with self._lock:
                self._done.add(request.id)
                self._running.pop(request.id, None)

    # --- one request -------------------------------------------------------------

    def _worktree(self, agent_id: str) -> Path | None:
        """The agent's worktree, if ``agent_id`` names a real agent of this run."""
        if not valid_agent_id(agent_id):
            return None
        try:
            worktree = Path(self.worktree_for(agent_id)).resolve()
            breadcrumb = (worktree / ".coral_agent_id").read_text(encoding="utf-8").strip()
        except (OSError, ValueError):
            return None
        if worktree.parent != self.agents_dir.resolve() or worktree.name != agent_id:
            return None
        return worktree if breadcrumb == agent_id else None

    @staticmethod
    def _refusal(request: SubagentRequest, definitions: dict) -> str | None:
        if request.kind not in definitions:
            return (
                f"unknown subagent type {request.kind!r}; "
                f"available: {', '.join(sorted(definitions))}"
            )
        task = request.task.strip()
        if not task:
            return "the subagent task is empty"
        if len(task) > MAX_TASK_CHARS:
            return f"the subagent task is {len(task)} characters; keep it under {MAX_TASK_CHARS}"
        return None

    def _run(self, request: SubagentRequest, island_id: str | None) -> dict[str, Any]:
        agents = self.config.agents
        snapshot = self._definitions.get(island_id)
        if snapshot is None:
            snapshot = self._definitions[island_id] = snapshot_definitions(
                self.coral_dir, island_id
            )
        definitions = read_definitions_dir(snapshot)
        refusal = self._refusal(request, definitions)
        if refusal:
            return {"ok": False, "error": refusal}
        worktree = self._worktree(request.agent_id)
        if worktree is None:
            logger.warning(f"Refused subagent request {request.id}: unknown agent")
            return {"ok": False, "error": f"unknown agent {request.agent_id!r}"}

        budget = self._reserve()
        if budget is None:
            return {
                "ok": False,
                "error": f"the run's subagent budget (${agents.subagent_run_budget_usd}) is used up",
            }
        cost = budget  # charged in full unless Claude Code reports what it spent
        scratch = self._scratch_root / request.id
        try:
            scratch.mkdir(parents=True)
            result, reported = self._call(
                request, island_id, definitions[request.kind], worktree, scratch, budget
            )
            if reported is not None:
                cost = reported
            return result
        finally:
            self._settle(request.id, budget, cost)
            shutil.rmtree(scratch, ignore_errors=True)

    def _call(
        self,
        request: SubagentRequest,
        island_id: str | None,
        definition,
        worktree: Path,
        scratch: Path,
        budget: float,
    ) -> tuple[dict[str, Any], float | None]:
        """Run `claude -p`; returns the result and its reported cost (None if unknown)."""
        agents = self.config.agents
        shared = island_root(self.coral_dir, island_id).resolve()
        writable = writable_dirs(worktree, shared)
        for d in writable[1:]:
            d.mkdir(parents=True, exist_ok=True)
        tools = subagent_tools(definition.tools, agents.internet)
        system_prompt = (
            f"{definition.prompt}\n\n"
            f"You are a worker for `{request.agent_id}`, a manager agent in a CORAL run "
            f"(objective: {self.config.task.name}). That agent's code is in {worktree}: "
            f"work there. The team's shared notes, skills and attempts are in {shared}. "
            "Your shell is sandboxed: no network, no access outside the run, and never use "
            "git (the manager scores your work with `coral eval`)."
            + ("" if agents.internet else " You have no internet access.")
            + " Finish with a concise report the manager can act on: what you did, which "
            "files changed, and what you measured."
        )
        cmd = build_command(
            command=list(agents.subagent_command),
            model=agents.subagent_model,
            tools=tools,
            system_prompt=system_prompt,
            add_dirs=[d.resolve() for d in writable],
            settings=sandbox_settings(
                coral_dir=self.coral_dir,
                worktree=worktree,
                shared=shared,
                repo_dir=self.coral_dir.parent / "repo",
                scratch=scratch,
            ),
            max_budget_usd=budget,
        )
        if self._stop.is_set():
            return {"ok": False, "error": "the run is stopping"}, 0.0
        logger.info(f"Subagent {request.kind} for {request.agent_id} ({request.id[:8]})")
        proc = subprocess.Popen(
            cmd,
            cwd=str(scratch),
            env=subagent_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,  # its own process group, so killpg reaches its children
        )
        with self._lock:
            self._procs[request.id] = proc
        try:
            stdout, stderr = proc.communicate(request.task.strip(), timeout=agents.subagent_timeout)
        except subprocess.TimeoutExpired:
            _killpg(proc, signal.SIGTERM)
            try:
                proc.communicate(timeout=_KILL_GRACE)
            except subprocess.TimeoutExpired:
                _killpg(proc, signal.SIGKILL)
                proc.communicate()
            return {"ok": False, "error": f"timed out after {agents.subagent_timeout}s"}, None
        finally:
            with self._lock:
                self._procs.pop(request.id, None)
        try:
            out = json.loads(stdout)
        except json.JSONDecodeError:
            out = None
        if not isinstance(out, dict):
            if self._stop.is_set():
                return {"ok": False, "error": "stopped with the run"}, None
            return {"ok": False, "error": (stderr or stdout)[-2000:]}, None
        reported = out.get("total_cost_usd")
        valid = isinstance(reported, int | float) and not isinstance(reported, bool)
        cost = float(reported) if valid and reported >= 0 else None
        return {
            "ok": not out.get("is_error", False),
            "result": out.get("result", ""),
            "tools": tools,
            "num_turns": out.get("num_turns"),
            "duration_ms": out.get("duration_ms"),
            "cost_usd": cost,
        }, cost
