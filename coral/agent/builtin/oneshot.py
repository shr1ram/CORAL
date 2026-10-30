"""One-shot runtime: runs coral.agent.oneshot as the agent process (agents.loop = "oneshot")."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from coral.agent.exit_classifier import classify_by_uptime
from coral.agent.process import open_agent_stderr_for_log_dir
from coral.agent.runtime import AgentHandle, write_coral_log_entry
from coral.sandbox.protocol import AgentSandboxSpec
from coral.workspace.repo import _clean_env

logger = logging.getLogger(__name__)


class OneShotRuntime:
    """Spawn the one-shot worker loop for an agent.

    Settings come from ``agents.runtime_options``: ``base_url`` (an
    OpenAI-compatible endpoint), ``target`` (the program file the model
    rewrites), ``context`` (extra files to include, e.g. a statement),
    ``language`` (code fence tag), ``max_tokens`` and ``temperature``. A
    ``provider/model`` model name is reduced to ``model``.
    """

    @property
    def instruction_filename(self) -> str:
        return "CORAL.md"

    @property
    def shared_dir_name(self) -> str:
        return ".coral-shared"

    def extract_session_id(self, log_path: Path) -> str | None:
        return None  # each start is a fresh loop; nothing to resume

    def classify_exit(
        self,
        log_path: Path,
        exit_code: int | None,
        uptime_seconds: float | None,
        min_clean_runtime_seconds: int = 60,
    ) -> str:
        return classify_by_uptime(exit_code, uptime_seconds, min_clean_runtime_seconds)

    def start(
        self,
        worktree_path: Path,
        coral_md_path: Path,
        model: str = "",
        runtime_options: dict[str, Any] | None = None,
        max_turns: int = 0,
        log_dir: Path | None = None,
        verbose: bool = False,
        resume_session_id: str | None = None,
        prompt: str | None = None,
        prompt_source: str | None = None,
        task_name: str | None = None,
        task_description: str | None = None,
        gateway_url: str | None = None,
        gateway_api_key: str | None = None,
        run_as_user: dict[str, Any] | None = None,
        sandbox: AgentSandboxSpec | None = None,
    ) -> AgentHandle:
        options = runtime_options or {}
        agent_id_file = worktree_path / ".coral_agent_id"
        agent_id = (
            agent_id_file.read_text(encoding="utf-8").strip()
            if agent_id_file.exists()
            else "unknown"
        )
        base_url = options.get("base_url") or gateway_url or os.environ.get("OPENAI_BASE_URL")
        target = options.get("target")
        if not base_url or not target:
            raise ValueError(
                "agents.loop=oneshot needs agents.runtime_options.base_url and .target"
            )
        if sandbox is not None:
            # The worker runs no model-written code itself (the grader does),
            # and the sandbox would block its localhost model endpoint.
            logger.info(f"One-shot agent {agent_id}: agents.sandbox does not apply, ignoring")

        if log_dir is None:
            log_dir = worktree_path / ".coral-logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{agent_id}.{len(list(log_dir.glob(f'{agent_id}*.log')))}.log"

        cmd = [
            sys.executable,
            "-m",
            "coral.agent.oneshot",
            "--base-url",
            str(base_url),
            "--model",
            str(model).rsplit("/", 1)[-1],
            "--target",
            str(target),
            "--language",
            str(options.get("language", "")),
            "--max-tokens",
            str(options.get("max_tokens", 32768)),
            "--temperature",
            str(options.get("temperature", 1.0)),
            "--prompt",
            prompt or "",
        ]
        for name in options.get("context") or []:
            cmd += ["--context", str(name)]

        env = _clean_env()
        if gateway_api_key:
            env["OPENAI_API_KEY"] = gateway_api_key

        log_file = open(log_path, "w", buffering=1, encoding="utf-8", errors="replace")
        write_coral_log_entry(
            log_file,
            prompt=prompt or "Begin.",
            source=prompt_source or "start",
            agent_id=agent_id,
            task_name=task_name,
            task_description=task_description,
        )
        stderr_target: Any = subprocess.STDOUT
        err_path: Path | None = None
        err_file: Any = None
        opened = open_agent_stderr_for_log_dir(log_dir, agent_id)
        if opened is not None:
            err_path, err_file = opened
            stderr_target = err_file

        process = subprocess.Popen(
            cmd,
            cwd=str(worktree_path),
            stdout=log_file,
            stderr=stderr_target,
            start_new_session=True,
            env=env,
        )
        logger.info(f"One-shot agent {agent_id} started with PID {process.pid}")
        return AgentHandle(
            agent_id=agent_id,
            process=process,
            worktree_path=worktree_path,
            log_path=log_path,
            err_file=err_file,
            err_path=err_path,
            _log_file=log_file,
        )
