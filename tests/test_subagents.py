"""Subagent mode: managers delegate to `claude -p` subagents through the broker."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from coral.agent.subagent_broker import SubagentBroker, subagent_tools
from coral.config import AgentConfig, CoralConfig, TaskConfig
from coral.hub.heartbeat import default_prompt
from coral.hub.subagents import get_result, read_definitions, running_count, submit, wait_result
from coral.template.coral_md import generate_coral_md

# Stands in for `claude`: records its argv, sleeps, prints Claude Code's JSON result.
FAKE_CLAUDE = """
import json, sys, time, os, pathlib
log = pathlib.Path(os.environ["FAKE_LOG"])
with log.open("a") as f:
    f.write(json.dumps({"argv": sys.argv[1:], "start": time.time()}) + "\\n")
time.sleep(float(os.environ.get("FAKE_SLEEP", "0")))
print(json.dumps({"result": "did it", "is_error": False, "total_cost_usd": 0.5,
                  "num_turns": 3, "duration_ms": 10}))
"""


def _config(**agents) -> CoralConfig:
    return CoralConfig(
        task=TaskConfig(name="Fishing", description="Catch mackerel."),
        agents=AgentConfig(count=2, subagents=True, **agents),
    )


def _run(tmp_path, monkeypatch, sleep=0.0, **agents):
    coral_dir = tmp_path / "run" / ".coral"
    (coral_dir / "public" / "agents").mkdir(parents=True)
    for agent in ("alpha", "beta"):
        (tmp_path / "run" / "agents" / agent).mkdir(parents=True)
    script = tmp_path / "fake_claude.py"
    script.write_text(FAKE_CLAUDE)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("FAKE_SLEEP", str(sleep))
    config = _config(subagent_command=[sys.executable, str(script)], **agents)
    broker = SubagentBroker(
        config, coral_dir, lambda a: tmp_path / "run" / "agents" / a, [None], poll=0.05
    )
    return coral_dir, broker, log


def _calls(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def test_subagent_settings_and_validation():
    a = AgentConfig()
    assert a.subagents is False and a.subagent_max_running == 2
    assert a.subagent_run_budget_usd == 20.0 and a.subagent_model == "sonnet"
    with pytest.raises(ValueError, match="loop=agentic"):
        AgentConfig(subagents=True, loop="oneshot")


def test_tools_drop_web_without_internet():
    declared = ["Bash", "Read", "Write", "WebSearch", "WebFetch", "Agent"]
    assert subagent_tools(declared, internet=False) == ["Bash", "Read", "Write"]
    assert subagent_tools(declared, internet=True) == [
        "Bash",
        "Read",
        "Write",
        "WebSearch",
        "WebFetch",
    ]


def test_broker_runs_claude_locked_down(tmp_path, monkeypatch):
    coral_dir, broker, log = _run(tmp_path, monkeypatch)
    request = submit(coral_dir, "alpha", "general", "Write an SA solver")
    broker.start()
    try:
        result = wait_result(coral_dir, request.id, timeout=10, poll=0.05)
    finally:
        broker.stop()

    assert result["ok"] and result["result"] == "did it" and result["cost_usd"] == 0.5
    argv = _calls(log)[0]["argv"]
    assert argv[argv.index("--model") + 1] == "sonnet"
    assert argv[argv.index("--tools") + 1] == "Bash,Read,Write,Edit,Glob,Grep"
    assert argv[argv.index("--setting-sources") + 1] == "project"
    assert "--strict-mcp-config" in argv and float(argv[argv.index("--max-budget-usd") + 1]) == 2.0
    sandbox = json.loads(argv[argv.index("--settings") + 1])["sandbox"]
    assert sandbox["enabled"] and not sandbox["allowUnsandboxedCommands"]
    assert str(Path.home()) in sandbox["filesystem"]["denyRead"]
    assert (
        str((tmp_path / "run" / "agents" / "alpha").resolve())
        in sandbox["filesystem"]["allowWrite"]
    )


def test_broker_limits_running_subagents_per_agent(tmp_path, monkeypatch):
    coral_dir, broker, log = _run(tmp_path, monkeypatch, sleep=0.6, subagent_max_running=2)
    ids = [submit(coral_dir, "alpha", "general", f"task {i}").id for i in range(3)]
    broker.start()
    try:
        time.sleep(0.4)
        assert len(_calls(log)) == 2  # third waits for a free slot
        assert running_count(coral_dir, "alpha") == 3
        for request_id in ids:
            assert wait_result(coral_dir, request_id, timeout=10, poll=0.05)["ok"]
    finally:
        broker.stop()
    starts = sorted(c["start"] for c in _calls(log))
    assert starts[2] - starts[0] >= 0.5


def test_broker_stops_at_run_budget(tmp_path, monkeypatch):
    coral_dir, broker, log = _run(tmp_path, monkeypatch, subagent_run_budget_usd=1.0)
    broker.start()
    try:
        results = []
        for i in range(3):  # sequential so the spend is counted before the next call
            r = submit(coral_dir, "beta", "general", f"task {i}")
            results.append(wait_result(coral_dir, r.id, timeout=10, poll=0.05))
    finally:
        broker.stop()
    assert [r["ok"] for r in results] == [True, True, False]
    assert "budget" in results[2]["error"] and len(_calls(log)) == 2


def test_unknown_type_is_rejected(tmp_path, monkeypatch):
    coral_dir, broker, _ = _run(tmp_path, monkeypatch)
    request = submit(coral_dir, "alpha", "wizard", "x")
    broker.start()
    try:
        result = wait_result(coral_dir, request.id, timeout=10, poll=0.05)
    finally:
        broker.stop()
    assert not result["ok"] and "unknown subagent type" in result["error"]


def test_definitions_include_seeded_types(tmp_path):
    coral_dir = tmp_path / ".coral"
    (coral_dir / "public" / "agents").mkdir(parents=True)
    (coral_dir / "public" / "agents" / "librarian.md").write_text(
        "---\nname: librarian\ntools:\n  Read: true\n  Write: true\n---\nTidy notes.\n"
    )
    defs = read_definitions(coral_dir)
    assert set(defs) == {"general", "librarian"}
    assert (
        defs["librarian"].tools == ["Read", "Write"] and defs["librarian"].prompt == "Tidy notes."
    )


@pytest.mark.parametrize("instructions", ["minimal", "full"])
def test_manager_instructions(instructions):
    md = generate_coral_md(_config(instructions=instructions), "alpha", shared_dir=".pi")
    assert "## Manager mode (subagents)" in md and "coral subagent run" in md
    assert "At most 2 of your subagents" in md and "about $20" in md
    # Claude Code's native subagent text is never included.
    assert "subagent_type" not in md and "Spawn specialist subagents" not in md
    off = generate_coral_md(
        CoralConfig(task=TaskConfig(name="F", description="d"), agents=AgentConfig(count=2)), "a"
    )
    assert "Manager mode" not in off


def test_lint_wiki_briefs_a_librarian_in_subagent_mode():
    assert "coral subagent run --type librarian" in default_prompt("lint_wiki", subagents=True)
    assert "organize-files" in default_prompt("lint_wiki", subagents=False)


def test_pi_manager_has_no_editing_tools(tmp_path, monkeypatch):
    from coral.agent.builtin import pi_agent

    captured = {}

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            self.pid = 1

    monkeypatch.setattr(pi_agent.subprocess, "Popen", FakePopen)
    (tmp_path / ".coral_agent_id").write_text("alpha")
    pi_agent.PiAgentRuntime().start(
        tmp_path,
        tmp_path / "AGENTS.md",
        model="local/flint",
        runtime_options={"manager_only": True},
        log_dir=tmp_path / "logs",
    )
    tools = captured["cmd"][captured["cmd"].index("--tools") + 1]
    assert tools == "read,bash,grep,find,ls"


def test_cli_refuses_beyond_running_limit(tmp_path, monkeypatch, capsys):
    from coral.cli import subagent as cli

    coral_dir = tmp_path / ".coral"
    (coral_dir / "public").mkdir(parents=True)
    (coral_dir / "config.yaml").write_text(
        "agents:\n  subagents: true\n  subagent_max_running: 1\n"
    )
    monkeypatch.setattr(cli, "find_coral_dir_and_island", lambda task, run: (coral_dir, None))
    monkeypatch.setattr(cli, "read_agent_id", lambda: "alpha")
    args = SimpleNamespace(
        action="run", type="general", no_wait=True, prompt=["do", "x"], task=None, run=None
    )

    cli.cmd_subagent(args)
    assert "Started subagent" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.cmd_subagent(args)
    assert "already have 1 subagents running" in capsys.readouterr().err
    assert get_result(coral_dir, "nonexistent") is None
