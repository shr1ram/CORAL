"""Agent-to-agent messaging (cooperate mode)."""

from __future__ import annotations

import argparse

import pytest
import yaml

from coral.cli import messages as cli_messages
from coral.config import AgentConfig, CoralConfig, TaskConfig
from coral.hub.messages import (
    MAX_BODY_CHARS,
    format_messages,
    list_messages,
    mark_read,
    read_unread,
    send,
)
from coral.template.coral_md import generate_coral_md


def _run_dir(tmp_path, mode="cooperate", agents=("alpha", "beta", "gamma")):
    coral_dir = tmp_path / ".coral"
    roles = coral_dir / "public" / "roles"
    roles.mkdir(parents=True)
    for agent in agents:
        (roles / f"{agent}.md").write_text("role")
    (coral_dir / "config.yaml").write_text(yaml.safe_dump({"agents": {"mode": mode}}))
    return coral_dir


def test_direct_message_reaches_only_recipient(tmp_path):
    coral_dir = _run_dir(tmp_path)
    send(coral_dir, "alpha", "beta", "taking annealing")

    assert [m.body for m in read_unread(coral_dir, "beta")] == ["taking annealing"]
    assert read_unread(coral_dir, "gamma") == []
    assert read_unread(coral_dir, "alpha") == []


def test_broadcast_reaches_everyone_but_sender(tmp_path):
    coral_dir = _run_dir(tmp_path)
    send(coral_dir, "alpha", "all", "grid init helps")

    assert len(read_unread(coral_dir, "beta")) == 1
    assert len(read_unread(coral_dir, "gamma")) == 1
    assert read_unread(coral_dir, "alpha") == []


def test_mark_read_is_per_recipient(tmp_path):
    coral_dir = _run_dir(tmp_path)
    message = send(coral_dir, "alpha", "all", "hello")

    mark_read(coral_dir, "beta", [message.id])

    assert read_unread(coral_dir, "beta") == []
    assert len(read_unread(coral_dir, "gamma")) == 1


def test_messages_are_listed_in_send_order(tmp_path):
    coral_dir = _run_dir(tmp_path)
    for body in ("one", "two", "three"):
        send(coral_dir, "alpha", "beta", body)

    assert [m.body for m in list_messages(coral_dir)] == ["one", "two", "three"]
    assert "From alpha" in format_messages(list_messages(coral_dir))


@pytest.mark.parametrize("body", ["", "   ", "x" * (MAX_BODY_CHARS + 1)])
def test_send_rejects_empty_or_oversized_body(tmp_path, body):
    with pytest.raises(ValueError):
        send(_run_dir(tmp_path), "alpha", "beta", body)


def test_config_mode_defaults_to_compete_and_is_validated():
    assert AgentConfig().mode == "compete"
    assert AgentConfig(mode="cooperate").mode == "cooperate"
    with pytest.raises(ValueError, match="agents.mode"):
        AgentConfig(mode="collude")


def test_config_mode_round_trips_through_yaml(tmp_path):
    path = tmp_path / "task.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "task": {"name": "t", "description": "d"},
                "grader": {"entrypoint": "grader:Grader"},
                "agents": {"mode": "cooperate"},
            }
        )
    )
    assert CoralConfig.from_yaml(path).agents.mode == "cooperate"


@pytest.mark.parametrize(
    ("mode", "single_agent", "expected"),
    [("cooperate", False, True), ("compete", False, False), ("cooperate", True, False)],
)
def test_coral_md_messaging_section(mode, single_agent, expected):
    config = CoralConfig(
        task=TaskConfig(name="t", description="d"), agents=AgentConfig(count=2, mode=mode)
    )
    md = generate_coral_md(config, "alpha", single_agent=single_agent)
    assert ("coral msg" in md) is expected


def _cli_args(**kwargs):
    defaults = {"task": None, "run": None, "sender": None, "all": False}
    return argparse.Namespace(**{**defaults, **kwargs})


@pytest.fixture
def cli_run(tmp_path, monkeypatch):
    def make(mode="cooperate", agent="alpha"):
        coral_dir = _run_dir(tmp_path, mode=mode)
        monkeypatch.setattr(
            cli_messages, "find_coral_dir_and_island", lambda task, run: (coral_dir, None)
        )
        monkeypatch.setattr(cli_messages, "read_agent_id", lambda: agent)
        return coral_dir

    return make


def test_cli_msg_then_inbox(cli_run, monkeypatch, capsys):
    coral_dir = cli_run(agent="alpha")
    cli_messages.cmd_msg(_cli_args(to="beta", body=["try", "a", "grid"]))
    assert "Sent to beta" in capsys.readouterr().out

    monkeypatch.setattr(cli_messages, "read_agent_id", lambda: "beta")
    cli_messages.cmd_inbox(_cli_args())
    assert "try a grid" in capsys.readouterr().out
    assert read_unread(coral_dir, "beta") == []

    cli_messages.cmd_inbox(_cli_args())
    assert "No unread messages" in capsys.readouterr().out


def test_cli_msg_rejects_unknown_agent_and_self(cli_run):
    cli_run(agent="alpha")
    with pytest.raises(SystemExit):
        cli_messages.cmd_msg(_cli_args(to="zeus", body=["hi"]))
    with pytest.raises(SystemExit):
        cli_messages.cmd_msg(_cli_args(to="alpha", body=["hi"]))


def test_cli_messaging_is_disabled_in_compete_mode(cli_run):
    cli_run(mode="compete")
    with pytest.raises(SystemExit):
        cli_messages.cmd_msg(_cli_args(to="beta", body=["hi"]))
    with pytest.raises(SystemExit):
        cli_messages.cmd_inbox(_cli_args())


def test_cli_msg_from_outside_a_worktree_is_from_operator(cli_run):
    coral_dir = cli_run(agent="unknown")
    cli_messages.cmd_msg(_cli_args(to="beta", body=["focus on packing density"]))
    assert read_unread(coral_dir, "beta")[0].sender == "operator"


def _manager_with_one_scored_attempt(tmp_path, monkeypatch, mode, heartbeat_every):
    """An AgentManager whose monitor loop sees one scored attempt from beta."""
    from types import SimpleNamespace

    from coral.agent.heartbeat import HeartbeatAction, HeartbeatRunner
    from coral.agent.manager import AgentManager
    from coral.workspace.project import ProjectPaths

    coral_dir = _run_dir(tmp_path, mode=mode)
    cfg = CoralConfig.from_dict(
        {"task": {"name": "t", "description": "d"}, "agents": {"count": 2, "mode": mode}}
    )
    mgr = AgentManager(cfg)
    mgr.paths = ProjectPaths(
        results_dir=tmp_path,
        task_dir=tmp_path,
        run_dir=tmp_path,
        coral_dir=coral_dir,
        agents_dir=tmp_path / "agents",
        repo_dir=tmp_path / "repo",
    )
    mgr._running = True
    handle = SimpleNamespace(agent_id="beta", alive=True)
    mgr.handles = [handle]
    seen = iter([set(), {"h1"}])
    reflect = HeartbeatAction(name="reflect", every=heartbeat_every, prompt="REFLECT")
    attempt = {"agent_id": "beta", "score": 1.0, "commit_hash": "abc123", "title": "t"}
    delivered = []

    def interrupt(idx, prompt, prompt_source):
        delivered.append(prompt)
        return handle

    monkeypatch.setattr(mgr, "_get_seen_attempts", lambda: next(seen, {"h1"}))
    monkeypatch.setattr(mgr, "_filter_scored", lambda attempts: attempts)
    monkeypatch.setattr(mgr, "_read_latest_attempt", lambda new, agent_id=None: attempt)
    monkeypatch.setattr(mgr, "_auto_stop_reason_from_current_state", lambda: None)
    monkeypatch.setattr(mgr, "_auto_stop_reason_from_attempt", lambda a: None)
    monkeypatch.setattr(mgr, "_get_eval_count", lambda: 1)
    monkeypatch.setattr(mgr, "_get_heartbeat_runner", lambda aid: HeartbeatRunner([reflect]))
    monkeypatch.setattr(mgr, "_interrupt_and_resume", interrupt)
    monkeypatch.setattr(mgr, "_write_agent_pids", lambda: None)
    monkeypatch.setattr(mgr, "_maybe_run_migration_cycle", lambda: None)
    mgr.config.agents.timeout = 0
    mgr._stop_event.set()  # one pass through the loop, then exit
    return mgr, coral_dir, delivered


def test_manager_delivers_messages_without_a_heartbeat(tmp_path, monkeypatch):
    mgr, coral_dir, delivered = _manager_with_one_scored_attempt(
        tmp_path, monkeypatch, "cooperate", heartbeat_every=100
    )
    send(coral_dir, "alpha", "beta", "your commit fails on n=26")

    mgr.monitor_loop(check_interval=0)

    assert len(delivered) == 1
    assert "your commit fails on n=26" in delivered[0]
    assert "REFLECT" not in delivered[0]
    assert read_unread(coral_dir, "beta") == []


def test_manager_appends_messages_to_heartbeat(tmp_path, monkeypatch):
    mgr, coral_dir, delivered = _manager_with_one_scored_attempt(
        tmp_path, monkeypatch, "cooperate", heartbeat_every=1
    )
    send(coral_dir, "alpha", "all", "grid init helps")

    mgr.monitor_loop(check_interval=0)

    assert "REFLECT" in delivered[0] and "grid init helps" in delivered[0]


def test_manager_ignores_messages_in_compete_mode(tmp_path, monkeypatch):
    mgr, coral_dir, delivered = _manager_with_one_scored_attempt(
        tmp_path, monkeypatch, "compete", heartbeat_every=100
    )
    send(coral_dir, "alpha", "beta", "hello")

    mgr.monitor_loop(check_interval=0)

    assert delivered == []
