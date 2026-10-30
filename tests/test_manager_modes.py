"""Run prerequisites, the monitor loop's per-tick attempt handling, and mode-aware instructions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from coral.agent import manager as manager_mod
from coral.agent.heartbeat import HeartbeatAction, HeartbeatRunner
from coral.agent.manager import AgentManager, check_run_prerequisites
from coral.config import CoralConfig
from coral.hub.heartbeat import (
    default_prompt,
    read_agent_heartbeat,
    run_default_prompt,
    write_agent_heartbeat,
)
from coral.template.coral_md import generate_coral_md
from coral.workspace.project import ProjectPaths


def _config(**agents) -> CoralConfig:
    return CoralConfig.from_dict({"task": {"name": "t", "description": "d"}, "agents": agents})


# --- run prerequisites ----------------------------------------------------------


def test_agentic_offline_without_sandbox_is_refused():
    with pytest.raises(RuntimeError, match="sandbox.enabled=true"):
        check_run_prerequisites(_config())


@pytest.mark.parametrize("kwargs", [{"docker": True}, {}])
def test_docker_or_isolate_user_cannot_enforce_offline(kwargs):
    config = _config(isolate_user=not kwargs)
    with pytest.raises(RuntimeError, match="Docker"):
        check_run_prerequisites(config, **kwargs)


@pytest.mark.parametrize(
    "agents",
    [{"loop": "oneshot"}, {"internet": True}, {"sandbox": {"enabled": True}}],
)
def test_runnable_configs_pass(agents):
    check_run_prerequisites(_config(**agents))


def test_subagents_need_the_claude_cli():
    config = _config(internet=True, subagents=True, subagent_command=["no-such-cli-xyz"])
    with pytest.raises(RuntimeError, match="no-such-cli-xyz"):
        check_run_prerequisites(config)


def test_refused_start_creates_no_project(monkeypatch):
    created = []
    monkeypatch.setattr(manager_mod, "create_project", lambda *a, **k: created.append(a))
    with pytest.raises(RuntimeError):
        AgentManager(_config()).start_all()
    assert created == []


# --- monitor loop ---------------------------------------------------------------


def _manager(tmp_path, monkeypatch, attempts, **agents):
    """An AgentManager whose monitor loop sees ``attempts`` scored in one tick."""
    coral_dir = tmp_path / ".coral"
    (coral_dir / "public").mkdir(parents=True)
    mgr = AgentManager(_config(count=2, internet=True, **agents))
    mgr.paths = ProjectPaths(
        results_dir=tmp_path,
        task_dir=tmp_path,
        run_dir=tmp_path,
        coral_dir=coral_dir,
        agents_dir=tmp_path / "agents",
        repo_dir=tmp_path / "repo",
    )
    mgr._running = True
    mgr.handles = [SimpleNamespace(agent_id=a, alive=True) for a in ("alpha", "beta")]
    seen = iter([set(), {"h1", "h2"}])
    calls = {"delivered": [], "migrations": 0}

    def interrupt(idx, prompt, prompt_source):
        calls["delivered"].append(mgr.handles[idx].agent_id)
        return mgr.handles[idx]

    def migrate():
        calls["migrations"] += 1

    reflect = HeartbeatAction(name="reflect", every=1, prompt="REFLECT")
    monkeypatch.setattr(mgr, "_get_seen_attempts", lambda: next(seen, {"h1", "h2"}))
    monkeypatch.setattr(mgr, "_filter_scored", lambda files: files)
    monkeypatch.setattr(mgr, "_read_scored_attempts", lambda files: attempts)
    monkeypatch.setattr(mgr, "_auto_stop_reason_from_current_state", lambda: None)
    monkeypatch.setattr(mgr, "_auto_stop_reason_from_attempt", lambda a: None)
    monkeypatch.setattr(mgr, "_get_eval_count", lambda: 1)
    monkeypatch.setattr(mgr, "_get_heartbeat_runner", lambda aid: HeartbeatRunner([reflect]))
    monkeypatch.setattr(mgr, "_interrupt_and_resume", interrupt)
    monkeypatch.setattr(mgr, "_write_agent_pids", lambda: None)
    monkeypatch.setattr(mgr, "_maybe_run_migration_cycle", migrate)
    mgr.config.agents.timeout = 0
    mgr._stop_event.set()  # one pass through the loop, then exit
    return mgr, calls


_TWO_ATTEMPTS = [
    {"agent_id": "alpha", "score": 1.0, "commit_hash": "a1", "title": "x"},
    {"agent_id": "beta", "score": 2.0, "commit_hash": "b1", "title": "y"},
]


def test_every_attempt_scored_in_a_tick_is_handled(tmp_path, monkeypatch):
    mgr, calls = _manager(tmp_path, monkeypatch, _TWO_ATTEMPTS)
    mgr.monitor_loop(check_interval=0)
    assert calls["delivered"] == ["alpha", "beta"]
    assert mgr._agent_eval_counts == {"alpha": 1, "beta": 1}


def test_oneshot_attempts_still_reach_the_loop_tail(tmp_path, monkeypatch):
    mgr, calls = _manager(tmp_path, monkeypatch, _TWO_ATTEMPTS, loop="oneshot")
    mgr.monitor_loop(check_interval=0)
    assert calls["delivered"] == []  # one-shot agents get no heartbeats
    assert mgr._agent_eval_counts == {"alpha": 1, "beta": 1}
    assert calls["migrations"] == 1


# --- heartbeat prompts ----------------------------------------------------------


@pytest.mark.parametrize("instructions", ["minimal", "full"])
def test_protected_readds_follow_the_runs_instructions(tmp_path, instructions):
    coral_dir = tmp_path / ".coral"
    (coral_dir / "public").mkdir(parents=True)
    (coral_dir / "config.yaml").write_text(
        yaml.safe_dump({"agents": {"instructions": instructions}})
    )
    write_agent_heartbeat(coral_dir, "alpha", [])
    [reflect] = read_agent_heartbeat(coral_dir, "alpha")
    assert reflect["prompt"] == default_prompt("reflect", instructions)
    assert run_default_prompt(coral_dir, "reflect") == default_prompt("reflect", instructions)


def test_run_default_prompt_without_config_is_minimal(tmp_path):
    assert run_default_prompt(tmp_path, "pivot") == default_prompt("pivot", "minimal")


# --- instructions ---------------------------------------------------------------


def _md(single_agent=False, **agents) -> str:
    config = CoralConfig.from_dict(
        {"task": {"name": "t", "description": "Build a subagent scheduler."}, "agents": agents}
    )
    return generate_coral_md(config, "alpha", single_agent=single_agent)


@pytest.mark.parametrize("instructions", ["minimal", "full"])
def test_subagent_strip_keeps_task_text(instructions):
    md = _md(instructions=instructions)
    assert "Build a subagent scheduler." in md
    assert "Spawn specialist subagents" not in md


def test_full_guide_drops_native_subagent_guidance():
    assert "/agents/" not in _md(instructions="full")


def test_single_agent_minimal_has_no_team():
    md = _md(single_agent=True)
    assert "coral/alpha" in md and "Your role" not in md and "teammates" not in md
    assert "Several agents work in parallel" in _md()


def test_research_line_depends_on_internet():
    online = _md(internet=True)
    offline = _md()
    assert "`deep-research` skill" in online and "no internet access" not in online
    assert "no internet access" in offline and "`deep-research` skill;" not in offline
    assert "no internet access" not in _md(research=False)


def test_full_single_agent_guide_offline_drops_web_search():
    assert "use web search" in _md(single_agent=True, instructions="full", internet=True)
    offline = _md(single_agent=True, instructions="full")
    assert "use web search" not in offline and "via web search" not in offline


def test_manager_section_offers_web_research_only_online():
    online = _md(internet=True, subagents=True)
    offline = _md(subagents=True)
    assert "(including web research)" in online
    assert "no web access" in offline and "(including web research)" not in offline
