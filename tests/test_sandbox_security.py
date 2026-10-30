"""Security of the subagent broker and the agents' sandbox network/filesystem rules."""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

import pytest

from coral.agent.subagent_broker import SubagentBroker, subagent_env
from coral.config import AgentConfig, CoralConfig, SandboxConfig, TaskConfig
from coral.hub.subagents import (
    private_dir,
    read_definitions,
    read_definitions_dir,
    submit,
    wait_result,
)
from coral.sandbox import srt
from coral.sandbox.srt import (
    AllowAllProxy,
    SrtSandbox,
    build_srt_settings,
    is_loopback_host,
    loopback_literal,
)

FAKE_CLAUDE = """
import json, os, pathlib, subprocess, sys, time
here = pathlib.Path(__file__).parent
stdin = sys.stdin.read()
mode = (here / "mode").read_text()
with (here / "calls.jsonl").open("a") as f:
    f.write(json.dumps({"argv": sys.argv[1:], "stdin": stdin, "env": dict(os.environ)}) + "\\n")
if mode == "hang":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    (here / "child.pid").write_text(str(child.pid))
    time.sleep(60)
elif mode == "garbage":
    print("not json")
else:
    time.sleep(float(mode))
    print(json.dumps({"result": "ok", "is_error": False, "total_cost_usd": 0.5}))
"""


def _setup(tmp_path, mode="0", **agents):
    run = tmp_path / "run"
    coral_dir = run / ".coral"
    (coral_dir / "public" / "agents").mkdir(parents=True)
    for agent in ("alpha", "beta"):
        (run / "agents" / agent).mkdir(parents=True)
        (run / "agents" / agent / ".coral_agent_id").write_text(agent)
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "fake_claude.py").write_text(FAKE_CLAUDE)
    (fake / "mode").write_text(mode)
    config = CoralConfig(
        task=TaskConfig(name="Fishing", description="d"),
        agents=AgentConfig(
            count=2,
            subagents=True,
            subagent_command=[sys.executable, str(fake / "fake_claude.py")],
            **agents,
        ),
    )
    return run, coral_dir, fake, config


def _broker(config, coral_dir, run, worktree_for=None):
    return SubagentBroker(
        config, coral_dir, worktree_for or (lambda a: run / "agents" / a), [None], poll=0.05
    )


def _calls(fake: Path) -> list[dict]:
    log = fake / "calls.jsonl"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def _request(coral_dir, agent, name, data):
    path = coral_dir / "public" / "subagents" / "requests" / agent / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def _private_result(coral_dir, request_id, timeout=10.0):
    path = private_dir(coral_dir) / "results" / f"{request_id}.json"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return json.loads(path.read_text())
        time.sleep(0.05)
    return None


# --- forged requests ----------------------------------------------------------


def test_request_id_comes_from_the_file_name(tmp_path):
    run, coral_dir, fake, config = _setup(tmp_path)
    rid = "a" * 32
    _request(
        coral_dir,
        "alpha",
        rid,
        {"id": "../../../escape", "agent_id": "beta", "kind": "general", "task": "t"},
    )
    _request(coral_dir, "alpha", "not-an-id", {"kind": "general", "task": "t"})
    broker = _broker(config, coral_dir, run)
    broker.start()
    try:
        result = _private_result(coral_dir, rid)
    finally:
        broker.stop()
    assert result["id"] == rid and result["agent_id"] == "alpha" and result["ok"]
    assert not list(tmp_path.rglob("escape*"))
    assert len(_calls(fake)) == 1  # the badly named file was ignored


@pytest.mark.parametrize("case", ["ghost", "no-breadcrumb", "outside"])
def test_requests_for_unknown_agents_are_refused(tmp_path, case):
    run, coral_dir, fake, config = _setup(tmp_path)
    agent = "ghost" if case == "ghost" else "alpha"
    if case == "no-breadcrumb":
        (run / "agents" / "alpha" / ".coral_agent_id").write_text("beta")
    worktree_for = None
    if case == "outside":
        outside = tmp_path / "elsewhere" / "alpha"
        outside.mkdir(parents=True)
        (outside / ".coral_agent_id").write_text("alpha")
        worktree_for = lambda a: outside  # noqa: E731
    rid = "b" * 32
    _request(coral_dir, agent, rid, {"kind": "general", "task": "t"})
    broker = _broker(config, coral_dir, run, worktree_for)
    broker.start()
    try:
        result = _private_result(coral_dir, rid)
    finally:
        broker.stop()
    assert not result["ok"] and "unknown agent" in result["error"]
    assert _calls(fake) == []


@pytest.mark.parametrize(
    "data, error",
    [
        ({"kind": "general", "task": "x" * 9000}, "characters"),
        ({"kind": "general", "task": "   "}, "empty"),
        ({"kind": "../../etc", "task": "t"}, "unknown subagent type"),
    ],
)
def test_broker_checks_kind_and_task(tmp_path, data, error):
    run, coral_dir, fake, config = _setup(tmp_path)
    rid = "c" * 32
    _request(coral_dir, "alpha", rid, data)
    _request(coral_dir, "alpha", "d" * 32, {"kind": ["general"], "task": 5})  # wrong types
    broker = _broker(config, coral_dir, run)
    broker.start()
    try:
        result = _private_result(coral_dir, rid)
        time.sleep(0.2)
    finally:
        broker.stop()
    assert not result["ok"] and error in result["error"]
    assert _calls(fake) == []
    assert not (private_dir(coral_dir) / "results" / f"{'d' * 32}.json").exists()


def test_task_is_passed_on_stdin_and_env_is_allowlisted(tmp_path, monkeypatch):
    for key in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_USE_BEDROCK",
        "AWS_SECRET_ACCESS_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.setenv(key, "secret")
    monkeypatch.setenv("LC_ALL", "C")
    run, coral_dir, fake, config = _setup(tmp_path)
    request = submit(coral_dir, "alpha", "general", "--dangerously-skip-permissions")
    broker = _broker(config, coral_dir, run)
    broker.start()
    try:
        assert wait_result(coral_dir, request.id, timeout=10, poll=0.05)["ok"]
    finally:
        broker.stop()
    call = _calls(fake)[0]
    assert call["stdin"] == "--dangerously-skip-permissions"
    assert "--dangerously-skip-permissions" not in call["argv"]
    assert "secret" not in call["env"].values()
    assert call["env"].get("LC_ALL") == "C" and "PATH" in call["env"]
    assert subagent_env({"PATH": "/bin", "ANTHROPIC_AUTH_TOKEN": "x", "LC_CTYPE": "u"}) == {
        "PATH": "/bin",
        "LC_CTYPE": "u",
    }


# --- results, ledger and definitions -----------------------------------------


def test_forged_mirror_results_do_not_skip_requests_or_spend_budget(tmp_path):
    run, coral_dir, fake, config = _setup(tmp_path, subagent_run_budget_usd=1.0)
    request = submit(coral_dir, "alpha", "general", "real work")
    mirror = coral_dir / "public" / "subagents" / "results"
    mirror.mkdir(parents=True)
    (mirror / f"{request.id}.json").write_text(json.dumps({"ok": True, "cost_usd": 100.0}))
    (mirror / f"{'e' * 32}.json").write_text(json.dumps({"ok": True, "cost_usd": 100.0}))
    broker = _broker(config, coral_dir, run)
    broker.start()
    try:
        result = _private_result(coral_dir, request.id)
    finally:
        broker.stop()
    assert result["ok"] and result["cost_usd"] == 0.5
    mirrored = json.loads((mirror / f"{request.id}.json").read_text())
    assert mirrored["result"] == "ok"  # the broker's mirror replaced the forgery
    ledger = json.loads((private_dir(coral_dir) / "ledger.json").read_text())
    assert ledger["spent_usd"] == 0.5


def test_budget_is_reserved_before_launch(tmp_path):
    run, coral_dir, fake, config = _setup(
        tmp_path,
        mode="0.5",
        subagent_run_budget_usd=1.0,
        subagent_call_budget_usd=0.6,
    )
    first = submit(coral_dir, "alpha", "general", "one")
    second = submit(coral_dir, "beta", "general", "two")
    broker = _broker(config, coral_dir, run)
    broker.start()
    try:
        results = [wait_result(coral_dir, r.id, timeout=10, poll=0.05) for r in (first, second)]
        third = submit(coral_dir, "alpha", "general", "three")
        refused = wait_result(coral_dir, third.id, timeout=10, poll=0.05)
    finally:
        broker.stop()
    assert all(r["ok"] for r in results)
    budgets = sorted(
        float(c["argv"][c["argv"].index("--max-budget-usd") + 1]) for c in _calls(fake)
    )
    assert budgets == [0.4, 0.6]  # the concurrent call got only what was left
    assert not refused["ok"] and "budget" in refused["error"]


def test_failed_call_is_charged_its_reservation_and_ledger_survives_restart(tmp_path):
    run, coral_dir, fake, config = _setup(
        tmp_path, mode="garbage", subagent_run_budget_usd=1.0, subagent_call_budget_usd=0.6
    )
    request = submit(coral_dir, "alpha", "general", "x")
    broker = _broker(config, coral_dir, run)
    broker.start()
    try:
        assert not wait_result(coral_dir, request.id, timeout=10, poll=0.05)["ok"]
    finally:
        broker.stop()
    ledger = json.loads((private_dir(coral_dir) / "ledger.json").read_text())
    assert ledger["spent_usd"] == pytest.approx(0.6)

    restarted = _broker(config, coral_dir, run)
    restarted.start()
    try:
        r = submit(coral_dir, "alpha", "general", "y")
        assert not wait_result(coral_dir, r.id, timeout=10, poll=0.05)["ok"]
    finally:
        restarted.stop()
    budgets = [float(c["argv"][c["argv"].index("--max-budget-usd") + 1]) for c in _calls(fake)]
    assert budgets == [0.6, 0.4]
    assert len(_calls(fake)) == 2  # the first request was not re-run after the restart


def test_timeout_kills_the_whole_process_group(tmp_path):
    run, coral_dir, fake, config = _setup(tmp_path, mode="hang", subagent_timeout=1)
    request = submit(coral_dir, "alpha", "general", "x")
    broker = _broker(config, coral_dir, run)
    broker.start()
    try:
        result = wait_result(coral_dir, request.id, timeout=15, poll=0.05)
    finally:
        broker.stop()
    assert not result["ok"] and "timed out" in result["error"]
    child = int((fake / "child.pid").read_text())
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("the subagent's child process survived the timeout")


def test_stop_kills_running_subagents(tmp_path):
    run, coral_dir, fake, config = _setup(tmp_path, mode="hang")
    submit(coral_dir, "alpha", "general", "x")
    broker = _broker(config, coral_dir, run)
    broker.start()
    deadline = time.time() + 10
    while not (fake / "child.pid").exists() and time.time() < deadline:
        time.sleep(0.05)
    with broker._lock:
        procs = list(broker._procs.values())
    broker.stop()
    assert procs and procs[0].wait(timeout=5) is not None


def test_definitions_are_snapshotted_and_parsed_safely(tmp_path):
    run, coral_dir, fake, config = _setup(tmp_path)
    agents = coral_dir / "public" / "agents"
    (agents / "helper.md").write_text("---\nname: helper\ntools: [Read]\n---\nHelp.\n")
    (agents / "broken.md").write_text("---\nname: [unclosed\n---\nx")
    (agents / "half.md").write_text("---\nno end")
    (agents / "list.md").write_text("---\n- a\n- b\n---\nbody")
    assert set(read_definitions(coral_dir)) == {"general", "helper", "list"}

    broker = _broker(config, coral_dir, run)
    broker.start()
    try:
        # Edits after the broker started do not reach it.
        (agents / "helper.md").write_text("---\nname: helper\ntools: [Bash]\n---\nEvil.\n")
        (agents / "late.md").write_text("---\nname: late\n---\nLate.\n")
        r = submit(coral_dir, "alpha", "late", "x")
        assert (
            "unknown subagent type" in wait_result(coral_dir, r.id, timeout=10, poll=0.05)["error"]
        )
    finally:
        broker.stop()
    snapshot = read_definitions_dir(private_dir(coral_dir) / "agents" / "_")
    assert snapshot["helper"].tools == ["Read"] and snapshot["helper"].prompt == "Help."


# --- agent sandbox: filesystem ---------------------------------------------


def test_srt_denies_writes_to_other_agents_requests_and_results(tmp_path):
    worktree = tmp_path / "agents" / "alpha"
    coral_dir = tmp_path / ".coral"
    repo_dir = tmp_path / "repo"
    for p in (worktree, tmp_path / "agents" / "gamma", coral_dir / "public", repo_dir / ".git"):
        p.mkdir(parents=True)
    settings = build_srt_settings(
        SandboxConfig(enabled=True),
        worktree_path=worktree,
        coral_dir=coral_dir,
        repo_dir=repo_dir,
        shared_dir_name=".pi",
        proxy_port=1,
        sibling_worktrees=[worktree, tmp_path / "agents" / "beta"],
    )
    deny = settings["filesystem"]["denyWrite"]
    sub = (coral_dir / "public" / "subagents").resolve()
    assert str(sub / "results") in deny
    assert str(sub / "requests" / "beta") in deny  # from the roster, before it exists
    assert str(sub / "requests" / "gamma") in deny  # from disk
    assert str(sub / "requests" / "alpha") not in deny


# --- agent sandbox: network ---------------------------------------------------


def test_loopback_literal_never_resolves(monkeypatch):
    def no_dns(*args, **kwargs):
        raise AssertionError("DNS lookup")

    monkeypatch.setattr(socket, "getaddrinfo", no_dns)
    assert loopback_literal("localhost") == "127.0.0.1"
    assert loopback_literal("LOCALHOST") == "127.0.0.1"
    assert loopback_literal("127.0.0.5") == "127.0.0.5"
    assert loopback_literal("::1") == "::1"
    assert loopback_literal("[::1]") == "::1"
    assert loopback_literal("::ffff:127.0.0.1") == "127.0.0.1"
    for host in (
        "localhost.localdomain",
        "evil.localhost",
        "127.0.0.1.nip.io",
        "0x7f000001",
        "10.0.0.1",
        "::ffff:10.0.0.1",
        "",
    ):
        assert loopback_literal(host) is None, host
        assert not is_loopback_host(host)


def _connect(proxy_port: int, target: str) -> bytes:
    with socket.create_connection(("127.0.0.1", proxy_port), timeout=10) as conn:
        conn.sendall(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode())
        return conn.recv(4096)


@pytest.fixture
def two_listeners():
    socks = []
    for _ in range(2):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        s.listen(8)
        socks.append(s)
    yield [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()


def test_proxy_allowed_ports_and_literal_dial(two_listeners):
    allowed, other = two_listeners
    proxy = AllowAllProxy(allow_host=is_loopback_host, allowed_ports={allowed})
    proxy.start()
    try:
        assert _connect(proxy.port, f"127.0.0.1:{allowed}").startswith(b"HTTP/1.1 200")
        assert _connect(proxy.port, f"localhost:{allowed}").startswith(b"HTTP/1.1 200")
        assert _connect(proxy.port, f"127.0.0.1:{other}").startswith(b"HTTP/1.1 403")
        assert _connect(proxy.port, "127.0.0.1:99999").startswith(b"HTTP/1.1 403")
        assert _connect(proxy.port, "[::1]:notaport").startswith(b"HTTP/1.1 403")
        assert _connect(proxy.port, f"localtest.me:{allowed}").startswith(b"HTTP/1.1 403")
    finally:
        proxy.stop()


def _agents(**kw) -> AgentConfig:
    return AgentConfig(count=1, **kw)


def test_allowlist_with_external_domains_needs_internet(monkeypatch):
    monkeypatch.setattr(srt, "ensure_sandbox_supported", lambda agents: None)
    cfg = SandboxConfig(enabled=True, network="allowlist", allowed_domains=["github.com"])
    with pytest.raises(RuntimeError, match="external hosts"):
        SrtSandbox(cfg).validate(_agents(internet=False))
    SrtSandbox(cfg).validate(_agents(internet=True))
    local = SandboxConfig(enabled=True, network="allowlist", allowed_domains=["localhost"])
    SrtSandbox(local).validate(_agents(internet=False))


def test_internet_off_limits_proxy_to_the_model_port(monkeypatch):
    monkeypatch.setattr(srt, "ensure_sandbox_supported", lambda agents: None)
    sandbox = SrtSandbox(SandboxConfig(enabled=True))
    sandbox.validate(
        _agents(internet=False, runtime_options={"base_url": "http://127.0.0.1:11501/v1"})
    )
    assert sandbox.allowed_ports == {11501}
    online = SrtSandbox(SandboxConfig(enabled=True))
    online.validate(_agents(internet=True, runtime_options={"base_url": "http://127.0.0.1:1/v1"}))
    assert online.allowed_ports is None
    remote = SrtSandbox(SandboxConfig(enabled=True))
    remote.validate(_agents(internet=False, runtime_options={"base_url": "https://api.x.com/v1"}))
    assert remote.allowed_ports is None
