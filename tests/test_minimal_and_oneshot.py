"""Minimal instructions (the default) and the one-shot loop."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from coral.agent import oneshot
from coral.config import AgentConfig, CoralConfig, HeartbeatActionConfig, TaskConfig
from coral.hub.heartbeat import DEFAULT_PROMPTS, default_local_actions, default_prompt
from coral.template.coral_md import generate_coral_md


def _config(**agents) -> CoralConfig:
    return CoralConfig(
        task=TaskConfig(name="Fishing", description="Catch mackerel.", tips="Stop at 1.8 s."),
        agents=AgentConfig(count=2, **agents),
    )


def test_minimal_instructions_are_the_default_and_short():
    md = generate_coral_md(_config(), "alpha", shared_dir=".pi")
    full = generate_coral_md(_config(instructions="full"), "alpha", shared_dir=".pi")

    assert AgentConfig().instructions == "minimal"
    assert len(md) < len(full) / 3
    # Every function of the full guide is still covered, briefly.
    for needed in (
        "Fishing",
        "Catch mackerel.",
        "Stop at 1.8 s.",
        "`alpha`",
        "Never run git yourself",
        "No user is watching",
        "Operation not permitted",
        "structural attempt 1/3",
        "local basin",
        ".pi/roles/alpha.md",
        "Lane",
        "Posture",
        ".pi/notes/focus/focus-alpha-",
        "coral log --search",
        "coral eval --tune",
        "create-notes",
        "skill-creator",
        "coral heartbeat",
    ):
        assert needed in md, needed
    # The research step follows agents.research (on by default).
    assert "deep-research` skill;" in md
    assert "deep-research` skill;" not in generate_coral_md(_config(research=False), "alpha")


def test_minimal_instructions_keep_messaging_section_in_cooperate_mode():
    assert "coral msg" in generate_coral_md(_config(mode="cooperate"), "alpha")
    assert "coral msg" not in generate_coral_md(_config(mode="compete"), "alpha")


def test_heartbeat_prompts_follow_instruction_set():
    for name in ("reflect", "pivot", "consolidate"):
        assert len(default_prompt(name)) < len(DEFAULT_PROMPTS[name]) / 2.5, name
    assert default_prompt("reflect", "full") == DEFAULT_PROMPTS["reflect"]

    config = _config(heartbeat=[HeartbeatActionConfig(name="reflect", every=1)])
    assert default_local_actions(config)[0]["prompt"] == default_prompt("reflect")
    custom = _config(heartbeat=[HeartbeatActionConfig(name="reflect", every=1, prompt="hi")])
    assert default_local_actions(custom)[0]["prompt"] == "hi"


@pytest.mark.parametrize("field,value", [("instructions", "short"), ("loop", "toolless")])
def test_new_agent_settings_are_validated(field, value):
    with pytest.raises(ValueError, match=f"agents.{field}"):
        AgentConfig(**{field: value})


def test_parse_reply_takes_last_block_title_and_messages():
    reply = (
        "TITLE: greedy merge of clusters\n"
        "```cpp\nold draft\n```\n"
        "Actually:\n```cpp\nint main() {}\n```\n"
        "MESSAGE all: merging clusters beats single boxes\n"
        "MESSAGE beta: your timer is off by 0.2s\n"
    )
    program, title, messages = oneshot.parse_reply(reply)

    assert program == "int main() {}\n"
    assert title == "greedy merge of clusters"
    assert messages == [
        ("all", "merging clusters beats single boxes"),
        ("beta", "your timer is off by 0.2s"),
    ]


def test_parse_reply_without_code():
    program, title, messages = oneshot.parse_reply("I think we should try annealing.")
    assert program is None and title == "one-shot attempt" and messages == []


def test_build_prompt_includes_context_program_and_cooperation():
    prompt = oneshot.build_prompt(
        task="# Fishing",
        context_files={"statement.md": "Maximize a - b."},
        target="solution.cpp",
        language="cpp",
        program="int main() {}",
        program_note="best so far",
        recent=["- beta: score 10 (improved): boxes"],
        messages=["- From beta: try merging"],
        direction="Try a different algorithm.",
        cooperate=True,
        agent_id="alpha",
    )
    for needed in (
        "Maximize a - b.",
        "```cpp\nint main() {}\n```",
        "beta: score 10",
        "try merging",
        "Try a different algorithm.",
        "MESSAGE",
    ):
        assert needed in prompt
    solo = oneshot.build_prompt(
        task="t",
        context_files={},
        target="s.cpp",
        language="cpp",
        program="",
        program_note="",
        recent=[],
        messages=[],
        direction="d",
        cooperate=False,
        agent_id="alpha",
    )
    assert "MESSAGE" not in solo


def test_chat_stream_collects_reasoning_text_and_usage(capsys):
    chunks = [
        {"choices": [{"delta": {"reasoning_content": "think "}}]},
        {"choices": [{"delta": {"reasoning_content": "more"}}]},
        {"choices": [{"delta": {"content": "TITLE: x\n```cpp\nint main(){}\n```"}}]},
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 600},
            },
        },
    ]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        reasoning, text, usage = oneshot.chat_stream(
            f"http://127.0.0.1:{server.server_port}/v1", "flint", "hi", 100, 1.0
        )
    finally:
        server.shutdown()

    assert reasoning == "think more"
    assert oneshot.parse_reply(text)[0] == "int main(){}\n"
    assert usage == {"input": 400, "output": 50, "cacheRead": 600, "cacheWrite": 0}
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[0]["type"] == "message_start"


def test_subagents_are_off_by_default_and_cannot_be_enabled_yet():
    assert AgentConfig().subagents is False
    with pytest.raises(ValueError, match="not implemented"):
        AgentConfig(subagents=True)


@pytest.mark.parametrize("instructions", ["minimal", "full"])
def test_instructions_and_heartbeats_do_not_mention_subagents(instructions):
    md = generate_coral_md(_config(instructions=instructions), "alpha", shared_dir=".pi")
    assert "subagent" not in md.lower()
    assert "librarian" not in md
    for name in ("reflect", "pivot", "consolidate", "lint_wiki"):
        assert "librarian" not in default_prompt(name, instructions), name


def test_internet_is_off_by_default():
    assert AgentConfig().internet is False


def test_loopback_filter():
    from coral.sandbox.srt import is_loopback_host

    for host in ("127.0.0.1", "::1", "localhost", "127.0.0.2"):
        assert is_loopback_host(host), host
    for host in ("1.1.1.1", "2606:4700::1111", "example.invalid"):
        assert not is_loopback_host(host), host


def test_proxy_refuses_non_loopback_when_internet_is_off():
    import socket

    from coral.sandbox.srt import AllowAllProxy, is_loopback_host

    target = socket.socket()
    target.bind(("127.0.0.1", 0))
    target.listen(1)
    proxy = AllowAllProxy(allow_host=is_loopback_host)
    port = proxy.start()

    def connect(authority: str) -> bytes:
        with socket.create_connection(("127.0.0.1", port), timeout=5) as c:
            c.sendall(f"CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n\r\n".encode())
            return c.recv(64)

    try:
        assert connect(f"127.0.0.1:{target.getsockname()[1]}").startswith(b"HTTP/1.1 200")
        assert connect("1.1.1.1:443").startswith(b"HTTP/1.1 403")
    finally:
        proxy.stop()
        target.close()


def test_agentic_without_sandbox_and_internet_off_refuses_to_start(tmp_path):
    from coral.agent.manager import AgentManager

    cfg = CoralConfig.from_dict({"task": {"name": "t", "description": "d"}, "agents": {"count": 1}})
    with pytest.raises(RuntimeError, match="agents.internet"):
        AgentManager(cfg)._start_sandbox_if_enabled()

    oneshot = CoralConfig.from_dict(
        {"task": {"name": "t", "description": "d"}, "agents": {"count": 1, "loop": "oneshot"}}
    )
    AgentManager(oneshot)._start_sandbox_if_enabled()  # one-shot agents have no tools
