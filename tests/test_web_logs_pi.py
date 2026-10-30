"""Pi agent logs: full turn parsing, live state, and slot matching."""

from __future__ import annotations

import json
from pathlib import Path

from coral.web.inference import match_slots, slot_detail
from coral.web.logs import live_state, parse_log_file


def _write(path: Path, events: list[dict]) -> Path:
    # Pi writes compact JSON; the parser's fast paths match on its exact prefixes.
    path.write_text("\n".join(json.dumps(e, separators=(",", ":")) for e in events) + "\n")
    return path


def _assistant(content, usage=None, ts=1793380000000):
    return {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": content,
            "usage": usage or {"input": 900, "output": 100, "cacheRead": 9000, "cacheWrite": 0},
            "timestamp": ts,
        },
    }


LONG = "\n".join(f"line {i}" for i in range(500))


def test_pi_turns_are_parsed_in_full(tmp_path):
    log = _write(
        tmp_path / "alpha.0.log",
        [
            {"type": "coral", "subtype": "prompt", "source": "start", "prompt": "Begin."},
            {"type": "session", "id": "s1"},
            {"type": "message_update", "assistantMessageEvent": {"partial": {}}},
            _assistant(
                [
                    {"type": "thinking", "thinking": LONG},
                    {
                        "type": "toolCall",
                        "id": "t1",
                        "name": "bash",
                        "arguments": {"command": "ls -la"},
                    },
                ]
            ),
            {
                "type": "message_end",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "t1",
                    "content": [{"type": "text", "text": LONG}],
                    "isError": True,
                },
            },
        ],
    )

    turns, _, _ = parse_log_file(log)

    entries = [e for t in turns for e in t.entries]
    assert [e.type for e in entries] == ["coral_prompt", "thinking", "tool_call", "tool_result"]
    assert entries[1].content == LONG  # no truncation
    assert entries[2].details["input"] == "ls -la"
    assert entries[3].content == LONG and entries[3].details["is_error"] is True
    assert turns[-1].usage == {
        "input_tokens": 900,
        "output_tokens": 100,
        "cache_creation": 0,
        "cache_read": 9000,
    }


def test_live_state_streaming_and_context(tmp_path):
    log = _write(
        tmp_path / "alpha.0.log",
        [
            _assistant([{"type": "text", "text": "done"}]),
            {"type": "message_start", "message": {"role": "assistant", "timestamp": 1}},
            {
                "type": "message_update",
                "assistantMessageEvent": {
                    "partial": {"content": [{"type": "thinking", "thinking": "so far"}]}
                },
            },
        ],
    )

    state = live_state(log)

    assert state["state"] == "streaming"
    assert state["entries"][0]["content"] == "so far"
    assert state["context_tokens"] == 900 + 9000 + 100


def test_live_state_waiting_and_tool(tmp_path):
    waiting = _write(
        tmp_path / "a.0.log",
        [{"type": "message_start", "message": {"role": "assistant", "timestamp": 1}}],
    )
    tool = _write(tmp_path / "b.0.log", [{"type": "tool_execution_start", "toolName": "bash"}])

    assert live_state(waiting)["state"] == "waiting"
    assert live_state(tool) == {"state": "tool", "tool": "bash", "since": "", "context_tokens": 0}


def test_live_state_finds_context_beyond_the_tail(tmp_path):
    big_update = {
        "type": "message_update",
        "assistantMessageEvent": {"partial": {"content": [{"type": "text", "text": "x" * 5000}]}},
    }
    log = _write(tmp_path / "a.0.log", [_assistant([]), big_update])

    assert live_state(log, tail_bytes=1000)["context_tokens"] == 10000


def _slot(i, prompt, cached, read, decoded, busy=True):
    return {
        "id": i,
        "is_processing": busy,
        "n_prompt_tokens": prompt,
        "n_prompt_tokens_cache": cached,
        "n_prompt_tokens_processed": read,
        "next_token": [{"n_decoded": decoded}],
    }


def test_slot_detail_and_matching():
    slots = slot_detail(
        [
            _slot(0, 40000, 39000, 1000, 250),  # generating, long context
            _slot(1, 12000, 0, 3000, 0),  # reading a 12k prompt
            _slot(2, 0, 0, 0, 0, busy=False),
        ]
    )
    assert slots[1]["to_read"] == 12000 and slots[2]["busy"] is False

    agents = {
        "alpha": {"state": "streaming", "context_tokens": 38000},
        "beta": {"state": "waiting", "context_tokens": 11000},
        "gamma": {"state": "tool", "context_tokens": 5000},
    }

    assert match_slots(agents, slots) == {"alpha": 0, "beta": 1}
