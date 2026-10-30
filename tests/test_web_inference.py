"""Inference throughput sampling from llama.cpp /slots."""

from __future__ import annotations

import json
from types import SimpleNamespace

from coral.web.api import get_inference
from coral.web.inference import (
    SAMPLES_FILE,
    InferenceSampler,
    downsample,
    match_slots,
    read_samples,
    samples_for,
    slot_progress,
    tokens_since,
)


def _slot(slot_id, task, decoded, read=0, processing=True):
    return {
        "id": slot_id,
        "id_task": task,
        "is_processing": processing,
        "n_prompt_tokens_processed": read,
        "next_token": [{"n_decoded": decoded}],
    }


def test_slot_progress_skips_idle_slots():
    slots = [_slot(0, 7, 120, read=900), _slot(1, 8, 50, processing=False)]
    assert slot_progress(slots) == {0: (7, 900, 120)}


def test_tokens_since_counts_growth_and_new_tasks():
    before = {0: (7, 900, 100), 1: (8, 300, 40)}
    # slot 0 keeps generating; slot 1 started a new task and is re-reading a prompt
    after = {0: (7, 900, 160), 1: (9, 20000, 0), 2: (10, 50, 5)}
    assert tokens_since(before, after) == (20000 + 50, 60 + 5)


def test_tokens_since_ignores_finished_slots():
    assert tokens_since({0: (7, 900, 100)}, {}) == (0, 0)


def test_samples_append_only_while_run_is_alive(tmp_path):
    alive = {"value": False}
    sampler = InferenceSampler("http://x", lambda: tmp_path, lambda _: alive["value"])

    sampler._append({"t": 1, "tok_s": 10.0})
    alive["value"] = True
    sampler._append({"t": 2, "tok_s": 20.0})

    assert [s["tok_s"] for s in read_samples(tmp_path)] == [20.0]


def _request(tmp_path, **params):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(coral_dir=tmp_path, inference=None)),
        query_params=params,
    )


async def test_inference_endpoint_history_is_opt_in(tmp_path):
    (tmp_path / SAMPLES_FILE).write_text(
        json.dumps({"t": 1, "tok_s": 42.0}) + "\n" + json.dumps({"t": 2, "tok_s": 7.0}) + "\n"
    )

    latest_only = json.loads((await get_inference(_request(tmp_path))).body)
    history = json.loads((await get_inference(_request(tmp_path, history="1"))).body)
    since = json.loads((await get_inference(_request(tmp_path, since="1"))).body)

    assert latest_only == {"enabled": False, "latest": None}
    assert [s["t"] for s in history["samples"]] == [1, 2]
    assert [s["t"] for s in since["samples"]] == [2]


def test_samples_are_read_incrementally(tmp_path):
    path = tmp_path / SAMPLES_FILE
    path.write_text(json.dumps({"t": 1}) + "\n" + '{"t": 2')  # last line still being written
    assert [s["t"] for s in samples_for(tmp_path)] == [1]
    with path.open("a") as f:
        f.write("}\n" + json.dumps({"t": 3}) + "\n")
    assert [s["t"] for s in samples_for(tmp_path)] == [1, 2, 3]


def test_downsample_preserves_token_totals():
    samples = [{"t": i, "dt": 1.0, "gen": 10, "prompt": 100, "tok_s": 10.0} for i in range(10)]
    out = downsample(samples, 3)
    assert len(out) <= 3
    assert sum(s["gen"] for s in out) == 100 and sum(s["prompt"] for s in out) == 1000
    assert all(s["tok_s"] == 10.0 for s in out)


def test_match_slots_never_pairs_waiting_with_generating():
    agents = {
        "a": {"state": "waiting", "context_tokens": 5000},
        "b": {"state": "streaming", "context_tokens": 5000},
    }
    slots = [
        {"id": 0, "busy": True, "prompt": 5000, "decoded": 30},  # generating
    ]
    assert match_slots(agents, slots) == {"b": 0}
    assert match_slots({"a": agents["a"]}, slots) == {}
