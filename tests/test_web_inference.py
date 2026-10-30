"""Inference throughput sampling from llama.cpp /slots."""

from __future__ import annotations

import json
from types import SimpleNamespace

from coral.web.api import get_inference
from coral.web.inference import (
    SAMPLES_FILE,
    InferenceSampler,
    decoded_since,
    read_samples,
    slot_progress,
)


def _slot(slot_id, task, decoded, processing=True):
    return {
        "id": slot_id,
        "id_task": task,
        "is_processing": processing,
        "next_token": [{"n_decoded": decoded}],
    }


def test_slot_progress_skips_idle_slots():
    slots = [_slot(0, 7, 120), _slot(1, 8, 50, processing=False)]
    assert slot_progress(slots) == {0: (7, 120)}


def test_decoded_since_counts_growth_and_new_tasks():
    before = {0: (7, 100), 1: (8, 40)}
    after = {0: (7, 160), 1: (9, 25), 2: (10, 5)}  # slot 1 started a new task
    assert decoded_since(before, after) == 60 + 25 + 5


def test_decoded_since_ignores_finished_slots():
    assert decoded_since({0: (7, 100)}, {}) == 0


def test_samples_append_only_while_run_is_alive(tmp_path):
    alive = {"value": False}
    sampler = InferenceSampler("http://x", lambda: tmp_path, lambda _: alive["value"])

    sampler._append({"t": 1, "tok_s": 10.0})
    alive["value"] = True
    sampler._append({"t": 2, "tok_s": 20.0})

    assert [s["tok_s"] for s in read_samples(tmp_path)] == [20.0]


async def test_inference_endpoint_without_sampler(tmp_path):
    (tmp_path / SAMPLES_FILE).write_text(json.dumps({"t": 1, "tok_s": 42.0}) + "\n")
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(coral_dir=tmp_path, inference=None))
    )

    body = json.loads((await get_inference(request)).body)

    assert body == {"enabled": False, "latest": None, "samples": [{"t": 1, "tok_s": 42.0}]}
