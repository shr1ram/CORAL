"""Inference throughput sampling from llama.cpp /slots."""

from __future__ import annotations

import json
from types import SimpleNamespace

from coral.web.api import get_inference
from coral.web.inference import (
    SAMPLES_FILE,
    InferenceSampler,
    read_samples,
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


async def test_inference_endpoint_without_sampler(tmp_path):
    (tmp_path / SAMPLES_FILE).write_text(json.dumps({"t": 1, "tok_s": 42.0}) + "\n")
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(coral_dir=tmp_path, inference=None))
    )

    body = json.loads((await get_inference(request)).body)

    assert body == {"enabled": False, "latest": None, "samples": [{"t": 1, "tok_s": 42.0}]}
