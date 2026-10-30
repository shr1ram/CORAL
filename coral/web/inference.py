"""Sample generation throughput from a llama.cpp server for the dashboard.

llama-server's ``/slots`` endpoint reports, for each busy slot, its task id,
how many prompt tokens it has read and how many tokens it has decoded so far.
Summing the growth of those counters between polls gives the server's total
generation tokens/s and prompt-reading tokens/s. A slot re-reading a long
prompt after a cache miss shows up as prompt tokens/s, not as idle time.
Samples are appended to ``<coral_dir>/inference.jsonl`` (outside ``public/``,
so agents never see it) so replays of a run can show throughput later. Each
sample stores the token counts themselves (``gen``, ``prompt``) as well as the
rates, so totals are sums rather than rate × gap estimates.

Limitation: ``/slots`` only shows requests that are still running. Tokens a
request decodes after the last poll and before it finishes are never seen, so
totals undercount slightly (by at most ``interval`` × rate per request). The
cumulative counters on ``/metrics`` would fix this, but llama-server only
serves them when started with ``--metrics``; the 1 s poll keeps the gap small.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

SAMPLES_FILE = "inference.jsonl"


def slot_progress(slots: list[dict[str, Any]]) -> dict[int, tuple[int, int, int]]:
    """Map slot id → (task id, prompt tokens read, tokens decoded) for busy slots."""
    progress = {}
    for slot in slots:
        if not slot.get("is_processing"):
            continue
        decoded = sum(t.get("n_decoded", 0) for t in slot.get("next_token") or [])
        read = slot.get("n_prompt_tokens_processed", 0)
        progress[slot["id"]] = (slot.get("id_task", -1), read, decoded)
    return progress


def slot_detail(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-slot progress on the current request, for the live view."""
    out = []
    for slot in slots:
        busy = bool(slot.get("is_processing"))
        prompt = slot.get("n_prompt_tokens", 0) if busy else 0
        cached = slot.get("n_prompt_tokens_cache", 0) if busy else 0
        out.append(
            {
                "id": slot.get("id"),
                "busy": busy,
                "prompt": prompt,
                "cached": cached,
                "read": slot.get("n_prompt_tokens_processed", 0) if busy else 0,
                "to_read": max(prompt - cached, 0),
                "decoded": sum(t.get("n_decoded", 0) for t in slot.get("next_token") or [])
                if busy
                else 0,
            }
        )
    return out


def match_slots(agents: dict[str, dict[str, Any]], slots: list[dict[str, Any]]) -> dict[str, int]:
    """Estimate which busy server slot is serving which agent.

    llama.cpp does not know which client a slot belongs to, so pair agents with
    an in-flight model call to busy slots: an agent still waiting for its first
    token goes with a slot that is reading its prompt, a streaming agent with a
    slot that is generating, and the prompt length should be at least the
    agent's last known context size. Greedy, cheapest pairing first.
    """
    pairs = []
    for agent, state in agents.items():
        if state.get("state") not in ("waiting", "streaming"):
            continue
        context = state.get("context_tokens", 0)
        for slot in slots:
            if not slot.get("busy"):
                continue
            if (state["state"] == "streaming") != (slot["decoded"] > 0):
                continue  # a waiting agent can't be on a generating slot, and vice versa
            shortfall = max(context - slot["prompt"], 0)  # compaction can shrink a prompt
            cost = abs(slot["prompt"] - context) + 4 * shortfall
            pairs.append((cost, agent, slot["id"]))
    assigned: dict[str, int] = {}
    used: set[int] = set()
    for _, agent, slot_id in sorted(pairs):
        if agent not in assigned and slot_id not in used:
            assigned[agent] = slot_id
            used.add(slot_id)
    return assigned


def tokens_since(
    before: dict[int, tuple[int, int, int]], after: dict[int, tuple[int, int, int]]
) -> tuple[int, int]:
    """(prompt tokens read, tokens generated) between two polls.

    A slot on a new task counts from zero.
    """
    read = generated = 0
    for slot, (task, now_read, now_decoded) in after.items():
        prev_task, prev_read, prev_decoded = before.get(slot, (None, 0, 0))
        if prev_task != task:
            prev_read = prev_decoded = 0
        read += max(now_read - prev_read, 0)
        generated += max(now_decoded - prev_decoded, 0)
    return read, generated


def manager_alive(coral_dir: Path) -> bool:
    """Cheap liveness check (no subprocess): is the run's manager PID running?"""
    from coral.cli._helpers import is_process_alive

    try:
        pid = int((Path(coral_dir) / "public" / "manager.pid").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    return is_process_alive(pid)


_SAMPLES: dict[str, tuple[int, int, list[dict[str, Any]]]] = {}  # path → (inode, offset, samples)
_SAMPLES_LOCK = threading.Lock()


def samples_for(coral_dir: Path) -> list[dict[str, Any]]:
    """All samples recorded for a run, read incrementally and kept in memory.

    Only bytes appended since the previous call are parsed. Returns a shared
    list; callers must not mutate it.
    """
    path = Path(coral_dir) / SAMPLES_FILE
    key = str(path)
    with _SAMPLES_LOCK:
        try:
            st = path.stat()
        except OSError:
            _SAMPLES.pop(key, None)
            return []
        inode, offset, samples = _SAMPLES.get(key, (st.st_ino, 0, []))
        if inode != st.st_ino or st.st_size < offset:
            offset, samples = 0, []
        if st.st_size > offset:
            with path.open("rb") as f:
                f.seek(offset)
                data = f.read()
            cut = data.rfind(b"\n") + 1  # leave a partly written last line for next time
            for line in data[:cut].splitlines():
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            offset += cut
        _SAMPLES[key] = (st.st_ino, offset, samples)
        return samples


def read_samples(coral_dir: Path) -> list[dict[str, Any]]:
    return list(samples_for(coral_dir))


def downsample(samples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """At most ``limit`` samples for charting, merging neighbours into buckets.

    Token counts are summed and rates recomputed over the bucket, so totals
    are preserved.
    """
    if limit <= 0 or len(samples) <= limit:
        return list(samples)
    size = -(-len(samples) // limit)
    out = []
    for i in range(0, len(samples), size):
        bucket = samples[i : i + size]
        last = dict(bucket[-1])
        if all("gen" in x and "dt" in x for x in bucket):
            dt = sum(x["dt"] for x in bucket) or 1.0
            last["gen"] = sum(x["gen"] for x in bucket)
            last["prompt"] = sum(x.get("prompt", 0) for x in bucket)
            last["dt"] = round(dt, 3)
            last["tok_s"] = round(last["gen"] / dt, 1)
            last["prompt_tok_s"] = round(last["prompt"] / dt, 1)
        else:
            for k in ("tok_s", "prompt_tok_s"):
                vals = [x[k] for x in bucket if k in x]
                if vals:
                    last[k] = round(sum(vals) / len(vals), 1)
        out.append(last)
    return out


class InferenceSampler:
    """Poll ``<url>/slots`` and record total tokens/s into the active run."""

    def __init__(self, url: str, coral_dir_getter, run_alive, interval: float = 1.0) -> None:
        self.url = url.rstrip("/")
        self.coral_dir = coral_dir_getter
        self.run_alive = run_alive  # only record into a run that is running
        self.interval = interval
        self.latest: dict[str, Any] | None = None
        self._running = False

    def _fetch(self) -> list[dict[str, Any]]:
        with urllib.request.urlopen(f"{self.url}/slots", timeout=2) as response:
            return json.load(response)

    async def run(self) -> None:
        self._running = True
        before: dict[int, tuple[int, int, int]] | None = None
        last_t = time.time()
        while self._running:
            await asyncio.sleep(self.interval)
            try:
                slots = await asyncio.to_thread(self._fetch)
            except Exception:
                self.latest = {"t": time.time(), "error": "inference server unreachable"}
                before = None
                continue
            now = time.time()
            after = slot_progress(slots)
            if before is not None:
                read, generated = tokens_since(before, after)
                dt = max(now - last_t, 1e-3)
                sample = {
                    "t": now,
                    "dt": round(dt, 3),
                    "gen": generated,
                    "prompt": read,
                    "tok_s": round(generated / dt, 1),
                    "prompt_tok_s": round(read / dt, 1),
                    "active": len(after),
                    "reading": sum(1 for _, _, decoded in after.values() if decoded == 0),
                    "slots": len(slots),
                }
                await asyncio.to_thread(self._append, sample)
                self.latest = {**sample, "slot_detail": slot_detail(slots)}
            before, last_t = after, now

    def _append(self, sample: dict[str, Any]) -> None:
        coral_dir = Path(self.coral_dir())
        if not self.run_alive(coral_dir):
            return
        path = coral_dir / SAMPLES_FILE
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(sample) + "\n")
        except OSError:
            pass

    def stop(self) -> None:
        self._running = False
