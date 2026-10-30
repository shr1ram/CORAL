"""Sample generation throughput from a llama.cpp server for the dashboard.

llama-server's ``/slots`` endpoint reports, for each slot that is generating,
its task id and how many tokens it has decoded so far. Summing the growth of
those counters between polls gives the server's total output tokens/s.
Samples are appended to ``<coral_dir>/inference.jsonl`` (outside ``public/``,
so agents never see it) so replays of a run can show throughput later.
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

SAMPLES_FILE = "inference.jsonl"


def slot_progress(slots: list[dict[str, Any]]) -> dict[int, tuple[int, int]]:
    """Map slot id → (task id, tokens decoded) for slots that are generating."""
    progress = {}
    for slot in slots:
        if not slot.get("is_processing"):
            continue
        decoded = sum(t.get("n_decoded", 0) for t in slot.get("next_token") or [])
        progress[slot["id"]] = (slot.get("id_task", -1), decoded)
    return progress


def decoded_since(before: dict[int, tuple[int, int]], after: dict[int, tuple[int, int]]) -> int:
    """Tokens generated between two polls. A slot on a new task counts from zero."""
    total = 0
    for slot, (task, decoded) in after.items():
        prev_task, prev_decoded = before.get(slot, (None, 0))
        total += decoded - prev_decoded if prev_task == task else decoded
    return max(total, 0)


def manager_alive(coral_dir: Path) -> bool:
    """Cheap liveness check (no subprocess): is the run's manager PID running?"""
    from coral.cli._helpers import is_process_alive

    try:
        pid = int((Path(coral_dir) / "public" / "manager.pid").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    return is_process_alive(pid)


def read_samples(coral_dir: Path) -> list[dict[str, Any]]:
    path = Path(coral_dir) / SAMPLES_FILE
    samples = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return samples
    for line in lines:
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return samples


class InferenceSampler:
    """Poll ``<url>/slots`` and record total tokens/s into the active run."""

    def __init__(self, url: str, coral_dir_getter, run_alive, interval: float = 2.0) -> None:
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
        before: dict[int, tuple[int, int]] | None = None
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
                sample = {
                    "t": now,
                    "tok_s": round(decoded_since(before, after) / (now - last_t), 1),
                    "active": len(after),
                    "slots": len(slots),
                }
                self.latest = sample
                self._append(sample)
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
