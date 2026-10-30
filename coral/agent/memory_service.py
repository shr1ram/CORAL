"""Team memory for one-shot runs: writer, librarian and state of play (agents.memory).

Runs in the manager process. Every call is one short, non-agent request to the
same OpenAI-compatible model the one-shot agents use, made one at a time so it
never holds more than one server slot:

- writer: each newly scored attempt (code diff, score change, grader feedback,
  the agent's status) becomes a memory item with reusable tags;
- librarian: when an agent posts a new STATUS or RECALL, it gets a fresh brief:
  the few memory facts that matter most for its next attempt, passed on close to
  their original wording. Candidates are all items, or the BM25 top hits plus the
  best and latest items once the store outgrows one prompt;
- state of play: every ``memory_state_every`` new items, a short team summary
  the librarian also draws on.

Agents never wait for this: the worker shows whatever brief is ready. A call
that fails or returns nothing is retried with backoff; after MAX_FAILURES the
writer stores a plain item built from the attempt, and a brief or state of play
waits for its next trigger. Attempts with no code change or no score get such a
plain item straight away, without a model call.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from coral.hub.attempts import read_attempts
from coral.hub.memory import (
    bm25_rank,
    item_text,
    known_tags,
    read_brief,
    read_items,
    read_state,
    write_brief,
    write_item,
    write_state,
)
from coral.hub.status import read_board

logger = logging.getLogger(__name__)

SHOW_ALL_ITEMS = 15  # up to this many items the librarian sees every one
BM25_TOP = 12
MAX_TOKENS = 6000  # the model reasons first; 1500 left no room for the answer
MAX_FAILURES = 3
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

WRITER_PROMPT = """You maintain a coding team's shared memory. One agent just submitted an attempt;
turn it into one memory item.

Task: {task}
Agent: {agent}. Its status: "{status}". Attempt title: "{title}".
Score: {score} ({outcome}). This agent's previous best: {agent_best}. Team best before: {team_best}.
Grader feedback:
{feedback}

Code change (diff against the program the attempt started from):
```diff
{diff}
```

Existing tags (reuse them where they fit; add a new tag only for a genuinely new idea): {tags}

Reply with JSON only:
{{"approach": "<the technique, a few words>", "change": "<what changed, one sentence>",
"result": "<what happened to the score and why, if visible>",
"lesson": "<one sentence a teammate could act on>", "tags": ["<2-5 short lowercase tags>"]}}"""

STATE_PROMPT = """You keep a coding team's state of play. Task: {task}

Memory (one line per attempt, oldest first):
{items}

What each agent says it is working on:
{board}

Write the state of play in at most 150 words: what works best so far (with scores), what has
failed and why, promising directions nobody has tried, and what each agent is working on."""

LIBRARIAN_PROMPT = """You are the team librarian. Agent {agent} is about to make its next attempt at: {task}

Its status: "{status}"
Its assigned direction: "{direction}"
{recall}Its last result: {last}

Team state of play:
{state}

Memory items:
{items}

Pass on the 3-8 facts from memory that matter most for this agent's next attempt: lessons about
what it is working on, failures to avoid, ideas that worked for others, untried directions{answer}.
Keep each fact close to its original wording and cite its source as [id, agent, score]. Do not add
advice of your own. Reply with the facts as a bullet list only."""


def chat_once(base_url: str, model: str, prompt: str, max_tokens: int = MAX_TOKENS) -> str:
    """One streamed completion; raises if it is empty (e.g. cut off while reasoning)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": True,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
    )
    parts, finish = [], None
    # Streaming keeps the connection alive while the model reasons, so a long
    # reply is not mistaken for a hung server.
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            for choice in json.loads(data).get("choices") or []:
                parts.append((choice.get("delta") or {}).get("content") or "")
                finish = choice.get("finish_reason") or finish
    text = "".join(parts).strip()
    if finish == "length":
        logger.warning("Memory call hit max_tokens=%d (finish_reason=length)", max_tokens)
    if not text:
        raise RuntimeError(f"empty reply (finish_reason={finish})")
    return text


def _parse_json(text: str) -> dict[str, Any] | None:
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _best(scores: list[float], minimize: bool) -> float | None:
    return (min(scores) if minimize else max(scores)) if scores else None


def _parse_item(text: str) -> dict[str, Any]:
    """Writer reply -> item fields; raises if it is not usable JSON."""
    fields = _parse_json(text)
    if not fields or not (fields.get("approach") or fields.get("lesson")):
        raise ValueError(f"writer reply is not item JSON: {text[:200]!r}")
    return fields


class MemoryService:
    """Background thread that keeps the team's memory and briefs up to date."""

    def __init__(
        self,
        config,
        coral_dir: Path,
        repo_dir: Path,
        island_ids: list[str | None],
        llm: Callable[[str], str],
        poll: float = 2.0,
    ) -> None:
        self.config = config
        self.coral_dir = Path(coral_dir)
        self.repo_dir = Path(repo_dir)
        self.island_ids = island_ids or [None]
        self.llm = llm
        self.poll = poll
        self.minimize = config.grader.direction == "minimize"
        self.target = (config.agents.runtime_options or {}).get("target", "")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # job key -> (failures, monotonic time before which it is not retried)
        self._failures: dict[tuple, tuple[int, float]] = {}

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="coral-memory", daemon=True)
        self._thread.start()
        logger.info("Memory service running (writer, librarian, state of play)")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            worked = False
            for island_id in self.island_ids:
                try:
                    worked = self.step(island_id) or worked
                except Exception:
                    logger.exception("Memory service step failed")
            if not worked:
                self._stop.wait(self.poll)

    def step(self, island_id: str | None = None) -> bool:
        """Do the most urgent pending job; True if one was done (or attempted)."""
        attempts = sorted(read_attempts(self.coral_dir, island_id), key=lambda a: a.timestamp)
        items = read_items(self.coral_dir, island_id)
        board = read_board(self.coral_dir, island_id)
        return (
            self._write_next_item(island_id, attempts, items, board)
            or self._write_next_brief(island_id, attempts, items, board)
            or self._write_state(island_id, items, board)
        )

    # --- failures ----------------------------------------------------------

    def _ready(self, key: tuple) -> bool:
        """False while a failed job waits out its backoff, or after it has given up."""
        count, retry_at = self._failures.get(key, (0, 0.0))
        return count < MAX_FAILURES and time.monotonic() >= retry_at

    def _call(self, key: tuple, prompt: str, parse: Callable[[str], Any] = str) -> Any | None:
        """Run one model call for ``key``; None on failure (recorded for backoff)."""
        try:
            result = parse(self.llm(prompt))
        except Exception as e:
            count = self._failures.get(key, (0, 0.0))[0] + 1
            self._failures[key] = (count, time.monotonic() + min(60 * count, 600))
            logger.warning("Memory %s failed (%d/%d): %s", key[0], count, MAX_FAILURES, e)
            return None
        self._failures.pop(key, None)
        return result

    # --- writer ------------------------------------------------------------

    def _write_next_item(self, island_id, attempts, items, board) -> bool:
        done = {i["id"] for i in items}
        todo = [a for a in attempts if a.status != "pending" and a.commit_hash not in done]
        for attempt in todo:
            key = ("writer", island_id, attempt.commit_hash)
            count = self._failures.get(key, (0, 0.0))[0]
            if count < MAX_FAILURES and not self._ready(key):
                continue  # backing off; try the next attempt meanwhile
            diff = self._diff(attempt.parent_hash, attempt.commit_hash)
            fields = None
            if count < MAX_FAILURES and attempt.score is not None and diff != "(no change)":
                fields = self._call(
                    key, self._writer_prompt(attempt, attempts, items, board, diff), _parse_item
                )
                if fields is None and self._failures.get(key, (0, 0.0))[0] < MAX_FAILURES:
                    return True  # retry later
            self._failures.pop(key, None)
            self._store_item(attempt, fields, diff, island_id)
            return True
        return False

    def _writer_prompt(self, attempt, attempts, items, board, diff: str) -> str:
        earlier = [a for a in attempts if a.timestamp < attempt.timestamp and a.score is not None]
        agent_best = _best(
            [a.score for a in earlier if a.agent_id == attempt.agent_id], self.minimize
        )
        return WRITER_PROMPT.format(
            task=self.config.task.name,
            agent=attempt.agent_id,
            status=(board.get(attempt.agent_id) or {}).get("status", ""),
            title=attempt.title,
            score=attempt.score,
            outcome=attempt.status,
            agent_best=agent_best,
            team_best=_best([a.score for a in earlier], self.minimize),
            feedback=(attempt.feedback or "")[:1500],
            diff=diff[:6000],
            tags=", ".join(known_tags(items)[:40]) or "(none yet)",
        )

    def _store_item(self, attempt, fields: dict | None, diff: str, island_id) -> None:
        """Write the item; without model fields, a plain one built from the attempt."""
        if fields is None:
            if diff == "(no change)":
                result = "no code change"
            elif attempt.score is None:
                result = "no score: " + (attempt.feedback or "").strip().split("\n")[0][:300]
            else:
                result = f"score {attempt.score}"
            fields = {"approach": attempt.title, "result": result, "tags": []}
        write_item(
            self.coral_dir,
            {
                "id": attempt.commit_hash,
                "agent": attempt.agent_id,
                "title": attempt.title,
                "score": attempt.score,
                "outcome": attempt.status,
                "created_at": attempt.timestamp,
                "approach": str(fields.get("approach") or attempt.title)[:200],
                "change": str(fields.get("change", ""))[:400],
                "result": str(fields.get("result", ""))[:400],
                "lesson": str(fields.get("lesson", ""))[:400],
                "tags": [str(t).lower()[:40] for t in (fields.get("tags") or [])][:5],
            },
            island_id,
        )

    def _diff(self, parent: str | None, commit: str) -> str:
        if not parent:
            return "(no parent)"
        cmd = ["git", "-C", str(self.repo_dir), "diff", parent, commit]
        if self.target:
            cmd += ["--", self.target]
        out = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        return out.stdout or "(no change)"

    # --- state of play -----------------------------------------------------

    def _write_state(self, island_id, items, board) -> bool:
        state = read_state(self.coral_dir, island_id) or {}
        every = self.config.agents.memory_state_every
        if not items or len(items) - state.get("n_items", 0) < every:
            return False
        key = ("state", island_id, len(items))
        if not self._ready(key):
            return False
        text = self._call(
            key,
            STATE_PROMPT.format(
                task=self.config.task.name,
                items="\n".join(item_text(i) for i in items[-60:]),
                board="\n".join(f"- {a}: {s.get('status', '')}" for a, s in board.items()) or "-",
            ),
        )
        if text is not None:
            write_state(
                self.coral_dir,
                {"text": text, "n_items": len(items), "created_at": datetime.now(UTC).isoformat()},
                island_id,
            )
        return True

    # --- librarian ---------------------------------------------------------

    def _write_next_brief(self, island_id, attempts, items, board) -> bool:
        """Brief the agent whose status, recall or newest attempt changed longest ago."""
        if not items:
            return False
        newest = {}
        for i in items:
            newest[i.get("agent")] = i  # items are oldest first
        agents = set(board) | {a for a in newest if a}
        pending = []
        for agent in agents:
            status = board.get(agent) or {}
            own = newest.get(agent)
            trigger = f"{status.get('updated_at', '')}|{own['id'] if own else ''}"
            brief = read_brief(self.coral_dir, agent, island_id) or {}
            if brief.get("trigger") == trigger:
                continue
            key = ("brief", island_id, agent, trigger)
            if self._ready(key):
                changed = max(status.get("updated_at", ""), (own or {}).get("created_at", ""))
                pending.append((changed, agent, status, trigger, key))
        if not pending:
            return False
        _, agent, status, trigger, key = min(pending)
        self._brief(agent, status, trigger, key, attempts, items, island_id)
        return True

    def _brief(self, agent, status, trigger, key, attempts, items, island_id) -> None:
        from coral.agent.oneshot import choose_direction, direction_history

        candidates = self.candidates(agent, status, items)
        own = [i for i in items if i.get("agent") == agent]
        recall = status.get("recall", "")
        state = read_state(self.coral_dir, island_id) or {}
        # The direction the agent's next attempt will get (status holds the last one).
        direction = choose_direction(
            direction_history(attempts, agent, self.minimize), agent, self.minimize
        )
        text = self._call(
            key,
            LIBRARIAN_PROMPT.format(
                agent=agent,
                task=self.config.task.name,
                status=status.get("status", ""),
                direction=direction,
                recall=f'Its question for you: "{recall}"\n' if recall else "",
                last=item_text(own[-1]) if own else "no attempts yet",
                state=state.get("text", "(none yet)"),
                items="\n".join(item_text(i) for i in candidates),
                answer=", and the answer to its question if memory has one" if recall else "",
            ),
        )
        if text is None:
            return
        write_brief(
            self.coral_dir,
            agent,
            {
                "text": text,
                "trigger": trigger,
                "for_updated_at": status.get("updated_at", ""),
                "recall": recall,
                "items_considered": [i["id"] for i in candidates],
                "created_at": datetime.now(UTC).isoformat(),
            },
            island_id,
        )

    def candidates(self, agent: str, status: dict, items: list[dict]) -> list[dict]:
        """Items the librarian sees: all of them, or BM25 hits plus best and latest."""
        if len(items) <= SHOW_ALL_ITEMS:
            return items
        query = " ".join(
            [status.get("status", ""), status.get("direction", ""), status.get("recall", "")]
            + [
                i.get("approach", "") + " " + " ".join(i.get("tags") or [])
                for i in items
                if i.get("agent") == agent
            ][-2:]
        )
        ranked = [items[i] for i in bm25_rank(query, [item_text(i) for i in items])[:BM25_TOP]]
        scored = [i for i in items if i.get("score") is not None]
        best = sorted(scored, key=lambda i: i["score"], reverse=not self.minimize)[:3]
        chosen = {i["id"]: i for i in [*ranked, *best, *items[-3:]]}
        return sorted(chosen.values(), key=lambda i: i.get("created_at", ""))
