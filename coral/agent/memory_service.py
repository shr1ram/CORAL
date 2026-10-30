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

Agents never wait for this: the worker shows whatever brief is ready.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
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

SHOW_ALL_ITEMS = 40  # below this the librarian sees every item
BM25_TOP = 20
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


def chat_once(base_url: str, model: str, prompt: str, max_tokens: int = 1500) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        data = json.load(response)
    return (data["choices"][0]["message"].get("content") or "").strip()


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
        """Do the most urgent pending job; True if one was done."""
        return (
            self._write_next_item(island_id)
            or self._write_next_brief(island_id)
            or self._write_state(island_id)
        )

    # --- writer ------------------------------------------------------------

    def _write_next_item(self, island_id: str | None) -> bool:
        attempts = sorted(read_attempts(self.coral_dir, island_id), key=lambda a: a.timestamp)
        done = {i["id"] for i in read_items(self.coral_dir, island_id)}
        todo = [a for a in attempts if a.status != "pending" and a.commit_hash not in done]
        if not todo:
            return False
        attempt = todo[0]
        earlier = [a for a in attempts if a.timestamp < attempt.timestamp and a.score is not None]
        agent_best = _best(
            [a.score for a in earlier if a.agent_id == attempt.agent_id], self.minimize
        )
        team_best = _best([a.score for a in earlier], self.minimize)
        board = read_board(self.coral_dir, island_id)
        items = read_items(self.coral_dir, island_id)
        prompt = WRITER_PROMPT.format(
            task=self.config.task.name,
            agent=attempt.agent_id,
            status=(board.get(attempt.agent_id) or {}).get("status", ""),
            title=attempt.title,
            score=attempt.score,
            outcome=attempt.status,
            agent_best=agent_best,
            team_best=team_best,
            feedback=(attempt.feedback or "")[:1500],
            diff=self._diff(attempt.parent_hash, attempt.commit_hash)[:6000],
            tags=", ".join(known_tags(items)[:40]) or "(none yet)",
        )
        fields = _parse_json(self.llm(prompt)) or {}
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
        return True

    def _diff(self, parent: str | None, commit: str) -> str:
        if not parent:
            return "(no parent)"
        cmd = ["git", "-C", str(self.repo_dir), "diff", parent, commit]
        if self.target:
            cmd += ["--", self.target]
        out = subprocess.run(cmd, capture_output=True, text=True)
        return out.stdout or "(no change)"

    # --- state of play -----------------------------------------------------

    def _write_state(self, island_id: str | None) -> bool:
        items = read_items(self.coral_dir, island_id)
        state = read_state(self.coral_dir, island_id) or {}
        every = self.config.agents.memory_state_every
        if not items or len(items) - state.get("n_items", 0) < every:
            return False
        board = read_board(self.coral_dir, island_id)
        text = self.llm(
            STATE_PROMPT.format(
                task=self.config.task.name,
                items="\n".join(item_text(i) for i in items[-60:]),
                board="\n".join(f"- {a}: {s.get('status', '')}" for a, s in board.items()) or "-",
            )
        )
        write_state(
            self.coral_dir,
            {"text": text, "n_items": len(items), "created_at": datetime.now(UTC).isoformat()},
            island_id,
        )
        return True

    # --- librarian ---------------------------------------------------------

    def _write_next_brief(self, island_id: str | None) -> bool:
        items = read_items(self.coral_dir, island_id)
        if not items:
            return False
        board = read_board(self.coral_dir, island_id)
        for agent, status in sorted(board.items(), key=lambda kv: kv[1].get("updated_at", "")):
            brief = read_brief(self.coral_dir, agent, island_id) or {}
            if brief.get("for_updated_at", "") >= status.get("updated_at", ""):
                continue
            self._brief(agent, status, items, island_id)
            return True
        return False

    def _brief(self, agent: str, status: dict, items: list[dict], island_id: str | None) -> None:
        candidates = self.candidates(agent, status, items)
        own = [i for i in items if i.get("agent") == agent]
        last = own[-1] if own else None
        recall = status.get("recall", "")
        state = read_state(self.coral_dir, island_id) or {}
        text = self.llm(
            LIBRARIAN_PROMPT.format(
                agent=agent,
                task=self.config.task.name,
                status=status.get("status", ""),
                direction=status.get("direction", ""),
                recall=f'Its question for you: "{recall}"\n' if recall else "",
                last=item_text(last) if last else "no attempts yet",
                state=state.get("text", "(none yet)"),
                items="\n".join(item_text(i) for i in candidates),
                answer=", and the answer to its question if memory has one" if recall else "",
            )
        )
        write_brief(
            self.coral_dir,
            agent,
            {
                "text": text,
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
