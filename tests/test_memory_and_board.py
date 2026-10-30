"""One-shot coordination: typed messages, the status board, and team memory."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from coral.agent.memory_service import MemoryService
from coral.config import AgentConfig, CoralConfig, GraderConfig, TaskConfig
from coral.hub.attempts import write_attempt
from coral.hub.memory import bm25_rank, read_brief, read_items, read_state, write_item
from coral.hub.messages import open_requests, recent_for, send
from coral.hub.status import read_board, write_status
from coral.types import Attempt


def _coral(tmp_path: Path) -> Path:
    coral_dir = tmp_path / ".coral"
    (coral_dir / "public").mkdir(parents=True)
    return coral_dir


# --- messages ---------------------------------------------------------------


def test_typed_messages_and_history(tmp_path):
    coral_dir = _coral(tmp_path)
    send(coral_dir, "alpha", "all", "notches help", kind="finding")
    send(coral_dir, "beta", "gamma", "private", kind="warning")
    send(coral_dir, "gamma", "alpha", "hi")

    seen = recent_for(coral_dir, "alpha")
    assert [(m.kind, m.body) for m in seen] == [("finding", "notches help"), ("message", "hi")]
    with pytest.raises(ValueError, match="kind"):
        send(coral_dir, "alpha", "all", "x", kind="shout")


def test_requests_stay_open_until_the_recipient_replies(tmp_path):
    coral_dir = _coral(tmp_path)
    send(coral_dir, "alpha", "beta", "can you check the timer?", kind="request")
    send(coral_dir, "alpha", "all", "anyone tried hulls?", kind="request")

    assert [r.body for r in open_requests(coral_dir, "beta")] == [
        "can you check the timer?",
        "anyone tried hulls?",
    ]
    assert [r.body for r in open_requests(coral_dir, "gamma")] == ["anyone tried hulls?"]
    send(coral_dir, "beta", "alpha", "timer is fine", kind="reply")
    assert open_requests(coral_dir, "beta") == []
    assert len(open_requests(coral_dir, "gamma")) == 1


# --- status board -------------------------------------------------------------


def test_status_board_keeps_last_status(tmp_path):
    coral_dir = _coral(tmp_path)
    write_status(coral_dir, "alpha", status="L-shapes", direction="refine")
    write_status(coral_dir, "alpha", status=None, direction="simplify", recall="what failed?")

    board = read_board(coral_dir)
    assert board["alpha"]["status"] == "L-shapes"
    assert board["alpha"]["direction"] == "simplify" and board["alpha"]["recall"] == "what failed?"


# --- BM25 ---------------------------------------------------------------------


def test_bm25_ranks_relevant_documents_first():
    docs = [
        "rectangle search around mackerel clusters",
        "simulated annealing schedule tuning",
        "L-shape corner notch polygon beats rectangle",
        "faster input parsing",
    ]
    assert bm25_rank("corner notch L-shape", docs)[0] == 2
    assert bm25_rank("annealing", docs)[0] == 1


# --- memory service -----------------------------------------------------------


class FakeLLM:
    def __init__(self):
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if prompt.startswith("You maintain a coding team's shared memory"):
            return json.dumps(
                {
                    "approach": "L-shape notch",
                    "change": "added a corner notch",
                    "result": "+97",
                    "lesson": "notches help near sardine corners",
                    "tags": ["l-shape", "notch"],
                }
            )
        if prompt.startswith("You keep a coding team's state of play"):
            return "L-shapes lead at 2133."
        return "- notches help near sardine corners [abc, beta, 2133]"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _service(tmp_path, every=2):
    coral_dir = _coral(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    commits = []
    for i in range(3):
        (repo / "solution.cpp").write_text(f"int main(){{return {i};}}\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", f"v{i}")
        commits.append(_git(repo, "rev-parse", "HEAD"))
    config = CoralConfig(
        task=TaskConfig(name="Fishing", description="d"),
        grader=GraderConfig(direction="maximize"),
        agents=AgentConfig(
            count=2,
            loop="oneshot",
            memory_state_every=every,
            runtime_options={"target": "solution.cpp"},
        ),
    )
    llm = FakeLLM()
    return coral_dir, commits, llm, MemoryService(config, coral_dir, repo, [None], llm=llm)


def _attempt(commit, parent, agent, score, second):
    return Attempt(
        commit_hash=commit,
        agent_id=agent,
        title=f"try {second}",
        score=score,
        status="improved",
        parent_hash=parent,
        timestamp=f"2099-01-01T00:00:{second:02d}+00:00",
        feedback="Mean score 2133 over 50 public cases",
    )


def test_writer_turns_attempts_into_items_with_diff(tmp_path):
    coral_dir, commits, llm, service = _service(tmp_path)
    write_attempt(coral_dir, _attempt(commits[1], commits[0], "beta", 2133.0, 1))

    assert service.step() is True
    item = read_items(coral_dir)[0]
    assert item["agent"] == "beta" and item["tags"] == ["l-shape", "notch"]
    assert "return 1" in llm.prompts[0] and "Mean score 2133" in llm.prompts[0]


def test_brief_follows_status_and_recall(tmp_path):
    coral_dir, commits, llm, service = _service(tmp_path, every=100)
    write_attempt(coral_dir, _attempt(commits[1], commits[0], "beta", 2133.0, 1))
    service.step()  # writer
    record = write_status(coral_dir, "alpha", status="rectangles", recall="did notches work?")

    assert service.step() is True  # librarian
    brief = read_brief(coral_dir, "alpha")
    assert (
        brief["text"].startswith("- notches help")
        and brief["for_updated_at"] == record["updated_at"]
    )
    assert 'Its question for you: "did notches work?"' in llm.prompts[-1]
    assert service.step() is False  # nothing new to do


def test_state_of_play_every_n_items(tmp_path):
    coral_dir, commits, llm, service = _service(tmp_path, every=2)
    write_attempt(coral_dir, _attempt(commits[1], commits[0], "beta", 2000.0, 1))
    write_attempt(coral_dir, _attempt(commits[2], commits[1], "alpha", 2133.0, 2))
    while service.step():
        pass
    assert read_state(coral_dir)["text"] == "L-shapes lead at 2133."


def test_librarian_uses_bm25_when_memory_is_large(tmp_path):
    coral_dir, _, _, service = _service(tmp_path)
    for i in range(60):
        write_item(
            coral_dir,
            {
                "id": f"{i:040d}",
                "agent": "beta",
                "score": float(i),
                "created_at": f"2099-{i:04d}",
                "approach": "annealing" if i != 7 else "convex hull",
                "tags": [],
            },
        )
    picked = service.candidates("alpha", {"status": "convex hull search"}, read_items(coral_dir))
    ids = {i["id"] for i in picked}
    assert f"{7:040d}" in ids and len(picked) < 60
