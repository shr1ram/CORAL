"""Team endpoint: agent interactions and shared-memory timeline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from coral.hub.attempts import write_attempt
from coral.hub.messages import send
from coral.types import Attempt
from coral.web.api import get_team
from coral.web.team import team_view


def _run(tmp_path: Path, mode: str = "cooperate") -> Path:
    coral_dir = tmp_path / ".coral"
    public = coral_dir / "public"
    for sub in ("attempts", "roles", "notes", "skills"):
        (public / sub).mkdir(parents=True)
    (coral_dir / "config.yaml").write_text(yaml.safe_dump({"agents": {"mode": mode}}))
    for agent in ("alpha", "beta", "gamma"):
        (public / "roles" / f"{agent}.md").write_text("role")
    return coral_dir


def _attempt(commit: str, agent: str, parent: str | None, second: int) -> Attempt:
    return Attempt(
        commit_hash=commit,
        agent_id=agent,
        title=f"attempt {commit}",
        score=float(second),
        status="improved",
        parent_hash=parent,
        timestamp=f"2099-06-01T10:00:{second:02d}+00:00",
    )


def _note(coral_dir: Path, name: str, creator: str, body: str = "") -> None:
    text = f"---\ncreator: {creator}\ncreated: 2099-06-01\n---\n\n# {name}\n\n{body}\n"
    (coral_dir / "public" / "notes" / f"{name}.md").write_text(text)


def test_built_on_links_cross_agent_parents_only(tmp_path):
    coral_dir = _run(tmp_path)
    write_attempt(coral_dir, _attempt("a1", "alpha", "base", 1))
    write_attempt(coral_dir, _attempt("b1", "beta", "a1", 2))  # beta builds on alpha
    write_attempt(coral_dir, _attempt("a2", "alpha", "a1", 3))  # alpha on itself

    view = team_view(coral_dir)

    built_on = [(i["from"], i["to"]) for i in view["interactions"] if i["kind"] == "built_on"]
    assert built_on == [("alpha", "beta")]
    assert [m["kind"] for m in view["memory"]].count("attempt") == 3


def test_messages_and_broadcasts_become_interactions(tmp_path):
    coral_dir = _run(tmp_path)
    send(coral_dir, "alpha", "beta", "taking annealing")
    send(coral_dir, "gamma", "all", "grid init helps")

    view = team_view(coral_dir)

    pairs = {(i["from"], i["to"], i["kind"]) for i in view["interactions"]}
    assert ("alpha", "beta", "message") in pairs
    assert ("gamma", "alpha", "broadcast") in pairs
    assert ("gamma", "beta", "broadcast") in pairs
    assert not any(i["to"] == "gamma" and i["from"] == "gamma" for i in view["interactions"])
    assert [m["body"] for m in view["messages"]] == ["taking annealing", "grid init helps"]
    assert view["mode"] == "cooperate"


def test_note_reference_links_point_from_cited_author_to_citer(tmp_path):
    coral_dir = _run(tmp_path, mode="compete")
    _note(coral_dir, "grid-init", "alpha")
    _note(coral_dir, "follow-up", "beta", body="Based on [grid](grid-init.md)")

    view = team_view(coral_dir)

    refs = [(i["from"], i["to"]) for i in view["interactions"] if i["kind"] == "note_ref"]
    assert refs == [("alpha", "beta")]
    assert sorted(m["label"] for m in view["memory"] if m["kind"] == "note") == [
        "follow-up",
        "grid-init",
    ]
    assert view["mode"] == "compete"


def test_timeline_is_sorted_and_lists_agents(tmp_path):
    coral_dir = _run(tmp_path)
    write_attempt(coral_dir, _attempt("b1", "beta", None, 9))
    write_attempt(coral_dir, _attempt("a1", "alpha", None, 1))

    view = team_view(coral_dir)

    times = [m["t"] for m in view["memory"]]
    assert times == sorted(times)
    assert view["agents"] == ["alpha", "beta", "gamma"]


async def test_team_endpoint_returns_json(tmp_path):
    coral_dir = _run(tmp_path)
    send(coral_dir, "alpha", "beta", "hi")
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(coral_dir=coral_dir)))

    response = await get_team(request)

    assert response.status_code == 200
    assert json.loads(response.body)["messages"][0]["body"] == "hi"


def test_run_start_comes_from_run_directory_name(tmp_path):
    from datetime import datetime

    from coral.web.team import _note_time, _run_start

    coral_dir = tmp_path / "2026-10-30_101500" / ".coral"
    coral_dir.mkdir(parents=True)
    assert _run_start(coral_dir) == datetime(2026, 10, 30, 10, 15).timestamp()

    other = tmp_path / "custom" / ".coral"
    other.mkdir(parents=True)
    (other / "config.yaml").write_text("{}")
    assert _run_start(other) == (other / "config.yaml").stat().st_mtime

    note = tmp_path / "n.md"
    note.write_text("x")
    assert _note_time(note, "2026-10-30T10:20:00") == datetime(2026, 10, 30, 10, 20).timestamp()
    assert _note_time(note, "2026-10-30") == note.stat().st_mtime
