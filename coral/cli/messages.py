"""Agent-to-agent messaging commands: msg, inbox (cooperate mode only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from coral.cli._helpers import find_coral_dir_and_island, read_agent_id


def _read_mode(coral_dir: Path) -> str:
    config_path = coral_dir / "config.yaml"
    if config_path.exists():
        import yaml

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        return (config.get("agents") or {}).get("mode", "compete")
    return "compete"


def _require_cooperate(coral_dir: Path) -> None:
    mode = _read_mode(coral_dir)
    if mode != "cooperate":
        print(
            f"Error: messaging is only available in cooperate mode (this run is '{mode}').",
            file=sys.stderr,
        )
        sys.exit(1)


def _find_run(args: argparse.Namespace) -> tuple[Path, str | None]:
    """The run's .coral dir and island; exits cleanly if no island can be chosen."""
    from coral.hub._island import island_root

    coral_dir, island_id = find_coral_dir_and_island(
        getattr(args, "task", None), getattr(args, "run", None)
    )
    _require_cooperate(coral_dir)
    try:
        island_root(coral_dir, island_id)
    except ValueError:
        print(
            "Error: this is a multi-island run, and messages belong to one island. "
            "Run this inside an agent worktree on that island.",
            file=sys.stderr,
        )
        sys.exit(1)
    return coral_dir, island_id


def _known_agents(coral_dir: Path, island_id: str | None) -> set[str]:
    from coral.hub._island import island_root

    return {p.stem for p in (island_root(coral_dir, island_id) / "roles").glob("*.md")}


def cmd_msg(args: argparse.Namespace) -> None:
    """Send a message to one agent, or to all agents with `all`.

    Examples:
      coral msg poseidon "Taking the annealing schedule for the next 3 evals"
      coral msg all "Note notes/findings/grid-init.md: grid init beats random by 0.02"
    """
    from coral.hub.messages import BROADCAST, send

    coral_dir, island_id = _find_run(args)
    # Outside an agent worktree (a human steering the run) the sender is "operator".
    sender = args.sender or read_agent_id()
    if sender == "unknown":
        sender = "operator"
    if args.to != BROADCAST:
        known = _known_agents(coral_dir, island_id)
        if known and args.to not in known:
            print(
                f"Error: unknown agent '{args.to}'. Known: {', '.join(sorted(known))}, or 'all'.",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.to == sender:
            print("Error: you can't message yourself.", file=sys.stderr)
            sys.exit(1)
    try:
        message = send(
            coral_dir,
            sender,
            args.to,
            " ".join(args.body),
            island_id=island_id,
            kind=getattr(args, "kind", "message") or "message",
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Sent to {message.to}.")


def cmd_inbox(args: argparse.Namespace) -> None:
    """Show unread messages for this agent and mark them read.

    Examples:
      coral inbox          Unread messages
      coral inbox --all    Every message in the run, read or not
    """
    from coral.hub.messages import format_messages, list_messages, mark_read, read_unread

    coral_dir, island_id = _find_run(args)
    if args.all:
        messages = list_messages(coral_dir, island_id)
        print(format_messages(messages) if messages else "No messages yet.")
        return
    agent_id = read_agent_id()
    if agent_id == "unknown":
        # Outside a worktree there is no reader to mark messages read for.
        print(
            "Error: `coral inbox` shows an agent's unread messages; run it inside an agent "
            "worktree, or use `coral inbox --all` to see every message.",
            file=sys.stderr,
        )
        sys.exit(1)
    messages = read_unread(coral_dir, agent_id, island_id)
    if not messages:
        print("No unread messages.")
        return
    print(format_messages(messages))
    mark_read(coral_dir, agent_id, [m.id for m in messages], island_id)
