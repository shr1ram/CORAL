"""`coral subagent`: delegate work to Claude subagents (agents.subagents)."""

from __future__ import annotations

import argparse
import sys

from coral.cli._helpers import find_coral_dir_and_island, read_agent_id


def _agents_config(coral_dir) -> dict:
    import yaml

    try:
        config = yaml.safe_load((coral_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    except OSError:
        return {}
    return config.get("agents") or {}


def _print_result(result: dict) -> None:
    if not result.get("ok"):
        print(f"Error: {result.get('error', 'subagent failed')}", file=sys.stderr)
        sys.exit(1)
    cost = result.get("cost_usd")
    print(result.get("result", ""))
    if cost is not None:
        print(
            f"\n[subagent {result.get('kind')} {result['id'][:8]}: ~${cost:.2f}]", file=sys.stderr
        )


def cmd_subagent(args: argparse.Namespace) -> None:
    """Run, wait for, or list subagents.

    Examples:
      coral subagent run "Implement simulated annealing in solution.cpp; test on inputs/"
      coral subagent run --no-wait --type deep-researcher "Survey AHC039 techniques"
      coral subagent wait 3f2a1b9c
      coral subagent list
    """
    from coral.hub.subagents import (
        list_results,
        pending_requests,
        read_definitions,
        running_count,
        submit,
        wait_result,
    )

    coral_dir, island_id = find_coral_dir_and_island(
        getattr(args, "task", None), getattr(args, "run", None)
    )
    agents = _agents_config(coral_dir)
    if not agents.get("subagents"):
        print("Error: subagents are off for this run (agents.subagents).", file=sys.stderr)
        sys.exit(1)
    timeout = int(agents.get("subagent_timeout", 900)) + 120
    agent_id = read_agent_id()

    if args.action == "list":
        mine = [r for r in pending_requests(coral_dir, island_id) if r.agent_id == agent_id]
        for r in mine:
            print(f"running  {r.id[:8]}  {r.kind:16s} {r.task[:80]}")
        for r in list_results(coral_dir, island_id):
            if r.get("agent_id") == agent_id:
                status = "done" if r.get("ok") else "failed"
                print(
                    f"{status:8s} {r['id'][:8]}  {r.get('kind', ''):16s} {r.get('task', '')[:80]}"
                )
        return

    if args.action == "wait":
        result = wait_result(coral_dir, args.id, island_id, timeout=timeout)
        if result is None:
            print(f"Error: no result for {args.id} yet.", file=sys.stderr)
            sys.exit(1)
        _print_result(result)
        return

    kinds = read_definitions(coral_dir, island_id)
    if args.type not in kinds:
        print(
            f"Error: unknown subagent type '{args.type}'. Available: {', '.join(sorted(kinds))}.",
            file=sys.stderr,
        )
        sys.exit(1)
    limit = int(agents.get("subagent_max_running", 2))
    if running_count(coral_dir, agent_id, island_id) >= limit:
        print(
            f"Error: you already have {limit} subagents running (the limit). "
            "Collect one with `coral subagent wait <id>` (see `coral subagent list`).",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        request = submit(coral_dir, agent_id, args.type, " ".join(args.prompt), island_id)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    if args.no_wait:
        print(
            f"Started subagent {request.id[:8]} ({args.type}). Collect it with `coral subagent wait {request.id[:8]}`."
        )
        return
    print(f"[subagent {args.type} {request.id[:8]}] working...", file=sys.stderr, flush=True)
    result = wait_result(coral_dir, request.id, island_id, timeout=timeout)
    if result is None:
        print(
            f"Error: no answer yet; check later with `coral subagent wait {request.id[:8]}`.",
            file=sys.stderr,
        )
        sys.exit(1)
    _print_result(result)
