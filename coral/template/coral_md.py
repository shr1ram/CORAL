"""Generate CORAL.md agent instructions from template.

``agents.instructions: full`` renders the original long-form guide, but intentionally
differs from upstream CORAL: its Claude Code native subagent guidance (``{shared_dir}/agents/``,
"spawn specialist subagents") is removed, because subagents here only exist as
``coral subagent`` in manager mode (agents.subagents), described by ``_MANAGER_SECTION``.
"""

from __future__ import annotations

from pathlib import Path

from coral.config import CoralConfig

_TEMPLATE_PATH = Path(__file__).parent / "coral.md.template"
_SINGLE_TEMPLATE_PATH = Path(__file__).parent / "coral_single.md.template"
_MINIMAL_TEMPLATE_PATH = Path(__file__).parent / "coral_minimal.md.template"
_MINIMAL_SINGLE_TEMPLATE_PATH = Path(__file__).parent / "coral_minimal_single.md.template"

_MANAGER_SECTION = """

## Manager mode (subagents)

You are a **manager**. You do no hands-on work yourself: no writing or editing code, no running
experiments or tests, no literature review, no writing notes or skills. Delegate all of it to
Claude subagents, and spend your effort on deciding what to try, writing clear briefs, reviewing
what comes back, scoring it, and coordinating with the team.

- `coral subagent run [--type general|deep-researcher|librarian] [--no-wait] "<brief>"` starts a
  subagent; without `--no-wait` it blocks and prints the subagent's report. Collect a `--no-wait`
  subagent with `coral subagent wait <id>`; `coral subagent list` shows yours.
- At most {max_running} of your subagents can run at once. They all work in your directory (they
  can edit, compile and test code; they cannot use git{web}), so give parallel subagents separate
  files, or have each write a new file that you then choose between.
- Briefs must be self-contained: the goal, the relevant files, what to try, how to test it, and
  what to report back. Subagents do not see this document, your history or the team's state
  unless your brief points them to it.
- `general` writes and tests code and analyses results; `deep-researcher` investigates
  techniques{researcher_scope}; `librarian` organises the shared notes.
- You still run `coral eval` to score your directory, read `coral log`, `coral show` and
  `coral notes`, use `coral checkout`, and message teammates. Whenever these instructions or a
  heartbeat ask you to write something (code, a note, a skill, your role or focus file), brief a
  subagent to do it.
- Subagent time and money are limited ({run_budget}): give each subagent a substantial, well-scoped
  task rather than many tiny ones.
"""

_COOPERATE_SECTION = """

## Messaging other agents (cooperate mode)

This run is in **cooperate** mode: besides the shared attempts, notes and skills,
you can message the other agents directly. Your id is `{agent_id}`.

```bash
coral msg <agent-id> "text"   # message one agent (ids: see `coral log` or `{shared_dir}/roles/`)
coral msg all "text"          # message every other agent
coral inbox                   # read unread messages (they are also appended to your eval results)
coral inbox --all             # full message history
```

Use messages for things that need to reach a specific agent soon:
- **Claims:** "I'm taking X for the next few evals" so nobody duplicates it.
- **Warnings:** "your commit abc123 fails on edge case Y".
- **Requests and handoffs:** "can someone profile Z?", "I found a lead on W but I'm on V; who wants it?".
- **Answers** to questions sent to you.

Keep each message short and actionable (under 2000 characters). Findings that everyone
should keep belong in a note; send the note's path instead of repeating it. Don't send
acknowledgements or small talk: every message costs the reader context and time.
"""


def generate_coral_md(
    config: CoralConfig,
    agent_id: str,
    single_agent: bool = False,
    shared_dir: str = ".claude",
    island_id: str | int | None = None,
) -> str:
    """Produce the CORAL.md file that agents read at startup.

    Args:
        config: The coral config
        agent_id: This agent's ID
        single_agent: If True, use simplified single-agent template (no sharing references)
        shared_dir: Name of the shared state directory (e.g. ".claude", ".codex", ".opencode")
        island_id: Agent's island in multi-island mode (formats a provenance hint)
    """
    # Build optional sections
    tips_section = ""
    if config.task.tips:
        tips_section = f"\n## Tips\n{config.task.tips}\n"

    # Determine score direction from config or grader type
    score_direction = _get_score_direction(config)

    if config.agents.instructions == "minimal":
        minimal_path = _MINIMAL_SINGLE_TEMPLATE_PATH if single_agent else _MINIMAL_TEMPLATE_PATH
        rendered = minimal_path.read_text(encoding="utf-8").format(
            task_name=config.task.name,
            task_description=config.task.description.strip(),
            tips_section=tips_section,
            score_direction=score_direction,
            agent_id=agent_id,
            shared_dir=shared_dir,
            research_line=_minimal_research_line(config, shared_dir),
        )
        return rendered + _team_sections(config, agent_id, single_agent, shared_dir, island_id)

    template_path = _SINGLE_TEMPLATE_PATH if single_agent else _TEMPLATE_PATH
    # Claude Code's native subagent guidance never applies: subagents, when on,
    # go through `coral subagent` (see the manager section). Stripped from the
    # template before formatting so task text mentioning subagents survives.
    template = _without_subagents(template_path.read_text(encoding="utf-8"))

    # Research step is conditional
    research_enabled = config.agents.research
    if research_enabled:
        workflow_summary = "research → plan → edit → eval → repeat"
        research_section = (
            "\n## 1. Research\n\n"
            "**On your first iteration and whenever you're changing direction**, "
            "invest time in deep research before planning. "
            f"Read the `deep-research` skill (`{shared_dir}/skills/deep-research/SKILL.md`) "
            "for a structured research workflow.\n\n"
            "**Research steps:**\n"
            "- **Understand the problem deeply** — read the grader source at "
            f"`{shared_dir}/grader/` (the exact code that scores you — read it to "
            "understand the objective, but never modify it), identify constraints "
            "and evaluation criteria.\n"
            + (
                "- **Survey the literature** — use web search to find state-of-the-art approaches, "
                "academic papers, benchmark comparisons, and existing implementations. "
                'Search broadly first (`"[problem] state of the art"`), then drill into '
                "specific techniques.\n"
                if config.agents.internet
                else "- **Work from what you have** — you have no internet access: draw on what "
                "you know, the problem statement, the grader and the files in your directory.\n"
            )
            + "- **Review domain knowledge** — if the task involves specialized domains "
            "(biology, chemistry, physics, math), research the underlying science. "
            "Understanding the domain often reveals approaches that pure ML/CS thinking misses.\n"
            "- **Analyze existing solutions** — check shared notes, past attempts, and "
            "what has been tried before. Build on what's known.\n"
            "- **Compare 2-4 candidate approaches** — document trade-offs, evidence, "
            "and implementation complexity for each.\n"
            f"- **Track coverage** — maintain the research coverage ledger at "
            f"`{shared_dir}/notes/research/_coverage.md` (task dimensions × "
            "covered/partial/missing); target the gaps, don't re-cover what's done.\n"
            f"- **Write a research summary** — save findings to `{shared_dir}/notes/research/[topic].md` "
            f"so all agents benefit (see `{shared_dir}/skills/deep-research/references/` for "
            f"templates), then run `{shared_dir}/skills/deep-research/scripts/check_grounding.py "
            f"{shared_dir}/notes` to verify every finding traces to a saved source.\n\n"
            "**When to research:**\n"
            "- First iteration: always. Understand the landscape before writing code.\n"
            "- After getting stuck (3+ evals without improvement): step back and "
            "look for new angles.\n"
            "- When pivoting to a fundamentally different approach.\n"
            "- When the task involves unfamiliar domain knowledge.\n\n"
            "**When to skip:** If you have a clear plan from your last eval's feedback "
            "and just need to iterate on an existing approach, go straight to Step 2.\n"
        )
        step_offset = 2  # Plan starts at step 2
        research_back_reference = " (or **Step 1: Research** if you need a new direction)"
        repeat_research_hint = "go back to **Step 1: Research** to find new techniques" + (
            " via web search, " if config.agents.internet else ", "
        )
    else:
        workflow_summary = "plan → edit → eval → repeat"
        research_section = ""
        step_offset = 1  # Plan starts at step 1
        research_back_reference = ""
        repeat_research_hint = "research new techniques, "

    rendered = template.format(
        task_name=config.task.name,
        task_description=config.task.description,
        tips_section=tips_section,
        score_direction=score_direction,
        agent_id=agent_id,
        shared_dir=shared_dir,
        workflow_summary=workflow_summary,
        research_section=research_section,
        plan_step_num=step_offset,
        edit_step_num=step_offset + 1,
        eval_step_num=step_offset + 2,
        results_step_num=step_offset + 3,
        knowledge_step_num=step_offset + 4,
        research_back_reference=research_back_reference,
        repeat_research_hint=repeat_research_hint,
    )
    return rendered + _team_sections(config, agent_id, single_agent, shared_dir, island_id)


def _minimal_research_line(config: CoralConfig, shared_dir: str) -> str:
    """Orient-step research hint; offline wording when agents have no internet."""
    if not config.agents.research:
        return ""
    if config.agents.internet:
        return (
            " Before your first implementation and when changing direction, research the "
            f"problem with the `deep-research` skill; track coverage in "
            f"`{shared_dir}/notes/research/_coverage.md`."
        )
    return (
        " Before your first implementation and when changing direction, study the problem "
        "statement, the grader source and the files in your directory (you have no internet "
        "access)."
    )


def _without_subagents(text: str) -> str:
    """Drop the full guide's native subagent guidance from template text (never task text)."""
    lines = []
    for line in text.splitlines(keepends=True):
        line = line.replace(" Often pairs with the `librarian` subagent.", "")
        if "subagent" not in line.lower():
            lines.append(line)
    return "".join(lines)


def _team_sections(
    config: CoralConfig,
    agent_id: str,
    single_agent: bool,
    shared_dir: str,
    island_id: str | int | None,
) -> str:
    """Sections appended to either template: island provenance and messaging."""
    out = ""
    # Multi-island provenance hint: only when there's more than one island.
    if island_id is not None and config.islands.count > 1:
        out += (
            "\n\n## Multi-island provenance\n\n"
            f"You are working on island `{island_id}` in a multi-island run "
            f"({config.islands.count} islands total). Other islands exist with "
            "their own attempts, notes, and skills, but you cannot see their "
            "state directly. Each island evolves independently.\n"
        )
    if config.agents.mode == "cooperate" and not single_agent:
        out += _COOPERATE_SECTION.format(agent_id=agent_id, shared_dir=shared_dir)
    if config.agents.subagents:
        a = config.agents
        out += _MANAGER_SECTION.format(
            max_running=a.subagent_max_running,
            web="" if a.internet else " or the internet",
            researcher_scope=(
                " (including web research)"
                if a.internet
                else " from the code, grader and notes (no web access)"
            ),
            run_budget=(
                f"about ${a.subagent_run_budget_usd:g} for the whole team this run"
                if a.subagent_run_budget_usd
                else "per-call budgets apply"
            ),
        )
    return out


def _get_score_direction(config: CoralConfig) -> str:
    """Return a human-readable description of what 'better' means for this grader."""
    if config.grader.direction == "minimize":
        return "lower is better"
    return "higher is better"
