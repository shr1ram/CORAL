# Notes

Open issues and caveats for the `modes` branch (coordination study: Flint on ALE-Bench AHC039).

## Things to know

- **Subagents haven't been tried against real `claude`.** The sandboxed broker runs `claude -p`
  with `--restricted`, `--setting-sources project`, `--strict-mcp-config`, a separate scratch
  working folder (`<run>/.subagent-scratch/<id>`) and the task sent on stdin. That was only tested
  with a fake `claude`. On the first real subagent call, check that:
  - Bash still works when `--tools` names it;
  - the personal `~/.claude/CLAUDE.md` isn't loaded.
- **Agents can still edit some shared files.** They can still write to the team memory, status and
  message files, so an agent could in principle forge entries. Manager-only mode (subagents on) is
  also still just a prompt instruction, since agents keep bash.
- **Late agents and request folders.** The srt sandbox blocks writes to other agents' subagent
  request folders only for agents known when the settings were built (roster plus worktrees on
  disk). A later agent gets protection on the next restart.
- **The model server build.** `experiments/compare_modes.sh` starts the patched llama.cpp build
  (`llama.cpp-flint`, commit 6358cf3, `--cache-ram-mlock` with the unlock-before-free fix) at the
  start of every run.
