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

## llama-server tuning (2026-10-30/31)

Benchmark: 4 concurrent clients on Flint, "agentic" (6 growing turns, 400 tokens each) and "one-shot"
(one ~6k-token program each). Scripts and raw results:
`/private/tmp/claude-501/-Users-shriramruppa-Git/263c584b-0045-4de0-9e63-c3552c930505/scratchpad/bench/`
(`bench.py`, `kld.sh`, `results*.jsonl`, `round2.jsonl`, `round3.jsonl`, `overnight-summary.md`).

**Caveat:** overnight the battery drained to 7% while plugged in (the load outdrew the charger), and
the Mac idle-slept at 03:03 when the driver's `caffeinate` exited. Round 2 shows ~10% drift from
the first to the second pass; round 4 (build-up test) was invalid and was stopped. Treat differences
under ~10% as unconfirmed.

Settings (vs `-b 2048 -ub 512`, which beat the old `-b 256 -ub 128` by +26% agentic):
- No gain: `-b 512`, `-ctxcp 6`, `--no-cache-idle-slots`, `--cache-ram-mlock`, n-gram speculation
  (any draft size), `-ub 1024`.
- Much worse: `-kvu` (unified KV), -30% to -60%.
- Rejected on quality: `LLAMA_ATTN_ROT_DISABLE=1` (KLD 0.25, top-1 agreement 89%).

Code changes (llama.cpp-flint worktrees in `../llama-wt/`, branches `opt/*`, each behind an env var,
all pass the quality check: KLD vs stock ~1e-6, top-1 agreement 99.98%):

| change | branch / switch | agentic total | per agent | one-shot |
|---|---|---|---|---|
| base | master | 126 | 46 | 205 |
| GQA packing in FA decode (Q=2) | `opt/gqa-pack`, `GGML_METAL_FA_GQA_PACK=1` | 140 | 53 | 226 |
| native MoE amax (drop graph chain) | `opt/server-easy`, `LLAMA_MOE_NATIVE_AMAX=1` | 133 | 47 | 223 |
| fused QKV + gate/up GGUF | `opt/fused-repack` + `flint-int4-q4_1-fused.gguf` | 135 | 46 | 222 |
| non-adjacent slots in one pass | `opt/noncontig-streams`, `LLAMA_KV_NONCONTIG_STREAMS=1` | 135 | 48 | 220 |
| shortest prompt first | `opt/server-easy`, `LLAMA_SERVER_PREFILL_SHORTEST_FIRST=1` | 135 | 44 | 219 |
| prefill cap while decoding | `opt/server-easy`, `LLAMA_SERVER_PREFILL_CAP=1` | 133 | 43 | 221 |
| exact sigmoid top-k fusion | `opt/sigmoid-fusion`, `GGML_METAL_FUSE_SIGMOID_MOE=1` | 129 | 49 | 209 |

Everything combined (`opt/combined`, all switches plus `-ctxcp 6 --no-cache-idle-slots
--cache-ram-mlock`): agentic 126 vs 120, one-shot 206 vs 196, but per-agent agentic speed fell
46 -> 39. The prefill switches trade per-agent speed for total, and the sigmoid fusion is slower.

Follow-up (2026-10-31 morning, battery, 3 interleaved runs each; `round5.jsonl`):

| config | agentic total | per agent | wait mean / p95 | one-shot |
|---|---|---|---|---|
| base (`-b 2048 -ub 512`) | 127 [122-137] | 46.7 | 3.6 s / 8.5 s | 210 [199-228] |
| GQA packing only | 126 [124-127] | 44.0 | 3.0 s / 7.5 s | 214 [212-218] |
| core + shortest-first + `-ctxcp 6` | 132 [131-132] | 45.1 | 2.7 s / 6.1 s | 219 [210-226] |

Verified in the real server: GQA packing engages on decode (G=8, q=2) and noncontig turns {0,2}
decode steps into one ubatch. The overnight +10% for GQA packing was noise; the code changes are
worth at most ~5% end to end, below this benchmark's ±10% run-to-run noise. The only confirmed win
is `-b 2048 -ub 512`. To resolve small effects, use `llama-batched-bench` (fixed workload, ~1-2%
noise) on mains power.
