#!/bin/bash
# Run CORAL on ALE-Bench ahc039 under several coordination conditions, with repeats.
#
# Each run gets the same agents, wall-clock budget and a freshly started Flint
# llama-server (so no prompt cache carries over between runs). The condition order
# alternates every repeat to avoid order effects. A run is stopped gracefully and
# the script waits until its manager, grader and agents are gone before the next.
#
#   CONDITIONS="oneshot:compete oneshot:cooperate" REPEATS=3 MINUTES=45 ./compare_modes.sh
set -uo pipefail

CONDITIONS=${CONDITIONS:-"oneshot:compete oneshot:cooperate"}  # loop:mode pairs
REPEATS=${REPEATS:-1}
MINUTES=${MINUTES:-60}
CORAL_DIR=/Users/shriramruppa/Side/autoresearch/CORAL
TASK=ale-bench-ahc039-purse-seine-fishing
RESULTS="$CORAL_DIR/results/$TASK"
LOGS=${LOGS:-$CORAL_DIR/results/experiments/$(date +%Y-%m-%d_%H%M%S)}
MODEL_DIR=/Users/shriramruppa/Side/autoresearch/local-autoresearch/runs/flint-runtime
SERVER_BIN=/Users/shriramruppa/Side/autoresearch/llama.cpp-flint/build/bin/llama-server
# Tuned 2026-10-31 (see notes.md): -b 2048 -ub 512 is +26% agentic over -b 256 -ub 128.
# --cache-ram-mlock keeps cached prompts out of swap (safety, not speed; it can lock ~1.5 GB before the
# macOS wire limit, then warns once and carries on unlocked).
SERVER_FLAGS=${SERVER_FLAGS:-"--alias flint -ngl all -c 262144 -np 4 --cont-batching --cache-ram 6144 --cache-ram-mlock -b 2048 -ub 512 -t 8 -fa on -ctk q8_0 -ctv q8_0 --jinja --temp 1 --top-p 0.95 --top-k 20 --min-p 0"}
PORT=11501
mkdir -p "$LOGS"
export DYLD_LIBRARY_PATH=/opt/homebrew/lib
cd "$CORAL_DIR"

log() { echo "$(date +%T) $*"; }

stop_server() {
    pkill -f "$SERVER_BIN" 2>/dev/null
    for _ in $(seq 1 60); do pgrep -f "$SERVER_BIN" >/dev/null || return 0; sleep 1; done
    log "server did not exit; killing"; pkill -9 -f "$SERVER_BIN"; sleep 2
}

start_server() {
    stop_server
    # The patched build must load its own libraries, not Homebrew's.
    # shellcheck disable=SC2086
    env -u DYLD_LIBRARY_PATH "$SERVER_BIN" -m "$MODEL_DIR/flint-int4-q4_1.gguf" $SERVER_FLAGS --host 127.0.0.1 --port $PORT \
        > "$LOGS/server-$1.log" 2>&1 &
    for _ in $(seq 1 300); do
        curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null && return 0
        pgrep -f "$SERVER_BIN" >/dev/null || { log "server failed to start"; return 1; }
        sleep 1
    done
    log "server not healthy after 300 s"; return 1
}

# Wait until nothing from this run directory is still running (manager, grader,
# agents and their children all have the run dir in their command line or cwd).
wait_run_gone() {
    local run_dir=$1
    for _ in $(seq 1 120); do
        pgrep -f "$run_dir" >/dev/null || return 0
        sleep 1
    done
    log "processes from $run_dir still alive after 120 s; killing them"
    pkill -9 -f "$run_dir"
    sleep 2
}

run_one() {
    local loop=$1 mode=$2 tag=$3
    local before
    before=$(readlink "$RESULTS/latest" 2>/dev/null || true)
    log "starting $tag ($loop/$mode) for $MINUTES min"
    start_server "$tag" || return 1
    .venv/bin/coral start -c examples/ale_bench_ahc039/task.yaml \
        agents.loop="$loop" agents.mode="$mode" > "$LOGS/$tag.log" 2>&1 &
    local manager=$!

    # Wait for this run's directory to appear before starting its dashboard.
    local run_dir=""
    for _ in $(seq 1 120); do
        local now
        now=$(readlink "$RESULTS/latest" 2>/dev/null || true)
        if [ -n "$now" ] && [ "$now" != "$before" ]; then run_dir="$RESULTS/$now"; break; fi
        kill -0 "$manager" 2>/dev/null || break
        sleep 1
    done
    if [ -z "$run_dir" ]; then
        log "$tag: run never started; see $LOGS/$tag.log"
        wait "$manager" 2>/dev/null
        return 1
    fi
    log "$tag run dir: $run_dir"
    PYTHONUNBUFFERED=1 .venv/bin/coral ui --task "$TASK" --run "$(basename "$run_dir")" --no-open \
        --port 8420 --inference-url "http://127.0.0.1:$PORT" > "$LOGS/$tag-ui.log" 2>&1 &
    local ui=$!

    local deadline=$(( $(date +%s) + MINUTES * 60 ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        if ! kill -0 "$manager" 2>/dev/null; then
            log "$tag: manager exited early; see $LOGS/$tag.log"
            break
        fi
        sleep 15
    done

    .venv/bin/coral stop --task "$TASK" --run "$(basename "$run_dir")" >> "$LOGS/$tag.log" 2>&1
    wait "$manager" 2>/dev/null
    wait_run_gone "$run_dir"
    kill "$ui" 2>/dev/null  # free port 8420 for the next run's dashboard
    wait "$ui" 2>/dev/null
    .venv/bin/coral log --task "$TASK" --run "$(basename "$run_dir")" -n 20 > "$LOGS/$tag-leaderboard.txt" 2>&1
    log "finished $tag"
}

read -r -a conditions <<< "$CONDITIONS"
for repeat in $(seq 1 "$REPEATS"); do
    order=("${conditions[@]}")
    if (( repeat % 2 == 0 )); then  # alternate the order every repeat
        reversed=()
        for (( i=${#order[@]}-1; i>=0; i-- )); do reversed+=("${order[i]}"); done
        order=("${reversed[@]}")
    fi
    for condition in "${order[@]}"; do
        run_one "${condition%%:*}" "${condition##*:}" "r${repeat}-${condition/:/-}"
    done
done
stop_server
log "all done"
