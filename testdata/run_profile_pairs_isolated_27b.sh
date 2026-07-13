#!/usr/bin/env bash
set -euo pipefail

PROFILE_ROOT="${PROFILE_ROOT:-/root/profile27b}"
ROUND="${ROUND:-1}"
START_SCRIPT="${START_SCRIPT:-/root/profile27b/tools/start_vllm.sh}"
CAPTURE_SCRIPT="${CAPTURE_SCRIPT:-/root/profile27b/tools/run_profile_capture_27b.sh}"

if [[ "$ROUND" != 1 && "$ROUND" != 2 ]]; then
    echo "ROUND must be 1 or 2" >&2
    exit 1
fi

cleanup_server() {
    if [[ -n "${server_pid:-}" ]]; then
        kill -TERM -- "-$server_pid" 2>/dev/null || true
        sleep 5
        kill -KILL -- "-$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    server_pid=""
}

trap cleanup_server EXIT

for band in 4-8K 8-16K 16-32K; do
    for quantile in p50 p90; do
        sample="$band-$quantile"
        dataset="$PROFILE_ROOT/manifests/$sample.jsonl"
        o1="$PROFILE_ROOT/captures/$sample-r$ROUND-o1"
        o65="$PROFILE_ROOT/captures/$sample-r$ROUND-o65"
        if [[ -s "$o1/result.json" && -s "$o1/trace-files.txt" \
            && -s "$o65/result.json" && -s "$o65/trace-files.txt" ]]; then
            echo "SKIP $sample round=$ROUND"
            continue
        fi

        cleanup_server
        echo "START_SERVER $sample round=$ROUND"
        mkdir -p "$PROFILE_ROOT/server-logs"
        setsid bash "$START_SCRIPT" \
            >"$PROFILE_ROOT/server-logs/$sample-r$ROUND.log" 2>&1 &
        server_pid=$!

        ready=0
        for _ in $(seq 1 72); do
            if curl --noproxy '*' -fsS http://127.0.0.1:8001/health >/dev/null 2>&1; then
                ready=1
                break
            fi
            if ! kill -0 "$server_pid" 2>/dev/null; then
                break
            fi
            sleep 5
        done
        if [[ "$ready" != 1 ]]; then
            echo "server failed to become ready for $sample round=$ROUND" >&2
            exit 1
        fi

        for output_len in 1 65; do
            label="$sample-r$ROUND-o$output_len"
            result_dir="$PROFILE_ROOT/captures/$label"
            if [[ -s "$result_dir/result.json" && -s "$result_dir/trace-files.txt" ]]; then
                echo "SKIP_CAPTURE $label"
                continue
            fi
            mkdir -p "$result_dir"
            echo "START_CAPTURE $label"
            bash "$CAPTURE_SCRIPT" "$label" "$dataset" "$output_len" \
                >"$result_dir/run.log" 2>&1
            echo "DONE_CAPTURE $label"
        done

        cleanup_server
        sleep 3
        echo "DONE_PAIR $sample round=$ROUND"
    done
done
