#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/root/models/Qwen3.5-27B}"
PROFILE_ROOT="${PROFILE_ROOT:-/root/profile27b}"

if [[ $# -ne 3 ]]; then
    echo "usage: $0 LABEL DATASET_JSONL OUTPUT_LEN"
    exit 1
fi

label="$1"
dataset_path="$2"
output_len="$3"
result_dir="$PROFILE_ROOT/captures/$label"
trace_dir="$PROFILE_ROOT/traces"
marker="$PROFILE_ROOT/.trace-marker-$label"

mkdir -p "$result_dir" "$trace_dir"
touch "$marker"

curl --noproxy '*' -fsS http://127.0.0.1:8001/health >/dev/null

vllm bench serve \
    --backend openai-chat \
    --host 127.0.0.1 \
    --port 8001 \
    --endpoint /v1/chat/completions \
    --model Qwen3.5-27B \
    --tokenizer "$MODEL_DIR" \
    --dataset-name custom \
    --dataset-path "$dataset_path" \
    --num-prompts 1 \
    --no-oversample \
    --max-concurrency 1 \
    --request-rate 1 \
    --temperature 0 \
    --disable-shuffle \
    --custom-output-len "$output_len" \
    --num-warmups 1 \
    --profile \
    --save-detailed \
    --extra-body '{"temperature":0.0}' \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99 \
    --save-result \
    --result-dir "$result_dir" \
    --result-filename result.json

find "$trace_dir" -maxdepth 1 -type f -newer "$marker" \
    -name '*.pt.trace.json*' -print \
    >"$result_dir/trace-files.txt"
rm -f "$marker"

if [[ ! -s "$result_dir/trace-files.txt" ]]; then
    echo "no new trace was produced for $label" >&2
    exit 1
fi

python - "$result_dir/result.json" "$output_len" <<'PY'
import json
import sys

path, expected = sys.argv[1], int(sys.argv[2])
data = json.load(open(path))
if data.get("completed") != 1 or data.get("failed") != 0:
    raise SystemExit("profile request did not complete successfully")
actual = data["output_lens"][0]
if actual != expected:
    raise SystemExit(f"expected {expected} output tokens, got {actual}")
print(f"validated output_tokens={actual}")
PY
