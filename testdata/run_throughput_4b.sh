#!/usr/bin/env bash
set -u
set -o pipefail

export PYTHONPATH="/public/home/xdzs2026_c203/haha/vllm_cscc:${PYTHONPATH:-}"
MODEL_DIR="${MODEL_DIR:-/public/home/xdzs2026_c203/models/Qwen3.5-4B}"

DATASET="${1:-all}"
NUM_PROMPTS="${2:-}"

run_one() {
    local label="$1"
    local dataset_path="$2"
    local result_dir="$3"
    local n_prompts="${NUM_PROMPTS:-$(wc -l < "$dataset_path")}"

    mkdir -p "$result_dir"
    echo "===== throughput: $label ====="

    vllm bench serve \
        --backend openai-chat \
        --host 127.0.0.1 \
        --port 8001 \
        --endpoint /v1/chat/completions \
        --model Qwen3.5-4B \
        --tokenizer "$MODEL_DIR" \
        --dataset-name custom \
        --dataset-path "$dataset_path" \
        --num-prompts "$n_prompts" \
        --no-oversample \
        --max-concurrency 1 \
        --request-rate 1 \
        --temperature 0 \
        --disable-shuffle \
        --custom-output-len 1024 \
        --num-warmups 2 \
        --save-detailed \
        --extra-body '{"temperature":0.0}' \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,95,99 \
        --save-result \
        --result-dir "$result_dir" \
        --result-filename result.json
}

case "$DATASET" in
    all)
        run_one "4-8K" "./4-8K_throughput.jsonl" "./test_4b/4-8K_throughput"
        run_one "8-16K" "./8-16K_throughput.jsonl" "./test_4b/8-16K_throughput"
        run_one "16-32K" "./16-32K_throughput.jsonl" "./test_4b/16-32K_throughput"
        ;;
    4-8K)
        run_one "4-8K" "./4-8K_throughput.jsonl" "./test_4b/4-8K_throughput"
        ;;
    8-16K)
        run_one "8-16K" "./8-16K_throughput.jsonl" "./test_4b/8-16K_throughput"
        ;;
    16-32K)
        run_one "16-32K" "./16-32K_throughput.jsonl" "./test_4b/16-32K_throughput"
        ;;
    *)
        echo "usage: $0 [all|4-8K|8-16K|16-32K] [num_prompts]"
        exit 1
        ;;
esac

echo
echo "===== result files ====="
find ./test_4b -name result.json -type f -print
