#!/usr/bin/env bash
set -u
set -o pipefail

MODEL_DIR="${MODEL_DIR:-/root/models/Qwen3.5-27B}"
VLLM_PROFILER_CONFIG="${VLLM_PROFILER_CONFIG:-}"

extra_args=()
if [[ -n "$VLLM_PROFILER_CONFIG" ]]; then
    extra_args+=(--profiler-config "$VLLM_PROFILER_CONFIG")
fi

vllm serve "$MODEL_DIR" \
    --served-model-name Qwen3.5-27B \
    --port 8001 \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.95 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    "${extra_args[@]}"
