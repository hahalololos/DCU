#!/usr/bin/env bash
set -u
set -o pipefail

export PYTHONPATH="/public/home/acoh0h1o0p/DCU/vllm_cscc:${PYTHONPATH:-}"
MODEL_DIR="${MODEL_DIR:-/root/models/Qwen3.5-4B}"

vllm serve "$MODEL_DIR" \
    --served-model-name Qwen3.5-4B \
    --port 8001 \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.45 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
