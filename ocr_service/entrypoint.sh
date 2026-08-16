#!/bin/bash
# Detect GPU memory and pick a sensible OCR_MAX_NUM_SEQS for it.
#
# All other vLLM knobs come from .env (GPU_MEM_UTIL, MAX_MODEL_LEN,
# OCR_MAX_TOKENS, VLLM_PORT, OCR_SERVICE_PORT). Only --max-num-seqs has no
# good static default since it scales with GPU memory, so we probe.
#
# Operator wins: if OCR_MAX_NUM_SEQS is already set (.env / compose), keep it.
#
# Tier table:
#   <= 12 GiB (T4)    →  2
#   <= 18 GiB (L4)    →  4
#   <= 26 GiB (A10G)  →  8
#   <= 50 GiB (A100)  → 16
#   >  50 GiB (H100)  → 32

set -euo pipefail

GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
              | head -1 | tr -d ' ' || true)
if [ -z "$GPU_MEM_MIB" ]; then
    echo "[probe] nvidia-smi unavailable, defaulting OCR_MAX_NUM_SEQS=2" >&2
    GPU_MEM_MIB=0
fi

if   [ "$GPU_MEM_MIB" -le 12000 ]; then T_SEQ=2;  T_TIER=small
elif [ "$GPU_MEM_MIB" -le 18000 ]; then T_SEQ=4;  T_TIER=medium-small
elif [ "$GPU_MEM_MIB" -le 26000 ]; then T_SEQ=8;  T_TIER=medium
elif [ "$GPU_MEM_MIB" -le 50000 ]; then T_SEQ=16; T_TIER=large
else                                    T_SEQ=32; T_TIER=xlarge
fi

: "${OCR_MAX_NUM_SEQS:=$T_SEQ}"
export OCR_MAX_NUM_SEQS

echo "[probe] GPU=${GPU_MEM_MIB}MiB tier=$T_TIER OCR_MAX_NUM_SEQS=$OCR_MAX_NUM_SEQS"
