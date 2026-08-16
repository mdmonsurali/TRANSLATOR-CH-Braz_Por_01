# translator_service
# ── Stage 1: pull llama-server binary from upstream's official CUDA build ──
FROM ghcr.io/ggml-org/llama.cpp:server-cuda AS llama-src

RUN test -x /app/llama-server || test -x /usr/local/bin/llama-server || \
    (echo "llama-server not found at expected paths" && ls -la /app /usr/local/bin && exit 1)


FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        curl ca-certificates libcurl4 libgomp1 libgcc-s1 \
        fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python
ENV PIP_BREAK_SYSTEM_PACKAGES=1

COPY --from=llama-src /app/ /opt/llama/
RUN ln -sf /opt/llama/llama-server /usr/local/bin/llama-server \
    && echo "/opt/llama" > /etc/ld.so.conf.d/llama.conf \
    && ldconfig

WORKDIR /workspace

COPY translator_service/requirements.txt /workspace/translator_service/
RUN pip3 install --no-cache-dir -r /workspace/translator_service/requirements.txt

# Download the GGUF at build time so first-boot is fast. Override at build
# with --build-arg TRANSLATOR_MODEL_REPO=... TRANSLATOR_MODEL_FILE=...
ARG TRANSLATOR_MODEL_REPO=Qwen/Qwen3-14B-GGUF
ARG TRANSLATOR_MODEL_FILE=Qwen3-14B-Q8_0.gguf

ENV TRANSLATOR_MODEL_PATH=/models/translator.gguf
ENV TRANSLATOR_MODEL_REPO=${TRANSLATOR_MODEL_REPO}
ENV TRANSLATOR_MODEL_FILE=${TRANSLATOR_MODEL_FILE}

RUN pip3 install --no-cache-dir "huggingface_hub[hf_transfer]==0.35.3" \
    && mkdir -p /models \
    && HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$TRANSLATOR_MODEL_REPO" "$TRANSLATOR_MODEL_FILE" \
         --local-dir /models \
    && ln -sf "/models/$TRANSLATOR_MODEL_FILE" /models/translator.gguf

COPY translator_service/src/ /workspace/translator_service/src/

COPY reconstruction_service/src/translation_reconstruction /workspace/translation_reconstruction

# Shared by BOTH reconstruction copies (ocr_service ships it too), so the
# content-vs-bbox table geometry rules cannot drift between them. Lands at
# /workspace/table_geometry.py, which PYTHONPATH already covers below.
COPY reconstruction_service/src/table_geometry.py /workspace/table_geometry.py

# Database schema bundled into the image; applied at startup (idempotent).
COPY database/schema.sql /workspace/database/schema.sql

ENV PYTHONPATH=/workspace:/workspace/translator_service/src:$PYTHONPATH
ENV TRANSLATOR_SERVICE_PORT=8003
ENV LLAMA_PORT=8090
ENV LLAMA_HOST=127.0.0.1
ENV LLAMA_GPU_LAYERS=999
ENV LLAMA_CTX_SIZE=8192
ENV LLAMA_PARALLEL=2
ENV TRANSLATE_TARGET_LANG=pt-BR
ENV TRANSLATE_CONCURRENCY=2
ENV TRANSLATE_CHUNK_ITEMS=20
ENV TRANSLATE_CHUNK_BYTES=4000
ENV TRANSLATE_MAX_TOKENS=4096

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=180s \
    CMD curl -f http://localhost:8003/health || exit 1

WORKDIR /workspace/translator_service/src

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["\
    set -euo pipefail; \
    : \"${TRANSLATOR_MODEL_PATH:=/models/translator.gguf}\"; \
    if [ ! -f \"$TRANSLATOR_MODEL_PATH\" ]; then \
        echo \"[translator] model not found at $TRANSLATOR_MODEL_PATH\" >&2; \
        exit 1; \
    fi; \
    echo \"[translator] starting llama-server on ${LLAMA_HOST}:${LLAMA_PORT} (model=$TRANSLATOR_MODEL_PATH)\"; \
    llama-server \
        --model \"$TRANSLATOR_MODEL_PATH\" \
        --host \"$LLAMA_HOST\" \
        --port \"$LLAMA_PORT\" \
        --n-gpu-layers \"$LLAMA_GPU_LAYERS\" \
        --ctx-size \"$LLAMA_CTX_SIZE\" \
        --parallel \"$LLAMA_PARALLEL\" \
        --no-warmup \
        --jinja \
        --reasoning-budget 0 \
        --log-disable & \
    LLAMA_PID=$!; \
    echo \"[translator] waiting for llama-server (PID $LLAMA_PID)...\"; \
    for i in $(seq 1 600); do \
        if ! kill -0 \"$LLAMA_PID\" 2>/dev/null; then \
            echo \"[translator] llama-server died before becoming ready\" >&2; \
            exit 1; \
        fi; \
        if curl -sf \"http://${LLAMA_HOST}:${LLAMA_PORT}/health\" >/dev/null 2>&1; then \
            echo \"[translator] llama-server ready after ${i}s\"; \
            break; \
        fi; \
        sleep 1; \
    done; \
    if ! curl -sf \"http://${LLAMA_HOST}:${LLAMA_PORT}/health\" >/dev/null 2>&1; then \
        echo \"[translator] llama-server did not become ready within 600s\" >&2; \
        exit 1; \
    fi; \
    exec uvicorn main:app --host 0.0.0.0 --port \"$TRANSLATOR_SERVICE_PORT\""]
