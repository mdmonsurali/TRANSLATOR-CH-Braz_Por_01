FROM vllm/vllm-openai:v0.17.0

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    curl \
    procps \
    libreoffice \
    poppler-utils \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    imagemagick \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# ImageMagick's default policy on Debian/Ubuntu blocks PDF reads (Ghostscript
# CVE mitigation). Our formula renderer generates the PDF itself via pdflatex
# and immediately converts it to PNG — there's no untrusted PDF input — so
# this is safe to re-enable. Without it the matplotlib mathtext fallback
# fires, which can't parse `\frac{\text{...}}{...}` or empty args.
RUN sed -i 's|<policy domain="coder" rights="none" pattern="PDF"|<policy domain="coder" rights="read\|write" pattern="PDF"|g' \
    /etc/ImageMagick-6/policy.xml 2>/dev/null || true

COPY ocr_service/requirements.txt /workspace/ocr_service/
RUN pip3 install --no-cache-dir -r /workspace/ocr_service/requirements.txt

# PaddleOCR (diagram text-label detection) is NOT installed here. It lives in
# its own image/service — see paddle_service/Dockerfile and the `paddle_service`
# compose service — because its paddlepaddle/paddlex stack floats
# numpy/opencv/pillow against vLLM's pins. `diagram_text_recovery` reaches it
# over HTTP at PADDLE_SERVICE_URL.

# Pre-bake the Chandra OCR 2 weights into the image so cold start does not
# depend on a runtime HuggingFace download. Pass --build-arg HF_TOKEN=... if
# the repo is gated. Weights land in the standard HF cache so `vllm serve
# datalab-to/chandra-ocr-2` resolves them offline.
ARG HF_TOKEN=""
ARG CHANDRA_MODEL=datalab-to/chandra-ocr-2
ENV HF_HOME=/opt/hf-cache
RUN pip3 install --no-cache-dir "huggingface_hub[cli]" && \
    HF_TOKEN="${HF_TOKEN}" huggingface-cli download "${CHANDRA_MODEL}" \
        --exclude "*.pth" "original/*" && \
    echo "chandra weights cached under ${HF_HOME}"

COPY ocr_service/src/ /workspace/ocr_service/src/
COPY ocr_service/entrypoint.sh /workspace/ocr_service/entrypoint.sh
RUN chmod +x /workspace/ocr_service/entrypoint.sh

# ocr_service's own copy of the reconstruction renderer modules. The
# translator_service ships an independent copy under translation_reconstruction
# so the two services can diverge without affecting each other.
COPY reconstruction_service/src/ocr_reconstruction /workspace/ocr_reconstruction

# Shared by BOTH reconstruction copies (translator_service ships it too), so the
# content-vs-bbox table geometry rules cannot drift between them. Lands at
# /workspace/table_geometry.py, which PYTHONPATH already covers below.
COPY reconstruction_service/src/table_geometry.py /workspace/table_geometry.py

# Database schema bundled into the image; applied at startup.
COPY database/schema.sql /workspace/database/schema.sql

ENV PYTHONPATH=/workspace:/workspace/ocr_service/src:$PYTHONPATH
ENV OCR_SERVICE_PORT=8001
ENV VLLM_PORT=8888
ENV VLLM_SERVICE_URL=http://localhost:8888
# Served-model name vLLM advertises; must match OCR_MODEL_NAME the client uses.
ENV OCR_MODEL_NAME=chandra
ENV CHANDRA_MODEL=datalab-to/chandra-ocr-2

EXPOSE 8888 8001

HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=180s \
    CMD curl -f http://localhost:8001/health || exit 1

WORKDIR /workspace/ocr_service/src

# entrypoint.sh probes the GPU and exports OCR_MAX_NUM_SEQS for the right tier.
# All other knobs (GPU_MEM_UTIL, MAX_MODEL_LEN, OCR_MAX_TOKENS, VLLM_PORT,
# OCR_SERVICE_PORT) come from .env via docker-compose.
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["\
     source /workspace/ocr_service/entrypoint.sh; \
     vllm serve $CHANDRA_MODEL \
         --tensor-parallel-size 1 \
         --gpu-memory-utilization $GPU_MEM_UTIL \
         --max-model-len $MAX_MODEL_LEN \
         --max-num-seqs $OCR_MAX_NUM_SEQS \
         --no-enforce-eager \
         --served-model-name $OCR_MODEL_NAME \
         --port $VLLM_PORT & \
     VLLM_PID=$!; \
     echo \"Waiting for vLLM (PID $VLLM_PID) to become ready on :$VLLM_PORT...\"; \
     for i in $(seq 1 300); do \
         if ! kill -0 $VLLM_PID 2>/dev/null; then \
             echo \"vLLM process died before becoming ready\" >&2; \
             exit 1; \
         fi; \
         if curl -sf http://localhost:$VLLM_PORT/health >/dev/null 2>&1; then \
             echo \"vLLM ready after ${i}s\"; \
             break; \
         fi; \
         sleep 1; \
     done; \
     if ! curl -sf http://localhost:$VLLM_PORT/health >/dev/null 2>&1; then \
         echo \"vLLM did not become ready within 300s\" >&2; \
         exit 1; \
     fi; \
     exec uvicorn main:app --host 0.0.0.0 --port $OCR_SERVICE_PORT"]
