"""Stop/start the OCR and translator containers so each gets the full GPU.

Uses the docker socket mounted at /var/run/docker.sock and the `docker`
Python SDK. Only operates on container names supplied via env, so user
input never reaches the docker client.

When GPU_SWAP=false every operation here becomes an awaitable no-op — used
for hosts where both services run side-by-side on a single shared GPU
(with manual MAX_MODEL_LEN / LLAMA_CTX_SIZE tuning) or on multi-GPU hosts.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Awaitable, Callable, Optional

import docker
from docker.errors import APIError, NotFound

log = logging.getLogger("orchestrator.gpu_swap")

GPU_SWAP_ENABLED = os.environ.get("GPU_SWAP", "true").strip().lower() in {
    "1", "true", "yes", "on",
}
OCR_CONTAINER_NAME = os.environ.get("OCR_CONTAINER_NAME", "ocr_service")
TRANSLATOR_CONTAINER_NAME = os.environ.get(
    "TRANSLATOR_CONTAINER_NAME", "translator_service",
)
HEALTH_TIMEOUT_SEC = int(os.environ.get("SWAP_HEALTH_TIMEOUT_SEC", "300"))

# Docker SDK is synchronous; we wrap calls in to_thread so the FastAPI event
# loop never blocks on a stop/start.
_client: Optional[docker.DockerClient] = None


def _get_client() -> docker.DockerClient:
    global _client
    if _client is None:
        _client = docker.from_env()
    return _client


EmitStatus = Callable[[str, dict], Awaitable[None]]


# ── Low-level state probes ────

def _container_state(name: str) -> dict:
    """Return {"running": bool, "health": str|None} or {"exists": False}."""
    try:
        c = _get_client().containers.get(name)
    except NotFound:
        return {"exists": False, "running": False, "health": None}
    c.reload()
    state = c.attrs.get("State") or {}
    health = (state.get("Health") or {}).get("Status")
    return {"exists": True, "running": state.get("Running", False),
            "health": health, "status": state.get("Status")}


async def state(name: str) -> dict:
    return await asyncio.to_thread(_container_state, name)


async def is_running(name: str) -> bool:
    return (await state(name)).get("running", False)


async def is_healthy(name: str) -> bool:
    s = await state(name)
    return s.get("running", False) and s.get("health") == "healthy"


# ── Lifecycle operations ────

def _stop_sync(name: str, timeout: int) -> None:
    c = _get_client().containers.get(name)
    c.stop(timeout=timeout)


def _start_sync(name: str) -> None:
    c = _get_client().containers.get(name)
    c.start()


async def stop(name: str, timeout: int = 15) -> None:
    log.info("[gpu_swap] stop  %s  (graceful timeout=%ds)", name, timeout)
    try:
        await asyncio.to_thread(_stop_sync, name, timeout)
        log.info("[gpu_swap] stop  %s  OK", name)
    except NotFound:
        log.warning("[gpu_swap] stop  %s  NOT FOUND (treating as already stopped)",
                    name)
    except APIError as e:
        if "is not running" in str(e).lower():
            log.info("[gpu_swap] stop  %s  already not running", name)
            return
        log.error("[gpu_swap] stop  %s  ERROR: %s", name, e)
        raise


async def start(name: str) -> None:
    log.info("[gpu_swap] start %s", name)
    try:
        await asyncio.to_thread(_start_sync, name)
        log.info("[gpu_swap] start %s  OK (container running, waiting for "
                 "app healthcheck)", name)
    except APIError as e:
        if "already started" in str(e).lower() or "is already" in str(e).lower():
            log.info("[gpu_swap] start %s  already running", name)
            return
        log.error("[gpu_swap] start %s  ERROR: %s", name, e)
        raise


async def wait_healthy(name: str, timeout_s: int = HEALTH_TIMEOUT_SEC,
                        emit: Optional[EmitStatus] = None) -> None:
    """Poll Inspect.State.Health.Status until 'healthy' or the deadline.

    If the container has no healthcheck, fall back to status=='running'.
    """
    deadline = asyncio.get_event_loop().time() + timeout_s
    last_emit = 0.0
    while True:
        s = await state(name)
        if not s.get("exists", True):
            raise RuntimeError(f"container {name} disappeared while waiting")
        h = s.get("health")
        if h == "healthy":
            return
        if h is None and s.get("running"):
            # No healthcheck declared — treat 'running' as ready.
            return
        if h == "unhealthy":
            raise RuntimeError(f"container {name} became unhealthy")

        now = asyncio.get_event_loop().time()
        if emit is not None and now - last_emit > 5:
            await emit("status", {
                "phase": "waiting-healthy",
                "container": name,
                "message": f"Waiting for {name} healthcheck ({h or 'starting'})",
            })
            last_emit = now

        if now >= deadline:
            raise TimeoutError(
                f"{name} did not become healthy within {timeout_s}s "
                f"(last health={h})"
            )
        await asyncio.sleep(2)


# ── High-level swaps used by the pipeline ──────

async def _emit(emit: Optional[EmitStatus], event: str, payload: dict) -> None:
    if emit is not None:
        await emit(event, payload)


async def swap_to_ocr(emit: Optional[EmitStatus] = None) -> None:
    """Ensure OCR is the active GPU tenant.

    Steps:
      1. If translator is running, stop it (free VRAM).
      2. If OCR is not running, start it.
      3. Wait for OCR healthcheck.

    No-op when GPU_SWAP=false.
    """
    if not GPU_SWAP_ENABLED:
        await _emit(emit, "status", {
            "phase": "swap-to-ocr",
            "message": "GPU_SWAP=false; assuming both services already running",
        })
        return

    tr = await state(TRANSLATOR_CONTAINER_NAME)
    if tr.get("running"):
        await _emit(emit, "status", {
            "phase": "swap-to-ocr",
            "message": f"Stopping {TRANSLATOR_CONTAINER_NAME} to free GPU VRAM",
        })
        await stop(TRANSLATOR_CONTAINER_NAME)
        # Brief pause for the GPU driver to release the context.
        await asyncio.sleep(3)

    ocr = await state(OCR_CONTAINER_NAME)
    if not ocr.get("running"):
        await _emit(emit, "status", {
            "phase": "swap-to-ocr",
            "message": f"Starting {OCR_CONTAINER_NAME}",
        })
        await start(OCR_CONTAINER_NAME)
    else:
        await _emit(emit, "status", {
            "phase": "swap-to-ocr",
            "message": f"{OCR_CONTAINER_NAME} already running",
        })

    await _emit(emit, "status", {
        "phase": "swap-to-ocr",
        "message": "Waiting for OCR healthcheck (vLLM cold-load takes 60-120s)",
    })
    await wait_healthy(OCR_CONTAINER_NAME, emit=emit)
    await _emit(emit, "status", {
        "phase": "swap-to-ocr", "message": "OCR is ready",
    })


async def swap_to_translator(emit: Optional[EmitStatus] = None) -> None:
    """Ensure translator is the active GPU tenant.

    Steps:
      1. If OCR is running, stop it.
      2. If translator is not running, start it.
      3. Wait for translator healthcheck.
    """
    if not GPU_SWAP_ENABLED:
        await _emit(emit, "status", {
            "phase": "swap-to-translator",
            "message": "GPU_SWAP=false; assuming both services already running",
        })
        return

    ocr = await state(OCR_CONTAINER_NAME)
    if ocr.get("running"):
        await _emit(emit, "status", {
            "phase": "swap-to-translator",
            "message": f"Stopping {OCR_CONTAINER_NAME} to free GPU VRAM",
        })
        await stop(OCR_CONTAINER_NAME)
        await asyncio.sleep(3)

    tr = await state(TRANSLATOR_CONTAINER_NAME)
    if not tr.get("running"):
        await _emit(emit, "status", {
            "phase": "swap-to-translator",
            "message": f"Starting {TRANSLATOR_CONTAINER_NAME}",
        })
        await start(TRANSLATOR_CONTAINER_NAME)
    else:
        await _emit(emit, "status", {
            "phase": "swap-to-translator",
            "message": f"{TRANSLATOR_CONTAINER_NAME} already running",
        })

    await _emit(emit, "status", {
        "phase": "swap-to-translator",
        "message": "Waiting for translator healthcheck (Qwen3 load takes 30-90s)",
    })
    await wait_healthy(TRANSLATOR_CONTAINER_NAME, emit=emit)
    await _emit(emit, "status", {
        "phase": "swap-to-translator", "message": "Translator is ready",
    })


# ── Read-only snapshot for /status endpoint ──────

async def snapshot() -> dict:
    ocr = await state(OCR_CONTAINER_NAME)
    tr = await state(TRANSLATOR_CONTAINER_NAME)
    return {
        "gpu_swap_enabled": GPU_SWAP_ENABLED,
        "ocr": {
            "name": OCR_CONTAINER_NAME,
            "running": ocr.get("running", False),
            "health": ocr.get("health"),
        },
        "translator": {
            "name": TRANSLATOR_CONTAINER_NAME,
            "running": tr.get("running", False),
            "health": tr.get("health"),
        },
    }


def docker_reachable() -> bool:
    """Cheap probe for /health: can we even talk to the docker socket?"""
    try:
        _get_client().ping()
        return True
    except Exception as e:
        log.warning("docker ping failed: %s", e)
        return False
