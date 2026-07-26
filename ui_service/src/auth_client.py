"""Thin httpx wrapper around auth_service, with a tiny in-process cache
on /auth/validate so we don't hit auth_service for every static asset."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import httpx

log = logging.getLogger("ui_service.auth_client")

AUTH_SERVICE_URL = os.environ.get("AUTH_SERVICE_URL", "http://auth_service:8002")
SESSION_COOKIE_NAME = "translator_ch_braz_por_session"
_VALIDATE_CACHE_TTL = 30.0  # seconds


@dataclass(frozen=True)
class CurrentUser:
    id: str
    username: str
    role: str
    must_change_password: bool


# session_id -> (expiry_epoch, CurrentUser)
_cache: dict[str, tuple[float, CurrentUser]] = {}


def _cache_get(sid: str) -> Optional[CurrentUser]:
    item = _cache.get(sid)
    if not item:
        return None
    expiry, user = item
    if expiry < time.monotonic():
        _cache.pop(sid, None)
        return None
    return user


def _cache_put(sid: str, user: CurrentUser) -> None:
    _cache[sid] = (time.monotonic() + _VALIDATE_CACHE_TTL, user)


def _cache_drop(sid: str) -> None:
    _cache.pop(sid, None)


async def validate(session_id: str) -> Optional[CurrentUser]:
    """Resolve a session_id to a CurrentUser, or None if invalid/expired."""
    if not session_id:
        return None
    hit = _cache_get(session_id)
    if hit is not None:
        return hit
    cookies = {SESSION_COOKIE_NAME: session_id}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{AUTH_SERVICE_URL}/auth/validate", cookies=cookies)
    except httpx.HTTPError as e:
        log.warning("auth validate failed: %s", e)
        return None
    if resp.status_code != 200:
        return None
    data = resp.json()
    user = CurrentUser(
        id=data["user_id"],
        username=data["username"],
        role=data["role"],
        must_change_password=bool(data.get("must_change_password")),
    )
    _cache_put(session_id, user)
    return user


async def login(username: str, password: str) -> tuple[Optional[str], dict]:
    """Returns (session_cookie_value, body) on success; (None, body) on failure."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{AUTH_SERVICE_URL}/auth/login",
            json={"username": username, "password": password},
        )
    body = {}
    try:
        body = resp.json()
    except Exception:
        body = {"detail": resp.text}
    if resp.status_code != 200:
        return None, body
    cookie = resp.cookies.get(SESSION_COOKIE_NAME)
    return cookie, body


async def logout(session_id: str) -> None:
    _cache_drop(session_id)
    if not session_id:
        return
    cookies = {SESSION_COOKIE_NAME: session_id}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(f"{AUTH_SERVICE_URL}/auth/logout", cookies=cookies)
    except httpx.HTTPError as e:
        log.warning("auth logout failed: %s", e)


async def delete_me(session_id: str) -> tuple[int, dict]:
    """User self-delete. Returns (status_code, body). On success the
    auth_service has already cleared the cookie and wiped the user's
    documents; the ui_service still re-clears the cookie locally."""
    _cache_drop(session_id)
    cookies = {SESSION_COOKIE_NAME: session_id}
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.delete(f"{AUTH_SERVICE_URL}/auth/me", cookies=cookies)
    if resp.status_code == 204:
        return 204, {}
    try:
        body = resp.json()
    except Exception:
        body = {"detail": resp.text}
    return resp.status_code, body


async def change_password(session_id: str, current: str, new: str) -> tuple[int, dict]:
    _cache_drop(session_id)
    cookies = {SESSION_COOKIE_NAME: session_id}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{AUTH_SERVICE_URL}/auth/change-password",
            cookies=cookies,
            json={"current_password": current, "new_password": new},
        )
    try:
        body = resp.json()
    except Exception:
        body = {"detail": resp.text}
    return resp.status_code, body


# ── Admin proxies ────────────────────────────────────────────────────────

async def admin_list_users(session_id: str) -> tuple[int, list]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{AUTH_SERVICE_URL}/auth/admin/users",
            cookies={SESSION_COOKIE_NAME: session_id},
        )
    try:
        body = resp.json()
    except Exception:
        body = []
    return resp.status_code, body


async def admin_create_user(session_id: str, username: str, password: str,
                            role: str) -> tuple[int, dict]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{AUTH_SERVICE_URL}/auth/admin/users",
            cookies={SESSION_COOKIE_NAME: session_id},
            json={"username": username, "password": password, "role": role},
        )
    try:
        body = resp.json()
    except Exception:
        body = {"detail": resp.text}
    return resp.status_code, body


async def admin_update_user(session_id: str, user_id: str,
                            payload: dict) -> tuple[int, dict]:
    _drop_target_sessions_cache()
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.patch(
            f"{AUTH_SERVICE_URL}/auth/admin/users/{user_id}",
            cookies={SESSION_COOKIE_NAME: session_id},
            json=payload,
        )
    try:
        body = resp.json()
    except Exception:
        body = {"detail": resp.text}
    return resp.status_code, body


async def admin_delete_user(session_id: str, user_id: str) -> tuple[int, dict]:
    _drop_target_sessions_cache()
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.delete(
            f"{AUTH_SERVICE_URL}/auth/admin/users/{user_id}",
            cookies={SESSION_COOKIE_NAME: session_id},
        )
    if resp.status_code == 204:
        return 204, {}
    try:
        body = resp.json()
    except Exception:
        body = {"detail": resp.text}
    return resp.status_code, body


def _drop_target_sessions_cache() -> None:
    """Any admin write may have revoked some sessions on the auth side;
    blow the whole cache so the next request re-validates."""
    _cache.clear()
