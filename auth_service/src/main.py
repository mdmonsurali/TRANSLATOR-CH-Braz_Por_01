"""auth_service — login, sessions, and admin user management.

Routes (all under /auth):
    GET    /auth/health
    POST   /auth/login
    POST   /auth/logout
    GET    /auth/me
    POST   /auth/change-password
    GET    /auth/validate                  (internal: ui_service middleware)
    POST   /auth/admin/users               (admin)
    GET    /auth/admin/users               (admin)
    PATCH  /auth/admin/users/{id}          (admin)
    DELETE /auth/admin/users/{id}          (admin)
"""
from __future__ import annotations

import logging
import os
import uuid as _uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Response, status

import db
from bootstrap import bootstrap_admin
from deps import current_session, require_admin
from schemas import (
    ChangePasswordRequest,
    CreateUserRequest,
    LoginRequest,
    UpdateUserRequest,
    UserOut,
    ValidateResponse,
)
from security import (
    COOKIE_SECURE,
    SESSION_COOKIE_NAME,
    SESSION_TTL,
    hash_password,
    verify_password,
)

OCR_SERVICE_URL = os.environ.get("OCR_SERVICE_URL", "http://ocr_service:8001")

log = logging.getLogger("auth_service")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_pool()
    await db.apply_schema()
    await bootstrap_admin()
    await db.cleanup_expired_sessions()
    log.info("auth_service ready")
    yield
    await db.close_pool()


app = FastAPI(title="TRANSLATOR-CH-Braz_Por Auth", version="1.0", lifespan=lifespan)


def _set_session_cookie(resp: Response, session_id: _uuid.UUID) -> None:
    resp.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=str(session_id),
        max_age=int(SESSION_TTL.total_seconds()),
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        path="/",
    )


def _clear_session_cookie(resp: Response) -> None:
    resp.delete_cookie(SESSION_COOKIE_NAME, path="/")


async def _wipe_owner_documents(owner_id: _uuid.UUID, role: str) -> None:
    """Best-effort cascade: ask ocr_service to delete every document owned by
    `owner_id`. Failure here is logged but does not block the user delete —
    the DB row going away is the source of truth, and orphan MinIO bytes
    can be reclaimed by a separate janitor."""
    headers = {"X-User-Id": str(owner_id), "X-User-Role": role}
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.delete(
                f"{OCR_SERVICE_URL}/internal/owner/{owner_id}/documents",
                headers=headers,
            )
        if resp.status_code != 200:
            log.warning("ocr_service document wipe for %s returned %d: %s",
                        owner_id, resp.status_code, resp.text[:200])
        else:
            log.info("ocr_service wiped %s documents for %s",
                     resp.json().get("deleted"), owner_id)
    except httpx.HTTPError as e:
        log.warning("ocr_service document wipe for %s failed: %s", owner_id, e)


def _user_out(row: dict) -> dict:
    return {
        "id": str(row["id"]),
        "username": row["username"],
        "role": row["role"],
        "must_change_password": row["must_change_password"],
        "disabled": row["disabled"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


# ── Public ───────────────────────────────────────────────────────────────

@app.get("/auth/health")
async def health():
    try:
        await db.health()
    except Exception as e:
        raise HTTPException(503, f"postgres not reachable: {e}")
    return {"status": "ok"}


@app.post("/auth/login")
async def login(body: LoginRequest, response: Response):
    user = await db.get_user_by_username(body.username)
    if not user or user.get("disabled"):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
    if not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")

    sess = await db.create_session(user["id"], SESSION_TTL)
    _set_session_cookie(response, sess["session_id"])
    return {
        "user_id": str(user["id"]),
        "username": user["username"],
        "role": user["role"],
        "must_change_password": user["must_change_password"],
    }


@app.post("/auth/logout")
async def logout(response: Response, session=Depends(current_session)):
    await db.delete_session(session["session_id"])
    _clear_session_cookie(response)
    return {"status": "ok"}


@app.get("/auth/me", response_model=ValidateResponse)
async def me(session=Depends(current_session)):
    await db.touch_session(session["session_id"])
    return ValidateResponse(
        user_id=str(session["user_id"]),
        username=session["username"],
        role=session["role"],
        must_change_password=session["must_change_password"],
    )


@app.get("/auth/validate", response_model=ValidateResponse)
async def validate(session=Depends(current_session)):
    """Internal: ui_service middleware calls this on every request."""
    await db.touch_session(session["session_id"])
    return ValidateResponse(
        user_id=str(session["user_id"]),
        username=session["username"],
        role=session["role"],
        must_change_password=session["must_change_password"],
    )


@app.delete("/auth/me", status_code=204)
async def delete_me(response: Response, session=Depends(current_session)):
    """User deletes their own account. Also wipes their OCR history."""
    uid = session["user_id"]
    role = session["role"]
    # Master is the un-deletable super-admin.
    if role == "master":
        raise HTTPException(400, "The master account cannot be deleted")
    # Guardrail: last admin can't self-delete (matches admin console rule).
    if role == "admin":
        admins = await db.count_admins()
        if admins <= 1:
            raise HTTPException(400, "Cannot delete the last admin account")
    await _wipe_owner_documents(uid, role)
    # delete_user_sessions then delete_user (sessions cascade on user delete,
    # but we drop the cookie explicitly first so the response always clears it)
    await db.delete_user_sessions(uid)
    ok = await db.delete_user(uid)
    if not ok:
        raise HTTPException(404, "User not found")
    _clear_session_cookie(response)
    return Response(status_code=204)


@app.post("/auth/change-password")
async def change_password(body: ChangePasswordRequest, session=Depends(current_session)):
    # Need the hash, which fetch_session_with_user doesn't return — fetch by id.
    user = await db.get_user_by_username(session["username"])
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User missing")
    if not verify_password(body.current_password, user["password_hash"]):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current password incorrect")
    if body.current_password == body.new_password:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "New password must differ from current")
    await db.update_user(
        user["id"],
        password_hash=hash_password(body.new_password),
        must_change_password=False,
    )
    # Revoke all other sessions for this user.
    await db.delete_user_sessions(user["id"], except_session=session["session_id"])
    return {"status": "ok"}


# ── Admin ────────────────────────────────────────────────────────────────
#
# Roles in this system:
#   master  — un-deletable super-admin. Created once by bootstrap. Can see and
#             act on every account except other 'master' rows (only one
#             should ever exist, but we still guard against the case).
#   admin   — created only by master. Can see/manage 'user' accounts only.
#             Cannot see other admins or the master, cannot delete them, and
#             cannot self-promote.
#   user    — regular account.

_USER_ONLY     = {"user"}
_USER_OR_ADMIN = {"user", "admin"}


def _visible_roles_for(caller_role: str) -> set[str]:
    """Which target roles can the caller see / act on?"""
    if caller_role == "master":
        return _USER_OR_ADMIN     # master sees users + admins
    if caller_role == "admin":
        return _USER_ONLY         # admin sees only users
    return set()


@app.post("/auth/admin/users", response_model=UserOut, status_code=201)
async def admin_create_user(body: CreateUserRequest,
                            session=Depends(require_admin)):
    # Only master may mint new admins. Admins can mint users only.
    if body.role == "admin" and session["role"] != "master":
        raise HTTPException(403, "Only the master can create admin accounts")
    existing = await db.get_user_by_username(body.username)
    if existing:
        raise HTTPException(409, "Username already taken")
    created = await db.create_user(
        username=body.username,
        password_hash=hash_password(body.password),
        role=body.role,
        must_change_password=True,
    )
    return _user_out(created)


@app.get("/auth/admin/users", response_model=list[UserOut])
async def admin_list_users(session=Depends(require_admin)):
    visible = _visible_roles_for(session["role"])
    rows = await db.list_users()
    return [_user_out(r) for r in rows if r["role"] in visible]


def _parse_uuid(raw: str) -> _uuid.UUID:
    try:
        return _uuid.UUID(raw)
    except (ValueError, TypeError):
        raise HTTPException(400, "Invalid user id (must be UUID)")


@app.patch("/auth/admin/users/{user_id}", response_model=UserOut)
async def admin_update_user(user_id: str, body: UpdateUserRequest,
                            session=Depends(require_admin)):
    uid = _parse_uuid(user_id)
    target = await db.get_user_by_id(uid)
    if not target or target["role"] not in _visible_roles_for(session["role"]):
        # 404 (not 403) so admins can't probe for admin/master existence.
        raise HTTPException(404, "User not found")

    updates: dict = {}
    if body.username is not None and body.username != target["username"]:
        clash = await db.get_user_by_username(body.username)
        if clash and clash["id"] != uid:
            raise HTTPException(409, "Username already taken")
        updates["username"] = body.username
    if body.password is not None:
        updates["password_hash"] = hash_password(body.password)
        updates["must_change_password"] = True
    if body.role is not None and body.role != target["role"]:
        # Only master can change roles (promote user→admin or demote admin→user).
        if session["role"] != "master":
            raise HTTPException(403, "Only the master can change roles")
        # Schemas already restrict to user|admin so master can't be assigned.
        if target["role"] == "admin" and body.role != "admin":
            admins = await db.count_admins()
            if admins <= 1:
                raise HTTPException(400, "Cannot demote the last admin")
        updates["role"] = body.role
    if body.disabled is not None:
        # Block disabling the last admin (master can never be disabled here
        # because admins can't see master and master can't see itself in list).
        if body.disabled and target["role"] == "admin":
            admins = await db.count_admins()
            if admins <= 1:
                raise HTTPException(400, "Cannot disable the last admin")
        updates["disabled"] = body.disabled

    if not updates:
        return _user_out(target)

    updated = await db.update_user(uid, **updates)
    # If we rotated password or disabled, revoke their sessions.
    if "password_hash" in updates or updates.get("disabled"):
        await db.delete_user_sessions(uid)
    return _user_out(updated)


@app.delete("/auth/admin/users/{user_id}", status_code=204)
async def admin_delete_user(user_id: str, session=Depends(require_admin)):
    uid = _parse_uuid(user_id)
    if uid == session["user_id"]:
        raise HTTPException(400, "Cannot delete your own account")
    target = await db.get_user_by_id(uid)
    if not target or target["role"] not in _visible_roles_for(session["role"]):
        # Hides master from admins, and prevents normal admins from
        # deleting other admins. Master sees admins, can delete them.
        raise HTTPException(404, "User not found")
    # Belt-and-braces: master is never deletable by anyone via this endpoint.
    # (Already filtered above because no role's visible set includes 'master'.)
    if target["role"] == "master":
        raise HTTPException(400, "The master account cannot be deleted")
    if target["role"] == "admin":
        admins = await db.count_admins()
        if admins <= 1:
            raise HTTPException(400, "Cannot delete the last admin")
    # Cascade: wipe the target's OCR history (DB + MinIO) before removing
    # the user row, so the documents query has a valid owner reference.
    await _wipe_owner_documents(uid, target["role"])
    ok = await db.delete_user(uid)
    if not ok:
        raise HTTPException(404, "User not found")
    return Response(status_code=204)
