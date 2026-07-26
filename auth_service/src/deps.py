"""FastAPI dependencies for session + role checks."""
from __future__ import annotations

import uuid as _uuid
from typing import Optional

from fastapi import Cookie, Depends, HTTPException, status

import db
from security import SESSION_COOKIE_NAME


async def current_session(
    session_cookie: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> dict:
    if not session_cookie:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    try:
        sid = _uuid.UUID(session_cookie)
    except (ValueError, TypeError):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid session")
    row = await db.fetch_session_with_user(sid)
    if not row:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Session expired")
    return row


async def require_admin(session: dict = Depends(current_session)) -> dict:
    """Authorizes both admins and the master user. Endpoint code further
    narrows by role (only master can act on admins, etc.)."""
    if session.get("role") not in {"admin", "master"}:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin only")
    return session
