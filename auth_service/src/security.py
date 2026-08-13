"""Password hashing + cookie/session constants."""
from __future__ import annotations

import os
from datetime import timedelta

from passlib.context import CryptContext

SESSION_COOKIE_NAME = "translator_ch_braz_por_01_session"
SESSION_TTL = timedelta(hours=int(os.getenv("SESSION_TTL_HOURS", "12")))
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").strip().lower() in {"1", "true", "yes", "on"}

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)


def hash_password(plain: str) -> str:
    return _pwd.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _pwd.verify(plain, hashed)
    except Exception:
        return False
