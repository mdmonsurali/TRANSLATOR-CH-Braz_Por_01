#!/usr/bin/env python3
"""
main.py — FastAPI UI service entry point.

Serves the HTML frontend, gates all OCR routes behind a session cookie, and
proxies authenticated requests to ocr_service with X-User-Id / X-User-Role
headers so ocr_service can scope every document to its owner.

    GET  /login                      login page
    POST /login                      establish session
    POST /logout                     drop session
    GET  /change-password            forced when must_change_password
    POST /change-password            update password
    GET  /me                         current user (for the UI chip)
    GET  /admin                      admin console (admin only)
    GET  /admin/users                list users (admin only)
    POST /admin/users                create user (admin only)
    PATCH  /admin/users/{id}         rename / reset password / role / disable
    DELETE /admin/users/{id}         delete

    POST /ocr                        single-file upload
    POST /ocr/batch                  SSE multi-file upload
    POST /ocr/batch-zip              ZIP bundle download
    GET  /documents                  list rows (owner-scoped)
    GET  /documents/{id}             one row metadata
    GET  /documents/{id}/{kind}      stream source|md|json|docx
    GET  /preview/{id}               DOCX -> PDF preview
    POST /preview-input              ad-hoc DOCX upload -> PDF preview
"""

import logging
import os
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote

import httpx
import uvicorn
from fastapi import Body, FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles

import auth_client
from middleware import AuthMiddleware

_HERE = Path(__file__).parent.resolve()
TEMPLATES_DIR = _HERE / "templates"

OCR_SERVICE_URL = os.environ.get("OCR_SERVICE_URL", "http://ocr_service:8001")
TRANSLATOR_SERVICE_URL = os.environ.get(
    "TRANSLATOR_SERVICE_URL", "http://translator_service:8003",
)
ORCHESTRATOR_SERVICE_URL = os.environ.get(
    "ORCHESTRATOR_SERVICE_URL", "http://orchestrator_service:8004",
)

app = FastAPI(title="TRANSLATOR-CH-Braz_Por UI", docs_url=None, redoc_url=None)
app.add_middleware(AuthMiddleware)


class _SilencePathsFilter(logging.Filter):
    """Drop uvicorn access lines for the noisiest endpoints: the mode-pill's
    /translator/status poll (every 7s per open tab) and /health probes.
    Real events still log normally."""
    def __init__(self, paths: tuple[str, ...]) -> None:
        super().__init__()
        self._paths = paths

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(f' {p} ' in msg for p in self._paths)


logging.getLogger("uvicorn.access").addFilter(
    _SilencePathsFilter(("/health", "/translator/status"))
)
app.mount("/static", StaticFiles(directory=str(TEMPLATES_DIR)), name="static")


# ── Template rendering ──────────────────────────────────────────────────

def _render(name: str, *, user: Optional[auth_client.CurrentUser] = None) -> str:
    """Load a static HTML file and substitute the @@PROFILE_MENU@@ sentinel
    so every page gets the same Profile dropdown without a templating engine."""
    text = (TEMPLATES_DIR / name).read_text(encoding="utf-8")
    if user is None:
        return text
    # Reusable inline SVGs (stroke=currentColor so CSS controls the colour).
    robot_svg = (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" '
        'aria-hidden="true">'
        '<rect x="4" y="8" width="16" height="11" rx="2.5"/>'
        '<path d="M12 5v3"/><circle cx="12" cy="4" r="1.2"/>'
        '<circle cx="9" cy="13" r="1.2"/><circle cx="15" cy="13" r="1.2"/>'
        '<path d="M9 17h6"/>'
        '<path d="M4 12H2.5"/><path d="M21.5 12H20"/>'
        '</svg>'
    )
    history_svg = (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" '
        'aria-hidden="true">'
        '<path d="M3 12a9 9 0 1 0 3-6.7"/><polyline points="3 4 3 9 8 9"/>'
        '<path d="M12 7v5l3 2"/>'
        '</svg>'
    )
    settings_svg = (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" '
        'aria-hidden="true">'
        '<circle cx="12" cy="12" r="3"/>'
        '<path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9c.3.4.8.7 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>'
        '</svg>'
    )
    admin_svg = (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" '
        'aria-hidden="true">'
        '<path d="M12 3l8 3v6c0 5-3.5 8.5-8 9-4.5-.5-8-4-8-9V6l8-3z"/>'
        '<path d="M9 12l2 2 4-4"/>'
        '</svg>'
    )
    logout_svg = (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" '
        'aria-hidden="true">'
        '<path d="M15 17l5-5-5-5"/><path d="M20 12H9"/>'
        '<path d="M12 21H6a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h6"/>'
        '</svg>'
    )

    admin_item = (
        f'<a class="profile-item" href="/admin">'
        f'<span class="profile-ico">{admin_svg}</span>'
        f'<span class="profile-item-label">Administration</span></a>'
        if user.role in ("admin", "master") else ""
    )
    menu = f"""
<div class="profile-wrap" id="profile-wrap">
  <button type="button" class="profile-btn" id="profile-btn"
          aria-haspopup="true" aria-expanded="false" aria-label="Profile menu">
    <span class="profile-avatar">{robot_svg}</span>
  </button>
  <div class="profile-menu" id="profile-menu" role="menu" hidden>
    <a class="profile-head" href="/settings">
      <span class="profile-head-avatar">{robot_svg}</span>
      <span class="profile-head-text">
        <span class="profile-head-name">{_e(user.username)}</span>
        <span class="profile-head-role">{_e(user.role)} account</span>
      </span>
    </a>
    <div class="profile-divider"></div>
    {admin_item}
    <a class="profile-item" href="/history">
      <span class="profile-ico">{history_svg}</span>
      <span class="profile-item-label">Task History</span>
    </a>
    <a class="profile-item" href="/settings">
      <span class="profile-ico">{settings_svg}</span>
      <span class="profile-item-label">Settings</span>
    </a>
    <div class="profile-divider"></div>
    <button type="button" class="profile-item profile-logout" id="profile-logout">
      <span class="profile-ico">{logout_svg}</span>
      <span class="profile-item-label">Log out</span>
    </button>
  </div>
</div>
<script>
(function() {{
  const wrap = document.getElementById('profile-wrap');
  const btn  = document.getElementById('profile-btn');
  const menu = document.getElementById('profile-menu');
  const logoutBtn = document.getElementById('profile-logout');
  if (!wrap || !btn || !menu) return;
  function close() {{ menu.hidden = true; btn.setAttribute('aria-expanded','false'); }}
  function toggle() {{
    const open = menu.hidden;
    menu.hidden = !open;
    btn.setAttribute('aria-expanded', String(open));
  }}
  btn.addEventListener('click', (e) => {{ e.stopPropagation(); toggle(); }});
  document.addEventListener('click', (e) => {{
    if (!wrap.contains(e.target)) close();
  }});
  document.addEventListener('keydown', (e) => {{ if (e.key === 'Escape') close(); }});
  if (logoutBtn) {{
    logoutBtn.addEventListener('click', async (e) => {{
      e.preventDefault();
      try {{ await fetch('/logout', {{ method: 'POST', credentials: 'same-origin' }}); }} catch (_) {{}}
      // Prevent back-button from re-showing the previous authed page
      window.location.replace('/login');
    }});
  }}
}})();

/* ── Centered modal helpers (window.uiConfirm / uiPrompt / uiAlert) ── */
(function() {{
  function escapeHtml(s) {{
    return String(s ?? "").replace(/[&<>"']/g, c => ({{
      "&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"
    }}[c]));
  }}
  function buildModal({{ title, body, html, input, inputType, inputPlaceholder,
                        okLabel, cancelLabel, danger, hideCancel }}) {{
    return new Promise((resolve) => {{
      const backdrop = document.createElement('div');
      backdrop.className = 'ui-modal-backdrop';
      backdrop.setAttribute('role', 'dialog');
      backdrop.setAttribute('aria-modal', 'true');
      const inputHtml = input
        ? `<input class="ui-modal-input" id="ui-modal-input" type="${{escapeHtml(inputType || 'text')}}" placeholder="${{escapeHtml(inputPlaceholder || '')}}" />`
        : '';
      const cancelHtml = hideCancel
        ? ''
        : `<button type="button" class="ui-modal-btn ui-modal-btn-cancel" id="ui-modal-cancel">${{escapeHtml(cancelLabel || 'Cancel')}}</button>`;
      const okCls = danger ? 'ui-modal-btn-danger' : 'ui-modal-btn-ok';
      const bodyHtml = html ? body : escapeHtml(body || '');
      backdrop.innerHTML = `
        <div class="ui-modal-card" role="document">
          ${{title ? `<h2 class="ui-modal-title">${{escapeHtml(title)}}</h2>` : ''}}
          ${{bodyHtml ? `<div class="ui-modal-body">${{bodyHtml}}</div>` : ''}}
          ${{inputHtml}}
          <div class="ui-modal-actions">
            ${{cancelHtml}}
            <button type="button" class="ui-modal-btn ${{okCls}}" id="ui-modal-ok">${{escapeHtml(okLabel || 'OK')}}</button>
          </div>
        </div>`;
      document.body.appendChild(backdrop);
      const inputEl  = backdrop.querySelector('#ui-modal-input');
      const okBtn    = backdrop.querySelector('#ui-modal-ok');
      const cancelBtn = backdrop.querySelector('#ui-modal-cancel');
      function close(result) {{
        document.removeEventListener('keydown', onKey, true);
        backdrop.remove();
        resolve(result);
      }}
      function onKey(e) {{
        if (e.key === 'Escape') {{ e.preventDefault(); close(hideCancel ? true : null); }}
        else if (e.key === 'Enter' && (!inputEl || document.activeElement === inputEl || document.activeElement === okBtn)) {{
          e.preventDefault();
          close(input ? (inputEl.value) : true);
        }}
      }}
      document.addEventListener('keydown', onKey, true);
      backdrop.addEventListener('click', (e) => {{
        if (e.target === backdrop) close(hideCancel ? true : null);
      }});
      okBtn.addEventListener('click', () => close(input ? inputEl.value : true));
      if (cancelBtn) cancelBtn.addEventListener('click', () => close(null));
      setTimeout(() => {{ (inputEl || okBtn).focus(); }}, 30);
    }});
  }}
  window.uiConfirm = (opts) => buildModal({{
    title: opts.title || 'Are you sure?',
    body:  opts.body  || '',
    html:  !!opts.html,
    okLabel: opts.okLabel || 'OK',
    cancelLabel: opts.cancelLabel || 'Cancel',
    danger: !!opts.danger,
  }}).then(v => v === true);
  window.uiPrompt = (opts) => buildModal({{
    title: opts.title || '',
    body:  opts.body  || '',
    html:  !!opts.html,
    input: true,
    inputType: opts.inputType || 'text',
    inputPlaceholder: opts.placeholder || '',
    okLabel: opts.okLabel || 'OK',
    cancelLabel: opts.cancelLabel || 'Cancel',
    danger: !!opts.danger,
  }}).then(v => (v === null ? null : String(v)));
  window.uiAlert = (opts) => buildModal({{
    title: opts.title || '',
    body:  opts.body  || '',
    html:  !!opts.html,
    okLabel: opts.okLabel || 'OK',
    hideCancel: true,
    danger: !!opts.danger,
  }}).then(() => undefined);
}})();
</script>
"""
    return text.replace("<!-- @@PROFILE_MENU@@ -->", menu)


def _e(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;"))


def _require_user(request: Request) -> auth_client.CurrentUser:
    user = getattr(request.state, "user", None)
    if user is None:
        # Should be unreachable because middleware would have redirected,
        # but keep the guard for routes that get called programmatically.
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def _require_admin(request: Request) -> auth_client.CurrentUser:
    user = _require_user(request)
    if user.role not in ("admin", "master"):
        raise HTTPException(status_code=403, detail="Admin only")
    return user


def _identity_headers(user: auth_client.CurrentUser) -> dict:
    return {"X-User-Id": user.id, "X-User-Role": user.role}


def _safe_next(raw: Optional[str]) -> str:
    if not raw:
        return "/"
    if not raw.startswith("/") or raw.startswith("//"):
        return "/"
    return raw


# ── Auth pages ──────────────────────────────────────────────────────────

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return _render("login.html")


@app.post("/login")
async def login_submit(
    username: str = Form(...),
    password: str = Form(...),
    next: Optional[str] = Form(None),
):
    cookie, body = await auth_client.login(username, password)
    if cookie is None:
        detail = body.get("detail") if isinstance(body, dict) else "Login failed"
        return RedirectResponse(
            url=f"/login?err={quote(str(detail))}",
            status_code=303,
        )
    redirect_to = _safe_next(next)
    if body.get("must_change_password"):
        redirect_to = "/change-password"
    resp = RedirectResponse(url=redirect_to, status_code=303)
    resp.set_cookie(
        key=auth_client.SESSION_COOKIE_NAME,
        value=cookie,
        httponly=True,
        secure=False,  # mirror auth_service COOKIE_SECURE default; nginx terminates TLS in prod
        samesite="lax",
        path="/",
    )
    return resp


@app.post("/logout")
async def logout(request: Request):
    sid = request.cookies.get(auth_client.SESSION_COOKIE_NAME)
    if sid:
        await auth_client.logout(sid)
    resp = JSONResponse({"status": "ok"})
    resp.delete_cookie(auth_client.SESSION_COOKIE_NAME, path="/")
    return resp


@app.post("/account/delete")
async def delete_account(request: Request):
    user = _require_user(request)
    sid = request.cookies.get(auth_client.SESSION_COOKIE_NAME) or ""
    code, body = await auth_client.delete_me(sid)
    if code != 204:
        return JSONResponse(content=body, status_code=code)
    resp = JSONResponse({"status": "ok", "username": user.username})
    resp.delete_cookie(auth_client.SESSION_COOKIE_NAME, path="/")
    return resp


@app.get("/change-password", response_class=HTMLResponse)
async def change_password_page(request: Request):
    return _render("change-password.html")


@app.post("/change-password")
async def change_password_submit(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
):
    if new_password != confirm_password:
        return RedirectResponse(
            "/change-password?err=" + quote("Passwords do not match"),
            status_code=303,
        )
    if len(new_password) < 8:
        return RedirectResponse(
            "/change-password?err=" + quote("Password must be at least 8 characters"),
            status_code=303,
        )
    sid = request.cookies.get(auth_client.SESSION_COOKIE_NAME) or ""
    status_code, body = await auth_client.change_password(sid, current_password, new_password)
    if status_code != 200:
        detail = body.get("detail") if isinstance(body, dict) else "Update failed"
        return RedirectResponse(
            "/change-password?err=" + quote(str(detail)),
            status_code=303,
        )
    return RedirectResponse("/", status_code=303)


@app.get("/me")
async def me(request: Request):
    user = _require_user(request)
    return {"id": user.id, "username": user.username, "role": user.role,
            "must_change_password": user.must_change_password}


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    user = _require_user(request)
    return _render("history.html", user=user)


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    user = _require_user(request)
    return _render("settings.html", user=user)


# ── Admin console ───────────────────────────────────────────────────────

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    user = _require_admin(request)
    return _render("admin.html", user=user)


@app.get("/admin/users")
async def admin_users_list(request: Request):
    _require_admin(request)
    sid = request.cookies.get(auth_client.SESSION_COOKIE_NAME) or ""
    code, body = await auth_client.admin_list_users(sid)
    return JSONResponse(content=body, status_code=code)


@app.post("/admin/users")
async def admin_users_create(request: Request, payload: dict = Body(...)):
    _require_admin(request)
    sid = request.cookies.get(auth_client.SESSION_COOKIE_NAME) or ""
    code, body = await auth_client.admin_create_user(
        sid,
        username=payload.get("username", ""),
        password=payload.get("password", ""),
        role=payload.get("role", "user"),
    )
    return JSONResponse(content=body, status_code=code)


@app.patch("/admin/users/{user_id}")
async def admin_users_update(request: Request, user_id: str, payload: dict = Body(...)):
    _require_admin(request)
    sid = request.cookies.get(auth_client.SESSION_COOKIE_NAME) or ""
    code, body = await auth_client.admin_update_user(sid, user_id, payload)
    return JSONResponse(content=body, status_code=code)


@app.delete("/admin/users/{user_id}")
async def admin_users_delete(request: Request, user_id: str):
    _require_admin(request)
    sid = request.cookies.get(auth_client.SESSION_COOKIE_NAME) or ""
    code, body = await auth_client.admin_delete_user(sid, user_id)
    if code == 204:
        return Response(status_code=204)
    return JSONResponse(content=body, status_code=code)


# ── Main UI pages ───────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    user = _require_user(request)
    return _render("index.html", user=user)


@app.get("/batch", response_class=HTMLResponse)
async def batch_page(request: Request):
    user = _require_user(request)
    return _render("batch.html", user=user)


@app.get("/translator", response_class=HTMLResponse)
async def translator_page(request: Request):
    user = _require_user(request)
    return _render("translator.html", user=user)


# ── OCR proxies — every call carries identity headers ───────────────────

@app.post("/ocr")
async def ocr(request: Request, file: UploadFile = File(...)):
    user = _require_user(request)
    content = await file.read()
    async with httpx.AsyncClient(timeout=600) as client:
        try:
            resp = await client.post(
                f"{OCR_SERVICE_URL}/ocr",
                files={"file": (file.filename, content, file.content_type)},
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="OCR service is unavailable")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


_ALLOWED_KINDS = {"source", "md", "json", "docx"}


@app.get("/documents/{doc_id}/images/{filename}")
async def document_image(request: Request, doc_id: str, filename: str):
    user = _require_user(request)
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.get(
                f"{OCR_SERVICE_URL}/documents/{doc_id}/images/{filename}",
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="OCR service is unavailable")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return StreamingResponse(
        iter([resp.content]),
        media_type=resp.headers.get("content-type", "image/png"),
    )


@app.get("/documents/{doc_id}/{kind}")
async def document_artifact(request: Request, doc_id: str, kind: str):
    user = _require_user(request)
    if kind not in _ALLOWED_KINDS:
        raise HTTPException(status_code=400, detail=f"Unknown kind: {kind}")
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            resp = await client.get(
                f"{OCR_SERVICE_URL}/documents/{doc_id}/{kind}",
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="OCR service is unavailable")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    content_type = resp.headers.get("content-type", "application/octet-stream")
    disposition  = resp.headers.get("content-disposition")
    headers = {}
    if disposition:
        headers["Content-Disposition"] = disposition
    return StreamingResponse(
        iter([resp.content]),
        media_type=content_type,
        headers=headers,
    )


@app.get("/preview/{doc_id}")
async def preview(request: Request, doc_id: str):
    user = _require_user(request)
    async with httpx.AsyncClient(timeout=360) as client:
        try:
            resp = await client.get(
                f"{OCR_SERVICE_URL}/preview/{doc_id}",
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="OCR service is unavailable")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return StreamingResponse(
        iter([resp.content]),
        media_type="application/pdf",
    )


@app.get("/documents")
async def list_documents(request: Request, limit: int = 50, offset: int = 0,
                         status: str | None = None):
    user = _require_user(request)
    params = {"limit": limit, "offset": offset}
    if status:
        params["status"] = status
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(
                f"{OCR_SERVICE_URL}/documents",
                params=params,
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="OCR service is unavailable")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    headers = {}
    total = resp.headers.get("x-total-count")
    if total is not None:
        headers["X-Total-Count"] = total
    return JSONResponse(content=resp.json(), headers=headers)


@app.delete("/documents/{doc_id}")
async def delete_document(request: Request, doc_id: str):
    user = _require_user(request)
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.delete(
                f"{OCR_SERVICE_URL}/documents/{doc_id}",
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="OCR service is unavailable")
    if resp.status_code == 204:
        return Response(status_code=204)
    detail = resp.text
    try:
        detail = resp.json().get("detail", detail)
    except Exception:
        pass
    raise HTTPException(status_code=resp.status_code, detail=detail)


@app.get("/documents/{doc_id}")
async def get_document(request: Request, doc_id: str):
    user = _require_user(request)
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(
                f"{OCR_SERVICE_URL}/documents/{doc_id}",
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="OCR service is unavailable")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


@app.post("/ocr/batch")
async def ocr_batch(request: Request, files: List[UploadFile] = File(...)):
    user = _require_user(request)
    file_tuples = []
    for f in files:
        content = await f.read()
        file_tuples.append(("files", (f.filename, content, f.content_type)))

    async def proxy():
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{OCR_SERVICE_URL}/ocr/batch",
                    files=file_tuples,
                    headers=_identity_headers(user),
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        yield (
                            "event: error\n"
                            f"data: {{\"status\": {resp.status_code}, "
                            f"\"detail\": {body.decode('utf-8', 'replace')!r}}}\n\n"
                        ).encode("utf-8")
                        return
                    async for chunk in resp.aiter_raw():
                        if chunk:
                            yield chunk
            except httpx.ConnectError:
                yield (b"event: error\n"
                       b"data: {\"detail\": \"OCR service is unavailable\"}\n\n")

    return StreamingResponse(
        proxy(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/ocr/batch-zip")
async def ocr_batch_zip(request: Request, payload: dict = Body(...)):
    user = _require_user(request)
    async with httpx.AsyncClient(timeout=600) as client:
        try:
            resp = await client.post(
                f"{OCR_SERVICE_URL}/ocr/batch-zip",
                json=payload,
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="OCR service is unavailable")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return StreamingResponse(
        iter([resp.content]),
        media_type="application/zip",
        headers={
            "Content-Disposition":
                'attachment; filename="OCR_results.zip"',
        },
    )


@app.post("/preview-input")
async def preview_input(request: Request, file: UploadFile = File(...)):
    user = _require_user(request)
    content = await file.read()
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            resp = await client.post(
                f"{OCR_SERVICE_URL}/preview-input",
                files={"file": (file.filename, content, file.content_type)},
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="OCR service is unavailable")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return StreamingResponse(
        iter([resp.content]),
        media_type="application/pdf",
    )


# ── Translator proxies — every call carries identity headers ───────────

_TRANSLATION_KINDS = {"translated_json", "translated_docx"}


@app.post("/translator/batch")
async def translator_batch(request: Request,
                            files: List[UploadFile] = File(...)):
    user = _require_user(request)
    file_tuples = []
    for f in files:
        content = await f.read()
        file_tuples.append(("files", (f.filename, content, f.content_type)))

    async def proxy():
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{TRANSLATOR_SERVICE_URL}/translate/batch",
                    files=file_tuples,
                    headers=_identity_headers(user),
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        yield (
                            "event: error\n"
                            f"data: {{\"status\": {resp.status_code}, "
                            f"\"detail\": {body.decode('utf-8', 'replace')!r}}}\n\n"
                        ).encode("utf-8")
                        return
                    async for chunk in resp.aiter_raw():
                        if chunk:
                            yield chunk
            except httpx.ConnectError:
                yield (b"event: error\n"
                       b"data: {\"detail\": \"Translator service is unavailable\"}\n\n")

    return StreamingResponse(
        proxy(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/translator-documents")
async def translator_documents(request: Request, limit: int = 200, offset: int = 0):
    """List OCR'd documents available for translation. Served by
    translator_service (reads postgres directly), so this works even when
    ocr_service is stopped in swap mode."""
    user = _require_user(request)
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(
                f"{TRANSLATOR_SERVICE_URL}/translator-documents",
                params={"limit": limit, "offset": offset},
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail=("Translator service is unavailable. "
                        "Start a translation from the Translator page "
                        "(orchestrator manages container lifecycle)."),
            )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    headers = {}
    total = resp.headers.get("x-total-count")
    if total is not None:
        headers["X-Total-Count"] = total
    return JSONResponse(content=resp.json(), headers=headers)


@app.post("/translations-by-source")
async def translations_by_source(request: Request, payload: dict = Body(...)):
    """Look up translations for a list of OCR documents. Returns a compact
    map keyed by source_document_id. Used by Task History to show
    translation-download buttons next to documents that have one.
    Proxied to orchestrator (translator may be stopped between batches)."""
    user = _require_user(request)
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(
                f"{ORCHESTRATOR_SERVICE_URL}/translations-by-source",
                json=payload,
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            # History page should still load even if orchestrator is down;
            # absence of translation buttons is fine.
            return JSONResponse(content={})
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return JSONResponse(content=resp.json())


@app.post("/translator/from-history")
async def translator_from_history(request: Request, payload: dict = Body(...)):
    """Forward a 'translate these existing OCR docs' request to translator
    via SSE. Used by the Translator UI in swap mode where ocr_service is
    stopped and we work from previously OCR'd documents in history."""
    user = _require_user(request)

    async def proxy():
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{TRANSLATOR_SERVICE_URL}/translate/from-history",
                    json=payload,
                    headers=_identity_headers(user),
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        yield (
                            "event: error\n"
                            f"data: {{\"status\": {resp.status_code}, "
                            f"\"detail\": {body.decode('utf-8', 'replace')!r}}}\n\n"
                        ).encode("utf-8")
                        return
                    async for chunk in resp.aiter_raw():
                        if chunk:
                            yield chunk
            except httpx.ConnectError:
                yield (b"event: error\n"
                       b"data: {\"detail\": \"Translator service is unavailable.\"}\n\n")

    return StreamingResponse(
        proxy(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/translator/run")
async def translator_run(request: Request,
                          files: List[UploadFile] = File(...),
                          target_lang: Optional[str] = Form(default=None)):
    """SSE proxy to orchestrator_service /run/batch.

    Forwards the multipart upload and identity headers so orchestrator
    can drive the full upload → OCR → swap GPU → translate pipeline.

    Implementation note: httpx 0.28 tightened its check that requires the
    request body iterator to be async when using AsyncClient. Passing
    files=...+data=... to client.stream() builds a sync multipart encoder
    and the call fails with "Attempted to send a sync request with an
    AsyncClient instance." We work around it by pre-encoding the multipart
    body via httpx.Request() then materializing it into a single bytes
    buffer that we pass via content=, which httpx is happy to send.
    """
    user = _require_user(request)

    # Drain UploadFile streams now — they will be closed once the SSE response
    # starts so we can't lazily read them inside the generator below.
    file_tuples: list = []
    for f in files:
        content = await f.read()
        file_tuples.append((
            "files",
            (f.filename, content, f.content_type or "application/octet-stream"),
        ))
    data_fields: dict = {}
    if target_lang:
        data_fields["target_lang"] = target_lang

    # Build a one-shot Request to get httpx to do the multipart encoding for
    # us, then flatten its body into raw bytes. This bypasses the sync/async
    # stream check entirely.
    encoded_req = httpx.Request(
        "POST",
        f"{ORCHESTRATOR_SERVICE_URL}/run/batch",
        files=file_tuples,
        data=data_fields or None,
        headers=_identity_headers(user),
    )
    body_bytes = encoded_req.read()
    content_type = encoded_req.headers.get("content-type", "")
    content_length = encoded_req.headers.get("content-length")

    async def proxy():
        proxy_headers = _identity_headers(user)
        proxy_headers["content-type"] = content_type
        if content_length is not None:
            proxy_headers["content-length"] = content_length
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{ORCHESTRATOR_SERVICE_URL}/run/batch",
                    content=body_bytes,
                    headers=proxy_headers,
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        yield (
                            "event: error\n"
                            f"data: {{\"status\": {resp.status_code}, "
                            f"\"detail\": {body.decode('utf-8', 'replace')!r}}}\n\n"
                        ).encode("utf-8")
                        return
                    async for chunk in resp.aiter_raw():
                        if chunk:
                            yield chunk
            except httpx.ConnectError:
                yield (b"event: error\n"
                       b"data: {\"detail\": \"Orchestrator service is "
                       b"unavailable.\"}\n\n")

    return StreamingResponse(
        proxy(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/translator/status")
async def translator_status(request: Request):
    """Snapshot of OCR + translator container state, current pipeline phase,
    and whether a batch is in flight. Used by the Translator UI to show a
    live mode indicator."""
    user = _require_user(request)
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(
                f"{ORCHESTRATOR_SERVICE_URL}/status",
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503,
                                detail="Orchestrator service is unavailable")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


@app.post("/translator/batch-zip")
async def translator_batch_zip(request: Request, payload: dict = Body(...)):
    # Proxied to orchestrator_service (not translator_service) so the bundle
    # still downloads after the GPU swap stopped the translator container.
    user = _require_user(request)
    async with httpx.AsyncClient(timeout=600) as client:
        try:
            resp = await client.post(
                f"{ORCHESTRATOR_SERVICE_URL}/translate/batch-zip",
                json=payload,
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503,
                                detail="Orchestrator service is unavailable")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return StreamingResponse(
        iter([resp.content]),
        media_type="application/zip",
        headers={
            "Content-Disposition":
                'attachment; filename="Translate_result.zip"',
        },
    )


@app.get("/translations")
async def list_translations(request: Request, limit: int = 50, offset: int = 0,
                            status: str | None = None):
    user = _require_user(request)
    params = {"limit": limit, "offset": offset}
    if status:
        params["status"] = status
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(
                f"{TRANSLATOR_SERVICE_URL}/translations",
                params=params,
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503,
                                detail="Translator service is unavailable")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    headers = {}
    total = resp.headers.get("x-total-count")
    if total is not None:
        headers["X-Total-Count"] = total
    return JSONResponse(content=resp.json(), headers=headers)


@app.get("/translations/{trans_id}")
async def get_translation(request: Request, trans_id: str):
    user = _require_user(request)
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(
                f"{TRANSLATOR_SERVICE_URL}/translations/{trans_id}",
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503,
                                detail="Translator service is unavailable")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


# NOTE: declaration order matters. FastAPI matches routes in registration
# order; the literal "/preview" path must be declared BEFORE the more
# general "/{kind}" path or every preview request gets captured by {kind}
# and rejected with 400 "Unknown kind: preview".

@app.get("/translations/{trans_id}/preview")
async def preview_translation(request: Request, trans_id: str):
    """DOCX → PDF preview. Orchestrator owns the round-trip (fetch DOCX from
    MinIO, forward to ocr_service /preview-input). Works while translator is
    stopped."""
    user = _require_user(request)
    async with httpx.AsyncClient(timeout=600) as client:
        try:
            pdf_resp = await client.get(
                f"{ORCHESTRATOR_SERVICE_URL}/translations/{trans_id}/preview",
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503,
                                detail="Orchestrator service is unavailable")
    if pdf_resp.status_code != 200:
        raise HTTPException(status_code=pdf_resp.status_code, detail=pdf_resp.text)
    return StreamingResponse(
        iter([pdf_resp.content]),
        media_type="application/pdf",
    )


@app.get("/translations/{trans_id}/{kind}")
async def get_translation_artifact(request: Request, trans_id: str, kind: str):
    """Stream a translated artifact (DOCX or JSON). Proxied to
    orchestrator_service, which reads from MinIO directly so this works
    even when translator_service is stopped between batches."""
    user = _require_user(request)
    if kind not in _TRANSLATION_KINDS:
        raise HTTPException(status_code=400, detail=f"Unknown kind: {kind}")
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            resp = await client.get(
                f"{ORCHESTRATOR_SERVICE_URL}/translations/{trans_id}/{kind}",
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503,
                                detail="Orchestrator service is unavailable")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    content_type = resp.headers.get("content-type", "application/octet-stream")
    disposition = resp.headers.get("content-disposition")
    headers = {}
    if disposition:
        headers["Content-Disposition"] = disposition
    return StreamingResponse(
        iter([resp.content]),
        media_type=content_type,
        headers=headers,
    )


@app.delete("/translations/{trans_id}")
async def delete_translation(request: Request, trans_id: str):
    user = _require_user(request)
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.delete(
                f"{TRANSLATOR_SERVICE_URL}/translations/{trans_id}",
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503,
                                detail="Translator service is unavailable")
    if resp.status_code == 204:
        return Response(status_code=204)
    raise HTTPException(status_code=resp.status_code, detail=resp.text)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
