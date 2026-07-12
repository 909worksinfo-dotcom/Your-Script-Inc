# -*- coding: utf-8 -*-
"""FastAPI 单体版 Web 入口。

该入口不使用 Streamlit，复用 studio.engine / prompts / employees / tasks / llm_service。
"""

import contextvars
import hashlib
import html
import json
import re
import threading
import uuid
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Request, Form
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from studio.employees import EMPLOYEES, AGENT_SKILLS, default_skills_block
from studio.tasks import TASK_MAP, TASK_ORDER, TASK_METHODS
from studio.llm_service import PROVIDERS, MODELS, PROVIDER_LABELS
from studio.store_logic import (
    get_batches,
    is_ready,
    task8_batch_passed,
    task8_target_episodes,
    task_done,
    task_manager_review_failed,
)
from studio.engine import (
    TASK_MAX_RETRIES,
    count_episodes,
    parse_script_to_df,
    run_generic_task,
    run_pipeline,
    run_task8_batch,
    run_task9,
    validate_manual_output,
)

from .sqlite_store import DEFAULT_RUNTIME_DIR, SQLiteRunStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent

app = FastAPI(title="AI Agent 短剧剧本工作室")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
DEFAULT_STORE = SQLiteRunStore()
_current_store = contextvars.ContextVar("script_studio_store", default=DEFAULT_STORE)
_store_registry = {}
_store_registry_lock = threading.RLock()
SESSION_COOKIE = "script_studio_session"


class StoreProxy:
    def __getattr__(self, name):
        return getattr(_current_store.get(), name)

    def __setattr__(self, name, value):
        setattr(_current_store.get(), name, value)


store = StoreProxy()


def _valid_session_id(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{32}", value) is not None


def _store_for_session(session_id):
    with _store_registry_lock:
        existing = _store_registry.get(session_id)
        if existing is not None:
            return existing
        session_dir = Path(DEFAULT_RUNTIME_DIR) / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)
        session_store = SQLiteRunStore(db_path=session_dir / f"{session_id}.sqlite3")
        _store_registry[session_id] = session_store
        return session_store


@app.middleware("http")
async def bind_session_store(request: Request, call_next):
    session_id = request.cookies.get(SESSION_COOKIE)
    new_session = False
    if not _valid_session_id(session_id):
        session_id = uuid.uuid4().hex
        new_session = True
    session_store = _store_for_session(session_id)
    token = _current_store.set(session_store)
    try:
        response = await call_next(request)
    finally:
        _current_store.reset(token)
    if new_session:
        response.set_cookie(
            SESSION_COOKIE,
            session_id,
            httponly=True,
            samesite="lax",
            max_age=60 * 60 * 24 * 365,
        )
    return response


EMP_POS = {
    "manager": (51, 32),
    "researcher": (39, 55),
    "creative": (62, 55),
    "writer": (11, 84),
    "assistant": (25, 85),
    "reviewer": (41, 84),
}
EMP_SHORT = {
    "manager": "管理者",
    "researcher": "研究员",
    "creative": "创意天才",
    "writer": "编剧",
    "reviewer": "审核员",
    "assistant": "文档助理",
}
EMP_SPRITE = {
    "manager": ("#1f4ed8", "#171717"),
    "researcher": ("#2bb3c0", "#3a2a22"),
    "creative": ("#f2b600", "#5a3a1a"),
    "writer": ("#7c5cff", "#241f2e"),
    "reviewer": ("#e5654a", "#2a1a14"),
    "assistant": ("#16a34a", "#3a2a16"),
}
PIPELINE_PALETTE = {
    "done": ("#16a34a", "#eaf7ef"),
    "run": ("#5147ff", "#eeecff"),
    "ready": ("#d8a500", "#fdf6e3"),
    "idle": ("#b8b3a6", "#f6f4ee"),
}
API_FAILURE_HINT_KEYWORDS = (
    "api",
    "api key",
    "key",
    "endpoint",
    "model",
    "provider",
    "unauthorized",
    "invalid_api_key",
    "invalid api key",
    "rate limit",
    "quota",
    "401",
    "403",
    "429",
    "模型",
    "型号",
    "服务商",
    "请在侧边栏",
    "填写 API Key",
    "选择模型",
)


def _sprite_svg(shirt, hair):
    return (
        '<svg class="emp-sprite" viewBox="0 0 24 30" shape-rendering="crispEdges" '
        'xmlns="http://www.w3.org/2000/svg">'
        f'<rect x="6" y="2" width="12" height="6" fill="{hair}"/>'
        f'<rect x="5" y="4" width="2" height="7" fill="{hair}"/>'
        f'<rect x="17" y="4" width="2" height="7" fill="{hair}"/>'
        '<rect x="7" y="6" width="10" height="8" fill="#f5c9a6"/>'
        f'<rect x="7" y="6" width="10" height="2" fill="{hair}"/>'
        '<rect x="9" y="9" width="2" height="2" fill="#26303a"/>'
        '<rect x="13" y="9" width="2" height="2" fill="#26303a"/>'
        '<rect x="10" y="12" width="4" height="1" fill="#d99a78"/>'
        f'<rect x="6" y="13" width="12" height="2" fill="{shirt}"/>'
        f'<rect x="5" y="14" width="14" height="10" fill="{shirt}"/>'
        f'<rect x="3" y="15" width="2" height="7" fill="{shirt}"/>'
        f'<rect x="19" y="15" width="2" height="7" fill="{shirt}"/>'
        '<rect x="3" y="22" width="2" height="2" fill="#f5c9a6"/>'
        '<rect x="19" y="22" width="2" height="2" fill="#f5c9a6"/>'
        '<rect x="10" y="14" width="4" height="3" fill="#ffffff" opacity="0.85"/>'
        "</svg>"
    )


def _redirect(tab="studio", mode=None):
    suffix = f"&mode={mode}" if mode else ""
    return RedirectResponse(f"/?tab={tab}{suffix}", status_code=303)


def _valid_provider(value, fallback="Mock (演示)"):
    if value in PROVIDERS:
        return value
    if fallback in PROVIDERS:
        return fallback
    return PROVIDERS[0]


def _model_from_form(provider, selected, custom, rendered_provider=None, fallback_model=""):
    provider = _valid_provider(provider)
    rendered_provider = str(rendered_provider or provider)
    if rendered_provider != provider:
        return ""
    selected = str(selected or "").strip()
    custom = str(custom or "").strip()
    if selected and selected not in (MODELS.get(provider) or []):
        selected = ""
    if custom and _model_known_for_other_provider(provider, custom):
        custom = ""
    return custom or selected or fallback_model


def _model_known_for_other_provider(provider, model):
    provider = _valid_provider(provider)
    model = str(model or "").strip()
    if not model or model in (MODELS.get(provider) or []):
        return False
    return any(
        other_provider != provider and model in (options or [])
        for other_provider, options in MODELS.items()
    )


def _config_for_render(cfg):
    cfg = dict(cfg or {})
    provider = _valid_provider(cfg.get("provider"), "Mock (演示)")
    model = str(cfg.get("model") or "").strip()
    if _model_known_for_other_provider(provider, model):
        model = ""
    return {
        "provider": provider,
        "provider_label": PROVIDER_LABELS.get(provider, provider),
        "key": cfg.get("key", ""),
        "model": model,
    }


def _global_config_for_render():
    cfg = getattr(store, "global_config", None)
    if isinstance(cfg, dict):
        return _config_for_render(cfg)
    first_emp_key = next(iter(EMPLOYEES), "manager")
    return _config_for_render(
        (getattr(store, "emp_configs", {}) or {}).get(
            first_emp_key,
            {"provider": "Mock (演示)", "model": "mock-studio-model"},
        )
    )


def _safe_download_name(filename, fallback):
    name = re.sub(r"[\r\n\\/]+", "_", str(filename or "").strip())
    name = re.sub(r"_+", "_", name).strip(" ._")
    return name or fallback


def _ascii_download_name(filename, fallback):
    name = _safe_download_name(filename, fallback)
    ascii_name = name.encode("ascii", "ignore").decode("ascii")
    ascii_name = re.sub(r'[^A-Za-z0-9._-]+', "_", ascii_name).strip("._")
    if "." in fallback and "." not in ascii_name:
        return fallback
    return ascii_name or fallback


def _download_headers(filename, fallback):
    safe_name = _safe_download_name(filename, fallback)
    ascii_name = _ascii_download_name(safe_name, fallback)
    quoted_name = quote(safe_name, safe="")
    return {
        "Content-Disposition": (
            f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{quoted_name}'
        )
    }


def _thread_alive():
    return store.thread is not None and store.thread.is_alive()


def _run_guard(bound_store, target, *args, **kwargs):
    token = _current_store.set(bound_store)
    try:
        target(*args, **kwargs)
    except Exception as exc:
        store.log_line(f"❌ 后台运行异常：{exc}")
        with store.lock:
            store.failed_task = store.running_task
    finally:
        with store.lock:
            store.running_task = None
            store.running_employee = None
            store.is_running = False
        store.save_state()
        _current_store.reset(token)


def _start_background(target, *args, **kwargs):
    bound_store = _current_store.get()
    if _thread_alive():
        return False
    with store.lock:
        store.cancel = False
        store.is_running = True
        store.failed_task = None
        store.interrupted_task = None
        store.interrupted_at = ""
    store.save_state()
    t = threading.Thread(target=_run_guard, args=(bound_store, target, *args), kwargs=kwargs, daemon=True)
    store.thread = t
    t.start()
    return True


def _downstream(tid):
    result = set()
    frontier = [tid]
    while frontier:
        cur = frontier.pop()
        for task_id, meta in TASK_MAP.items():
            if cur in meta["deps"] and task_id not in result:
                result.add(task_id)
                frontier.append(task_id)
    return sorted(result)


def _invalidate_downstream(tid):
    for task_id in _downstream(tid):
        store.clear_output(task_id)


def _run_single_task(tid):
    with store.lock:
        store.running_task = tid
    if tid == 9:
        res = run_task9(store)
    else:
        res = run_generic_task(store, tid)
    if isinstance(res, str) and res.startswith("❌"):
        with store.lock:
            store.failed_task = tid
        store.save_state()
    else:
        _invalidate_downstream(tid)


def _run_task8_range(start, end):
    with store.lock:
        store.running_task = 8
    res = run_task8_batch(store, start, end)
    if isinstance(res, str) and res.startswith("❌"):
        with store.lock:
            store.failed_task = 8
        store.save_state()
    else:
        _invalidate_downstream(8)


def _task8_range_from_label(label):
    match = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*集\s*$", label or "")
    if not match:
        return None, None
    start, end = int(match.group(1)), int(match.group(2))
    return (start, end) if start <= end else (None, None)


def _run_manual_task_review(tid, content):
    with store.lock:
        store.running_task = tid
    validate_manual_output(store, tid, content)


def _run_manual_task8_review(label, content):
    start, end = _task8_range_from_label(label)
    with store.lock:
        store.running_task = 8
    validate_manual_output(store, 8, content, batch_label=label, start=start, end=end)


def _status_badge(tid):
    if store.running_task == tid:
        return "🟣 执行中"
    if task_manager_review_failed(store, tid):
        return "🔴 未通过"
    if task_done(store, tid):
        return "🟢 已完成"
    if is_ready(store, tid):
        return "🟡 可执行"
    return "⚪ 等待依赖"


def _employee_state(emp_key):
    if store.running_employee == emp_key:
        return "working"
    if store.running_task and TASK_MAP[store.running_task]["owner"] == emp_key:
        return "working"
    if emp_key == "manager":
        return "done" if any((store.manager_reviews or {}).values()) else "idle"
    owned = [tid for tid, meta in TASK_MAP.items() if meta["owner"] == emp_key]
    if owned and all(task_done(store, tid) for tid in owned):
        return "done"
    return "idle"


def _task_output_preview(tid):
    value = store.outputs.get(tid)
    if tid == 8:
        return value if isinstance(value, dict) else {}
    return value if isinstance(value, str) else ""


def _markdown_html(content):
    text = content or ""
    lines = html.escape(text).splitlines()
    out = []
    in_ul = False
    in_code = False
    code_lines = []

    def inline_markup(value):
        value = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", value)
        value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
        return value

    def close_ul():
        nonlocal in_ul
        if in_ul:
            out.append("</ul>")
            in_ul = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                out.append("<pre>" + "\n".join(code_lines) + "</pre>")
                code_lines = []
                in_code = False
            else:
                close_ul()
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not stripped:
            close_ul()
            out.append("")
            continue
        if stripped.startswith("#"):
            close_ul()
            level = min(len(stripped) - len(stripped.lstrip("#")), 6)
            body = stripped[level:].strip()
            out.append(f"<h{level}>{inline_markup(body)}</h{level}>")
            continue
        if re.match(r"^[-*]\s+", stripped):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            item_text = re.sub(r"^[-*]\s+", "", stripped)
            out.append(f"<li>{inline_markup(item_text)}</li>")
            continue
        close_ul()
        out.append(f"<p>{inline_markup(stripped)}</p>")
    close_ul()
    if in_code:
        out.append("<pre>" + "\n".join(code_lines) + "</pre>")
    return "\n".join(out)


def _storyboard_table(content):
    try:
        df = parse_script_to_df(content)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "columns": [], "rows": [], "shot_count": 0, "episode_count": 0}
    if df is None or len(df) == 0:
        return {"ok": False, "error": "", "columns": [], "rows": [], "shot_count": 0, "episode_count": 0}
    try:
        marker_mask = df["镜号"].astype(str).str.contains("🎬")
        shot_count = int((~marker_mask).sum())
        episode_count = int(marker_mask.sum())
    except Exception:
        shot_count = len(df)
        episode_count = 0
    return {
        "ok": True,
        "error": "",
        "columns": list(df.columns),
        "rows": df.fillna("").astype(str).to_dict(orient="records"),
        "shot_count": shot_count,
        "episode_count": episode_count,
    }


def _manager_review_key(tid, batch_label=None):
    return f"{tid}:{batch_label}" if batch_label else str(tid)


def _manager_reviews(tid, batch_label=None):
    reviews = (getattr(store, "manager_reviews", {}) or {}).get(
        _manager_review_key(tid, batch_label), []
    )
    return reviews if isinstance(reviews, list) else []


def _running_employee_label():
    emp_key = getattr(store, "running_employee", None)
    if emp_key in EMPLOYEES:
        emp = EMPLOYEES[emp_key]
        return f"{emp['emoji']} {emp['name']}"
    return ""


def _task_rows():
    rows = []
    for tid in TASK_ORDER:
        meta = TASK_MAP[tid]
        owner = EMPLOYEES[meta["owner"]]
        rows.append({
            "id": tid,
            "title": meta["title"],
            "short": meta["short"],
            "desc": meta["desc"],
            "owner_key": meta["owner"],
            "owner_name": owner["name"],
            "owner_emoji": owner["emoji"],
            "badge": _status_badge(tid),
            "done": task_done(store, tid),
            "ready": is_ready(store, tid),
            "output": _task_output_preview(tid),
            "output_html": _markdown_html(_task_output_preview(tid)) if tid != 8 else "",
            "outline_count": count_episodes(store.outputs.get(tid) or "") if tid in (5, 7) else None,
            "reviews": _manager_reviews(tid),
        })
    return rows


def _employee_rows():
    rows = []
    state_cn = {"working": "工作中", "done": "已完成", "idle": "待命"}
    for key, emp in EMPLOYEES.items():
        owned = [tid for tid, meta in TASK_MAP.items() if meta["owner"] == key]
        state = _employee_state(key)
        shirt, hair = EMP_SPRITE[key]
        if key == "manager":
            reviews = sum(len(v) for v in (store.manager_reviews or {}).values())
            pin_title = f"{emp['name']} · {state_cn[state]} · 已验收 {reviews} 轮"
        else:
            done_n = sum(1 for tid in owned if task_done(store, tid))
            pin_title = f"{emp['name']} · {state_cn[state]} · 任务 {done_n}/{len(owned)}"
        rows.append({
            "key": key,
            "name": emp["name"],
            "emoji": emp["emoji"],
            "title": emp["title"],
            "intro": (store.emp_settings.get(key) or {}).get("intro") or emp["intro"],
            "skills": (store.emp_settings.get(key) or {}).get("skills") or default_skills_block(key),
            "agent_skills": AGENT_SKILLS[key],
            "tool": emp["tool"],
            "state": state,
            "pos": EMP_POS[key],
            "short": EMP_SHORT[key],
            "sprite": _sprite_svg(shirt, hair),
            "pin_title": pin_title,
            "memory": store.memory.get(key, []),
            "owned_count": len(owned),
            "done_count": sum(1 for tid in owned if task_done(store, tid)),
            "config": _config_for_render(
                store.emp_configs.get(key, {"provider": "Mock (演示)", "model": "mock-studio-model"})
            ),
        })
    return rows


def _doc_rows():
    docs = []
    hist = list(store.doc_history)
    for i, doc in enumerate(reversed(hist)):
        docs.append({
            "display_index": len(hist) - i,
            "original_index": len(hist) - 1 - i,
            "title": doc.get("title", "未命名短剧"),
            "time": doc.get("time", ""),
            "content": doc.get("content", ""),
            "content_html": _markdown_html(doc.get("content", "")),
        })
    return docs


def _task8_outputs():
    scripts = store.outputs.get(8) if isinstance(store.outputs.get(8), dict) else {}
    rows = []
    for label, content in scripts.items():
        rows.append({
            "label": label,
            "content": content,
            "table": _storyboard_table(content),
            "reviews": _manager_reviews(8, label),
        })
    return rows


def _stringify_output(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(str(v) for v in value.values() if isinstance(v, str))
    return ""


def _api_failure_hint_needed():
    failed_task = store.failed_task
    if not failed_task:
        return False
    output_text = _stringify_output(store.outputs.get(failed_task))
    recent_log = "\n".join(store.log[-80:])
    text = f"{output_text}\n{recent_log}".lower()
    return any(keyword.lower() in text for keyword in API_FAILURE_HINT_KEYWORDS)


def _results_revision():
    current = _current_store.get()
    with current.lock:
        payload = {
            "outputs": current.outputs,
            "manager_reviews": current.manager_reviews,
            "running_task": current.running_task,
            "failed_task": current.failed_task,
            "is_running": current.is_running,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _render_results_panel():
    tasks = _task_rows()
    return templates.env.get_template("partials/results.html").render(
        tasks=tasks,
        task8_outputs=_task8_outputs(),
        done_count=sum(1 for tid in TASK_ORDER if task_done(store, tid)),
        store=store,
        results_revision=_results_revision(),
    )


def _runtime_pipeline_rows(tasks):
    rows = []
    for task in tasks:
        if store.running_task == task["id"]:
            status_key = "run"
        elif task["done"]:
            status_key = "done"
        elif task["ready"]:
            status_key = "ready"
        else:
            status_key = "idle"
        color, bg = PIPELINE_PALETTE[status_key]
        rows.append({
            "id": task["id"],
            "short": task["short"],
            "owner_emoji": task["owner_emoji"],
            "owner_name": task["owner_name"],
            "color": color,
            "bg": bg,
        })
    return rows


def _runtime_employee_rows():
    rows = []
    for employee in _employee_rows():
        rows.append({
            "key": employee["key"],
            "state": employee["state"],
            "pin_title": employee["pin_title"],
        })
    return rows


def _runtime_payload():
    tasks = _task_rows()
    return {
        "ok": True,
        "running": bool(store.is_running),
        "running_task": store.running_task,
        "running_employee_label": _running_employee_label(),
        "failed_task": store.failed_task,
        "api_failure_hint": _api_failure_hint_needed(),
        "done_count": sum(1 for tid in TASK_ORDER if task_done(store, tid)),
        "total_tasks": len(TASK_ORDER),
        "employee_rows": _runtime_employee_rows(),
        "pipeline_rows": _runtime_pipeline_rows(tasks),
        "log_text": "\n".join(store.log[-200:]),
        "last_saved_at": getattr(store, "last_saved_at", ""),
        "results_revision": _results_revision(),
    }


def _snapshot_view(tab, mode="auto", confirm_doc=None):
    if store.is_running and not _thread_alive():
        store.mark_interrupted(store.running_task)
        store.log_line(f"⚠️ 后台线程已结束，任务{store.interrupted_task or '?'} 已标记为可继续执行。")

    batches = get_batches(task8_target_episodes(store))
    per_emp_enabled = bool(getattr(store, "per_emp", False))
    tasks = _task_rows()
    pipeline_rows = _runtime_pipeline_rows(tasks)
    return {
        "tab": tab,
        "mode": mode,
        "store": store,
        "employees": _employee_rows(),
        "tasks": tasks,
        "pipeline_rows": pipeline_rows,
        "providers": PROVIDERS,
        "provider_labels": PROVIDER_LABELS,
        "models": MODELS,
        "global_config": _global_config_for_render(),
        "task_methods": {tid: store.task_methods.get(tid, TASK_METHODS[tid]) for tid in TASK_ORDER},
        "default_emp_skills": {key: default_skills_block(key) for key in EMPLOYEES},
        "doc_rows": _doc_rows(),
        "task8_batches": [{"start": a, "end": b, "label": f"{a}-{b}集"} for a, b in batches],
        "task8_scripts": store.outputs.get(8) if isinstance(store.outputs.get(8), dict) else {},
        "task8_outputs": _task8_outputs(),
        "done_count": sum(1 for tid in TASK_ORDER if task_done(store, tid)),
        "results_revision": _results_revision(),
        "task_max_retries": TASK_MAX_RETRIES,
        "office_url": "/studio_office.png",
        "log_text": "\n".join(store.log[-200:]),
        "per_emp_enabled": per_emp_enabled,
        "running_employee_label": _running_employee_label(),
        "api_failure_hint": _api_failure_hint_needed(),
        "confirm_doc": confirm_doc,
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request, tab: str = "studio", mode: str = "auto", confirm_doc: int = None):
    if tab not in {"studio", "settings", "docs"}:
        tab = "studio"
    if mode not in {"auto", "manual"}:
        mode = "auto"
    return templates.TemplateResponse(
        request=request,
        name="app.html",
        context=_snapshot_view(tab, mode, confirm_doc=confirm_doc),
    )


@app.get("/studio_office.png")
def studio_office():
    return FileResponse(PROJECT_ROOT / "studio_office.png")


@app.post("/config")
async def save_config(request: Request):
    if store.is_running:
        return _redirect("studio")
    form = await request.form()
    try:
        total_episodes = int(form.get("total_episodes") or store.total_episodes)
    except (TypeError, ValueError):
        total_episodes = store.total_episodes
    total_episodes = max(1, min(100, total_episodes))
    script_mode = form.get("script_mode") or "standard"
    if script_mode not in {"standard", "comic"}:
        script_mode = "standard"
    seed = form.get("seed") or ""

    old_configs = dict(store.emp_configs or {})
    old_global = getattr(store, "global_config", None)
    if not isinstance(old_global, dict):
        first_emp_key = next(iter(EMPLOYEES), "manager")
        old_global = old_configs.get(first_emp_key) or {}
    global_provider = _valid_provider(form.get("global_provider"), old_global.get("provider", "Mock (演示)"))
    global_rendered_provider = _valid_provider(
        form.get("global_rendered_provider"),
        old_global.get("provider", global_provider),
    )
    selected_model = (form.get("global_selected_model") or "").strip()
    custom_model = (form.get("global_custom_model") or "").strip()
    global_model = _model_from_form(
        global_provider,
        selected_model,
        custom_model,
        rendered_provider=global_rendered_provider,
    )
    old_global_key = old_global.get("key", "")
    global_key = str(form.get("global_key", old_global_key) or "")
    global_config = {"provider": global_provider, "key": global_key, "model": global_model}
    per_emp = form.get("per_emp") == "on"
    emp_configs = {}
    for key in EMPLOYEES:
        old = old_configs.get(key, {})
        if per_emp:
            if f"emp_{key}_provider" in form:
                provider = _valid_provider(form.get(f"emp_{key}_provider"), old.get("provider", global_provider))
                rendered_provider = _valid_provider(
                    form.get(f"emp_{key}_rendered_provider"),
                    old.get("provider", provider),
                )
                selected = (form.get(f"emp_{key}_selected_model") or "").strip()
                custom = (form.get(f"emp_{key}_custom_model") or "").strip()
                model = _model_from_form(
                    provider,
                    selected,
                    custom,
                    rendered_provider=rendered_provider,
                    fallback_model=global_model if provider == global_provider and rendered_provider == provider else "",
                )
                key_field = f"emp_{key}_key"
                api_key = str(form.get(key_field, old.get("key", "") or global_key) or "")
            else:
                provider = _valid_provider(old.get("provider"), global_provider)
                model = old.get("model") or global_model
                api_key = old.get("key") or global_key
        else:
            provider = global_provider
            model = global_model
            api_key = global_key
        emp_configs[key] = {"provider": provider, "key": api_key, "model": model}

    store.snapshot_config(
        total_episodes,
        script_mode,
        seed,
        emp_configs,
        emp_settings=store.emp_settings,
        task_methods=store.task_methods,
        per_emp=per_emp,
        global_config=global_config,
    )
    return _redirect("studio")


@app.post("/settings")
async def save_settings(request: Request):
    if store.is_running:
        return _redirect("settings")
    form = await request.form()
    emp_settings = {}
    for key, emp in EMPLOYEES.items():
        emp_settings[key] = {
            "intro": form.get(f"emp_intro_{key}") or emp["intro"],
            "skills": form.get(f"emp_skills_{key}") or default_skills_block(key),
        }
    task_methods = {
        tid: form.get(f"task_method_{tid}") or TASK_METHODS[tid]
        for tid in TASK_ORDER
    }
    store.snapshot_config(
        store.total_episodes,
        store.script_mode,
        store.seed,
        store.emp_configs,
        emp_settings=emp_settings,
        task_methods=task_methods,
    )
    return _redirect("settings")


@app.post("/settings/reset/emp/{emp_key}")
def reset_employee_setting(emp_key: str):
    if store.is_running:
        return _redirect("settings")
    if emp_key in EMPLOYEES:
        settings = dict(store.emp_settings or {})
        settings[emp_key] = {
            "intro": EMPLOYEES[emp_key]["intro"],
            "skills": default_skills_block(emp_key),
        }
        store.snapshot_config(
            store.total_episodes,
            store.script_mode,
            store.seed,
            store.emp_configs,
            emp_settings=settings,
            task_methods=store.task_methods,
        )
    return _redirect("settings")


@app.post("/settings/reset/task/{tid}")
def reset_task_method(tid: int):
    if store.is_running:
        return _redirect("settings")
    if tid in TASK_ORDER:
        methods = dict(store.task_methods or {})
        methods[tid] = TASK_METHODS[tid]
        store.snapshot_config(
            store.total_episodes,
            store.script_mode,
            store.seed,
            store.emp_configs,
            emp_settings=store.emp_settings,
            task_methods=methods,
        )
    return _redirect("settings")


@app.post("/run/all")
def run_all():
    _start_background(run_pipeline, store, from_progress=False)
    return _redirect("studio")


@app.post("/run/rest")
def run_rest():
    _start_background(run_pipeline, store, from_progress=True)
    return _redirect("studio")


@app.post("/run/task/{tid}")
def run_task(tid: int):
    if tid in TASK_ORDER and is_ready(store, tid):
        _start_background(_run_single_task, tid)
    return _redirect("studio", mode="manual")


@app.post("/run/task8/{start}/{end}")
def run_task8(start: int, end: int):
    if is_ready(store, 8):
        _start_background(_run_task8_range, start, end)
    return _redirect("studio", mode="manual")


@app.post("/run/task8/all")
def run_task8_all():
    def job():
        for start, end in get_batches(task8_target_episodes(store)):
            if store.cancel:
                break
            if task8_batch_passed(store, f"{start}-{end}集"):
                continue
            _run_task8_range(start, end)
    if is_ready(store, 8):
        _start_background(job)
    return _redirect("studio", mode="manual")


@app.post("/stop")
def stop_run():
    store.force_stop()
    return _redirect("studio")


@app.post("/reset")
def reset_run():
    store.reset()
    return _redirect("studio")


@app.post("/resume")
def resume_run():
    store.clear_interrupted()
    _start_background(run_pipeline, store, from_progress=True)
    return _redirect("studio")


@app.post("/dismiss-interrupted")
def dismiss_interrupted():
    store.clear_interrupted()
    return _redirect("studio")


@app.post("/task/{tid}/save")
async def save_task_output(tid: int, content: str = Form("")):
    if tid in TASK_ORDER and tid != 8:
        store.set_output(tid, content)
        store.clear_manager_review(str(tid))
        _invalidate_downstream(tid)
        _start_background(_run_manual_task_review, tid, content)
    return _redirect("studio", mode="manual")


@app.post("/task/{tid}/clear")
def clear_task_output(tid: int):
    if tid in TASK_ORDER:
        store.clear_output(tid)
        _invalidate_downstream(tid)
    return _redirect("studio", mode="manual")


@app.post("/task8/save")
async def save_task8_batch(label: str = Form("本地粘贴"), content: str = Form("")):
    label = (label or "本地粘贴").strip() or "本地粘贴"
    if content.strip():
        store.set_batch(label, content)
        store.clear_manager_review(f"8:{label}")
        _invalidate_downstream(8)
        _start_background(_run_manual_task8_review, label, content)
    return _redirect("studio", mode="manual")


@app.post("/task8/update")
async def update_task8_batch(label: str = Form(""), content: str = Form("")):
    label = (label or "").strip()
    if label:
        store.set_batch(label, content)
        store.clear_manager_review(f"8:{label}")
        _invalidate_downstream(8)
        _start_background(_run_manual_task8_review, label, content)
    return _redirect("studio", mode="manual")


@app.post("/task8/clear")
def clear_task8():
    store.clear_output(8)
    _invalidate_downstream(8)
    return _redirect("studio", mode="manual")


@app.post("/docs/clear")
def clear_docs():
    store.clear_doc_history()
    return _redirect("docs")


@app.post("/docs/delete/{index}")
def delete_doc(index: int):
    store.delete_doc_history(index)
    return _redirect("docs")


@app.get("/download/task/{tid}")
def download_task(tid: int):
    content = store.outputs.get(tid)
    if not isinstance(content, str):
        content = ""
    filename = f"task{tid}_{TASK_MAP.get(tid, {}).get('short', 'output')}.txt"
    return Response(
        content,
        media_type="text/plain; charset=utf-8",
        headers=_download_headers(filename, f"task{tid}_output.txt"),
    )


@app.get("/download/task8")
def download_task8(label: str):
    scripts = store.outputs.get(8) if isinstance(store.outputs.get(8), dict) else {}
    content = scripts.get(label, "")
    data = content.encode("utf-8-sig")
    try:
        df = parse_script_to_df(content)
        if df is not None and len(df) > 0:
            data = df.to_csv(index=False).encode("utf-8-sig")
    except Exception:
        pass
    return Response(
        data,
        media_type="text/csv; charset=utf-8",
        headers=_download_headers(f"storyboard_{label}.csv", "storyboard.csv"),
    )


@app.get("/download/doc/{index}")
def download_doc(index: int):
    docs = store.doc_history if isinstance(store.doc_history, list) else []
    if not (0 <= index < len(docs)):
        return PlainTextResponse("文档不存在", status_code=404)
    doc = docs[index]
    if not isinstance(doc, dict):
        return PlainTextResponse("文档不存在", status_code=404)
    title = doc.get("title") or "未命名短剧"
    content = doc.get("content", "")
    if not isinstance(content, str):
        content = str(content or "")
    return Response(
        content,
        media_type="text/markdown; charset=utf-8",
        headers=_download_headers(f"飞书剧本_{title}.md", f"feishu_script_{index + 1}.md"),
    )


@app.get("/health")
def health():
    return {"ok": True, "running": store.is_running}


@app.get("/partials/results", response_class=HTMLResponse)
def partial_results():
    return HTMLResponse(_render_results_panel())


@app.get("/api/runtime")
def api_runtime():
    return _runtime_payload()
