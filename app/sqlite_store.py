# -*- coding: utf-8 -*-
"""FastAPI 单体版 SQLite 运行存储。

目标：
- 保持与 Streamlit 版 RunStore 接近的接口，复用 studio.engine；
- 所有运行状态落入 SQLite，刷新/断连后可恢复查看；
- API Key 不落盘，只保存在当前进程内存，延续原产品的安全策略。
"""

import json
import os
import sqlite3
import tempfile
import threading
import time
from pathlib import Path

from studio.employees import EMPLOYEES
from studio.tasks import TASK_ORDER


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_DIR = (
    os.environ.get("SCRIPT_STUDIO_RUNTIME_DIR")
    or os.environ.get("RAILWAY_VOLUME_MOUNT_PATH")
    or str(PROJECT_ROOT / ".script_studio_runtime")
)
DEFAULT_DB_PATH = os.environ.get(
    "SCRIPT_STUDIO_DB",
    str(Path(DEFAULT_RUNTIME_DIR) / "fastapi_run_state.sqlite3"),
)


class SQLiteRunStore:
    """线程安全的 SQLite 运行状态存储。"""

    def __init__(self, db_path=DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.lock = threading.RLock()
        self.thread = None
        self._init_defaults()
        self._init_db()
        self._load_state()

    def _init_defaults(self):
        self.outputs = {i: None for i in TASK_ORDER}
        self.outputs[8] = {}
        self.memory = {k: [] for k in EMPLOYEES}
        self.running_task = None
        self.running_employee = None
        self.is_running = False
        self.cancel = False
        self.failed_task = None
        self.log = []
        self.manager_reviews = {}
        self.total_episodes = 50
        self.script_mode = "standard"
        self.seed = ""
        self.global_config = {"provider": "Mock (演示)", "key": "", "model": "mock-studio-model"}
        self.emp_configs = {
            k: {"provider": "Mock (演示)", "key": "", "model": "mock-studio-model"}
            for k in EMPLOYEES
        }
        self.per_emp = False
        self.emp_settings = {}
        self.task_methods = {}
        self.doc_history = []
        self.interrupted_task = None
        self.interrupted_at = ""
        self.last_saved_at = ""

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS app_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _sanitize_emp_configs(self):
        clean = {}
        for emp_key, cfg in (self.emp_configs or {}).items():
            if not isinstance(cfg, dict):
                continue
            clean[emp_key] = self._sanitize_model_config(cfg)
        return clean

    @staticmethod
    def _sanitize_model_config(cfg):
        cfg = cfg if isinstance(cfg, dict) else {}
        return {
            "provider": cfg.get("provider", "Mock (演示)"),
            "key": "",
            "model": cfg.get("model", "mock-studio-model"),
        }

    @staticmethod
    def _coerce_task_id(value):
        try:
            tid = int(value)
        except (TypeError, ValueError):
            return None
        return tid if tid in TASK_ORDER else None

    def _state_payload_locked(self):
        self.last_saved_at = time.strftime("%Y-%m-%d %H:%M:%S")
        return {
            "version": 1,
            "updated_at": self.last_saved_at,
            "outputs": {str(k): v for k, v in self.outputs.items()},
            "memory": self.memory,
            "running_task": self.running_task,
            "running_employee": self.running_employee,
            "is_running": self.is_running,
            "failed_task": self.failed_task,
            "log": self.log,
            "manager_reviews": self.manager_reviews,
            "total_episodes": self.total_episodes,
            "script_mode": self.script_mode,
            "seed": self.seed,
            "global_config": self._sanitize_model_config(self.global_config),
            "emp_configs": self._sanitize_emp_configs(),
            "per_emp": self.per_emp,
            "emp_settings": self.emp_settings,
            "task_methods": {str(k): v for k, v in (self.task_methods or {}).items()},
            "doc_history": self.doc_history,
            "interrupted_task": self.interrupted_task,
            "interrupted_at": self.interrupted_at,
        }

    def _save_state_locked(self):
        payload = self._state_payload_locked()
        raw = json.dumps(payload, ensure_ascii=False)
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix="fastapi_state_", suffix=".json.tmp", dir=str(self.db_path.parent), text=True
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(raw)
            with open(tmp_path, "r", encoding="utf-8") as f:
                checked_raw = f.read()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO app_state (id, payload, updated_at)
                    VALUES (1, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        payload = excluded.payload,
                        updated_at = excluded.updated_at
                    """,
                    (checked_raw, self.last_saved_at),
                )
                conn.commit()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def save_state(self):
        with self.lock:
            self._save_state_locked()

    def _load_state(self):
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT payload FROM app_state WHERE id = 1").fetchone()
        if not row:
            self._import_legacy_doc_history_if_empty()
            self.save_state()
            return
        try:
            data = json.loads(row[0])
        except Exception as exc:
            self.log.append(f"[{time.strftime('%H:%M:%S')}] ⚠️ 读取 SQLite 运行状态失败：{exc}")
            return

        outputs = {i: None for i in TASK_ORDER}
        outputs[8] = {}
        raw_outputs = data.get("outputs") if isinstance(data.get("outputs"), dict) else {}
        for tid in TASK_ORDER:
            outputs[tid] = raw_outputs.get(str(tid), raw_outputs.get(tid))
        if not isinstance(outputs.get(8), dict):
            outputs[8] = {}
        self.outputs = outputs

        raw_memory = data.get("memory") if isinstance(data.get("memory"), dict) else {}
        self.memory = {
            k: raw_memory.get(k, []) if isinstance(raw_memory.get(k, []), list) else []
            for k in EMPLOYEES
        }
        self.log = data.get("log", []) if isinstance(data.get("log"), list) else []
        self.manager_reviews = data.get("manager_reviews", {}) if isinstance(data.get("manager_reviews"), dict) else {}
        try:
            self.total_episodes = int(data.get("total_episodes") or self.total_episodes)
        except (TypeError, ValueError):
            self.total_episodes = 50
        self.script_mode = data.get("script_mode") or self.script_mode
        self.seed = data.get("seed") or ""
        self.emp_configs = data.get("emp_configs", {}) if isinstance(data.get("emp_configs"), dict) else {}
        self.per_emp = bool(data.get("per_emp", False))
        raw_global_config = data.get("global_config")
        if isinstance(raw_global_config, dict):
            self.global_config = self._sanitize_model_config(raw_global_config)
        elif self.per_emp:
            self.global_config = {"provider": "Mock (演示)", "key": "", "model": "mock-studio-model"}
        else:
            first_emp_key = next(iter(EMPLOYEES), "manager")
            self.global_config = self._sanitize_model_config(
                self.emp_configs.get(first_emp_key) or {}
            )
        self.emp_settings = data.get("emp_settings", {}) if isinstance(data.get("emp_settings"), dict) else {}
        raw_methods = data.get("task_methods", {}) if isinstance(data.get("task_methods"), dict) else {}
        self.task_methods = {}
        for k, v in raw_methods.items():
            try:
                self.task_methods[int(k)] = v
            except (TypeError, ValueError):
                pass
        self.doc_history = data.get("doc_history", []) if isinstance(data.get("doc_history"), list) else []
        self.failed_task = self._coerce_task_id(data.get("failed_task"))
        self.last_saved_at = data.get("updated_at", "")
        self.running_task = None
        self.running_employee = None
        self.is_running = False

        if data.get("is_running"):
            self.interrupted_task = (
                self._coerce_task_id(data.get("running_task"))
                or self._coerce_task_id(data.get("interrupted_task"))
            )
            self.interrupted_at = data.get("updated_at", "")
            self.log.append(
                f"[{time.strftime('%H:%M:%S')}] ⚠️ 检测到上次运行在任务{self.interrupted_task or '?'}中断，"
                "可点击“继续执行中断任务”恢复。"
            )
            self.save_state()
        else:
            self.interrupted_task = self._coerce_task_id(data.get("interrupted_task"))
            self.interrupted_at = data.get("interrupted_at", "")
        self._import_legacy_doc_history_if_empty()

    def _import_legacy_doc_history_if_empty(self):
        if self.doc_history:
            return
        legacy_path = Path(DEFAULT_RUNTIME_DIR) / "run_state.json"
        if not legacy_path.exists():
            return
        try:
            data = json.loads(legacy_path.read_text(encoding="utf-8"))
        except Exception:
            return
        docs = data.get("doc_history")
        if not isinstance(docs, list) or not docs:
            return
        self.doc_history = [
            doc for doc in docs
            if isinstance(doc, dict) and isinstance(doc.get("content"), str)
        ][-50:]
        if self.doc_history:
            self._save_state_locked()

    def reset(self):
        with self.lock:
            docs = list(self.doc_history)
            self.cancel = True
            self.outputs = {i: None for i in TASK_ORDER}
            self.outputs[8] = {}
            self.memory = {k: [] for k in EMPLOYEES}
            self.running_task = None
            self.running_employee = None
            self.is_running = False
            self.failed_task = None
            self.log = []
            self.manager_reviews = {}
            self.interrupted_task = None
            self.interrupted_at = ""
            self.doc_history = docs
            self._save_state_locked()

    def force_stop(self):
        with self.lock:
            self.cancel = True
            self.is_running = False
            self.running_task = None
            self.running_employee = None
            self.interrupted_task = None
            self.interrupted_at = ""
            self._save_state_locked()

    def log_line(self, msg):
        with self.lock:
            self.log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            if len(self.log) > 500:
                self.log = self.log[-500:]
            self._save_state_locked()

    def set_output(self, tid, value):
        with self.lock:
            self.outputs[tid] = value
            self._save_state_locked()

    def set_batch(self, label, value):
        with self.lock:
            if not isinstance(self.outputs.get(8), dict):
                self.outputs[8] = {}
            self.outputs[8][label] = value
            self._save_state_locked()

    def clear_output(self, tid):
        with self.lock:
            if tid == 8:
                self.outputs[8] = {}
            else:
                self.outputs[tid] = None
            prefix = f"{tid}:"
            self.manager_reviews = {
                k: v for k, v in self.manager_reviews.items()
                if k != str(tid) and not k.startswith(prefix)
            }
            self._save_state_locked()

    def add_memory(self, emp_key, note):
        with self.lock:
            mem = self.memory.setdefault(emp_key, [])
            if note not in mem:
                mem.append(note)
                self._save_state_locked()

    def add_manager_review(self, key, review):
        with self.lock:
            reviews = self.manager_reviews.setdefault(str(key), [])
            reviews.append(review)
            if len(reviews) > 10:
                self.manager_reviews[str(key)] = reviews[-10:]
            self._save_state_locked()

    def clear_manager_review(self, key):
        with self.lock:
            self.manager_reviews.pop(str(key), None)
            self._save_state_locked()

    def add_doc_history(self, title, content):
        with self.lock:
            if not content:
                return
            if self.doc_history and self.doc_history[-1]["content"] == content:
                return
            self.doc_history.append({
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "title": title or "未命名短剧",
                "content": content,
            })
            if len(self.doc_history) > 50:
                self.doc_history = self.doc_history[-50:]
            self._save_state_locked()

    def clear_doc_history(self):
        with self.lock:
            self.doc_history = []
            self._save_state_locked()

    def delete_doc_history(self, index):
        with self.lock:
            try:
                idx = int(index)
            except (TypeError, ValueError):
                return False
            if not (0 <= idx < len(self.doc_history)):
                return False
            self.doc_history.pop(idx)
            self._save_state_locked()
            return True

    def snapshot_config(self, total_episodes, script_mode, seed, emp_configs,
                        emp_settings=None, task_methods=None, per_emp=None,
                        global_config=None):
        with self.lock:
            self.total_episodes = total_episodes
            self.script_mode = script_mode
            self.seed = seed
            if global_config is not None:
                self.global_config = global_config
            self.emp_configs = emp_configs
            if per_emp is not None:
                self.per_emp = bool(per_emp)
            if emp_settings is not None:
                self.emp_settings = emp_settings
            if task_methods is not None:
                self.task_methods = task_methods
            self._save_state_locked()

    def mark_interrupted(self, task_id=None):
        with self.lock:
            self.interrupted_task = task_id or self.running_task
            self.interrupted_at = time.strftime("%Y-%m-%d %H:%M:%S")
            self.is_running = False
            self.running_task = None
            self.running_employee = None
            self._save_state_locked()

    def clear_interrupted(self):
        with self.lock:
            self.interrupted_task = None
            self.interrupted_at = ""
            self._save_state_locked()
