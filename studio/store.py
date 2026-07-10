# -*- coding: utf-8 -*-
"""会话无关的全局运行存储（单例）+ 基于它的纯逻辑工具。

为什么需要它（修复线上稳定性）：
- Streamlit 的脚本运行与浏览器会话/WebSocket 绑定。电脑休眠/长时间不操作时连接会断开，
  Streamlit 会中断当前脚本运行 → "一键运行全流程"被打断、跑不到最后。
- st.session_state 是「按会话隔离」的，浏览器重连会新建会话 → 之前的产出/过程输出丢失。

解决方案：
- 用 @st.cache_resource 提供一个「跨会话、跨重连、跨刷新」都存在的单例 RunStore，
  把产出 / 进度 / 过程日志都存在这里（而不是 session_state）。
- 长流程在「后台线程」里跑（见 engine.run_pipeline），只读写本 RunStore，不触碰 st.*，
  因此浏览器断连不会中断它；重连后 UI 从 RunStore 读取，进度与过程输出都不丢失
  （除非用户主动点「重置工作室」）。
"""

import json
import os
import re
import tempfile
import time
import threading

import streamlit as st

from .tasks import TASK_MAP, TASK_ORDER
from .employees import EMPLOYEES
from .llm_service import LLMService

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_PKG_DIR)
_RUNTIME_DIR = os.path.join(_PROJECT_ROOT, ".script_studio_runtime")
_RUN_STATE_FILE = os.path.join(_RUNTIME_DIR, "run_state.json")


class RunStore:
    """单例运行存储。所有写操作均加锁，可被后台线程与前台脚本安全并发访问。"""

    def __init__(self):
        self.lock = threading.RLock()
        self.outputs = {i: None for i in TASK_ORDER}
        self.outputs[8] = {}
        self.memory = {k: [] for k in EMPLOYEES}
        self.running_task = None       # 当前正在执行的任务号（用于办公室/流水线高亮）
        self.running_employee = None   # 当前真正工作的数字员工（生成者或工作室管理者验收）
        self.is_running = False        # 是否有后台流水线正在跑
        self.cancel = False            # 取消标志（点重置时置位，后台线程尽快停止并丢弃在途写入）
        self.failed_task = None        # 最近失败的任务号（用于"任务失败"提示）
        self.log = []                  # 过程输出日志（逐行，持久化、不随刷新消失）
        self.manager_reviews = {}      # {"任务号" / "任务号:批次": [验收记录]}
        self.thread = None
        # 运行参数快照（启动一次运行时从 session_state 拷入；后台线程只读它，不访问 session_state）
        self.total_episodes = 50
        self.script_mode = "standard"
        self.seed = ""
        self.emp_configs = {}          # {emp_key: {"provider","key","model"}}
        # 用户自定义的「数字员工设定」与「任务协作方式」（执行时实际生效；留空则用 PRD 默认）
        self.emp_settings = {}         # {emp_key: {"intro": str, "skills": str}}
        self.task_methods = {}         # {tid: str}
        # 任务9最终飞书交付文档的历史归档（当前 + 过去），跨刷新/重连存活
        self.doc_history = []          # [{"time","title","content"}]
        self.interrupted_task = None   # 上次进程/线程中断时正在执行的任务号，用于重连后继续
        self.interrupted_at = ""
        self.last_saved_at = ""
        self._load_state()

    # ---------- 持久化（线程安全调用方需已持锁，或使用 save_state） ----------
    def _sanitize_emp_configs(self):
        """运行状态落盘时不保存 API Key，避免把密钥明文写入本地文件。"""
        clean = {}
        for emp_key, cfg in (self.emp_configs or {}).items():
            if not isinstance(cfg, dict):
                continue
            clean[emp_key] = {
                "provider": cfg.get("provider", "Mock (演示)"),
                "key": "",
                "model": cfg.get("model", "mock-studio-model"),
            }
        return clean

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
            "emp_configs": self._sanitize_emp_configs(),
            "emp_settings": self.emp_settings,
            "task_methods": {str(k): v for k, v in (self.task_methods or {}).items()},
            "doc_history": self.doc_history,
            "interrupted_task": self.interrupted_task,
            "interrupted_at": self.interrupted_at,
        }

    def _save_state_locked(self):
        os.makedirs(_RUNTIME_DIR, exist_ok=True)
        payload = self._state_payload_locked()
        fd, tmp_path = tempfile.mkstemp(
            prefix="run_state_", suffix=".json.tmp", dir=_RUNTIME_DIR, text=True
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, _RUN_STATE_FILE)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def save_state(self):
        with self.lock:
            self._save_state_locked()

    def _load_state(self):
        if not os.path.exists(_RUN_STATE_FILE):
            return
        try:
            with open(_RUN_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            self.log.append(f"[{time.strftime('%H:%M:%S')}] ⚠️ 读取持久化运行状态失败：{exc}")
            return

        outputs = {i: None for i in TASK_ORDER}
        outputs[8] = {}
        raw_outputs = data.get("outputs") if isinstance(data.get("outputs"), dict) else {}
        for tid in TASK_ORDER:
            value = raw_outputs.get(str(tid), raw_outputs.get(tid))
            outputs[tid] = value
        if not isinstance(outputs.get(8), dict):
            outputs[8] = {}
        self.outputs = outputs

        raw_memory = data.get("memory") if isinstance(data.get("memory"), dict) else {}
        self.memory = {
            k: raw_memory.get(k, []) if isinstance(raw_memory.get(k, []), list) else []
            for k in EMPLOYEES
        }
        self.log = data.get("log", []) if isinstance(data.get("log"), list) else []
        self.manager_reviews = (
            data.get("manager_reviews", {}) if isinstance(data.get("manager_reviews"), dict) else {}
        )
        try:
            self.total_episodes = int(data.get("total_episodes") or self.total_episodes)
        except (TypeError, ValueError):
            self.total_episodes = 50
        self.script_mode = data.get("script_mode") or self.script_mode
        self.seed = data.get("seed") or ""
        self.emp_configs = (
            data.get("emp_configs", {}) if isinstance(data.get("emp_configs"), dict) else {}
        )
        self.emp_settings = (
            data.get("emp_settings", {}) if isinstance(data.get("emp_settings"), dict) else {}
        )
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
        else:
            self.interrupted_task = self._coerce_task_id(data.get("interrupted_task"))
            self.interrupted_at = data.get("interrupted_at", "")

    # ---------- 数据操作（线程安全） ----------
    def reset(self):
        with self.lock:
            self.cancel = True         # 通知后台线程停止并丢弃在途写入
            self.is_running = False    # 强制解锁（即使后台线程异常未复位，也立刻可手动操作）
            self.outputs = {i: None for i in TASK_ORDER}
            self.outputs[8] = {}
            self.memory = {k: [] for k in EMPLOYEES}
            self.running_task = None
            self.running_employee = None
            self.failed_task = None
            self.log = []
            self.manager_reviews = {}
            self.interrupted_task = None
            self.interrupted_at = ""
            self._save_state_locked()

    def force_stop(self):
        """停止当前后台运行并立即解锁手动操作（不清空已产出的结果）。"""
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
        """记录工作室管理者对某个任务/批次的验收结果。"""
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
        """把一份任务9最终飞书文档归档进历史（与上一条内容相同则跳过，避免重复）。"""
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
        """删除指定下标的一份历史交付文档；下标无效时不做任何修改。"""
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
                        emp_settings=None, task_methods=None):
        with self.lock:
            self.total_episodes = total_episodes
            self.script_mode = script_mode
            self.seed = seed
            self.emp_configs = emp_configs
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


@st.cache_resource
def get_store():
    """跨会话 / 重连 / 刷新都返回同一个 RunStore 单例。"""
    return RunStore()


# ---------- 基于 store 的纯逻辑工具 ----------
def task8_batch_success(value):
    """任务8单个分镜批次是否为可用产出。"""
    return isinstance(value, str) and value.strip() != "" and not value.strip().startswith("❌")


def _latest_manager_review(store, key):
    reviews = (getattr(store, "manager_reviews", {}) or {}).get(str(key), [])
    if not reviews:
        return None
    return reviews[-1] if isinstance(reviews[-1], dict) else None


def _manager_review_failed(store, key):
    latest = _latest_manager_review(store, key)
    return bool(latest and latest.get("passed") is False)


def task_manager_review_failed(store, tid):
    """某任务是否已有管理者最终未通过记录。

    该状态用于展示“最后一轮产出”，但不能算作 task_done，避免误入下游任务。
    """
    if tid == 8:
        batches = (getattr(store, "outputs", {}) or {}).get(8)
        if not isinstance(batches, dict):
            return False
        return any(_manager_review_failed(store, f"8:{label}") for label in batches)
    return _manager_review_failed(store, str(tid))


def _coerce_positive_int(value, default=1):
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return n if n > 0 else default


def outline_episode_markers(content):
    """提取分集大纲里的集数标题。

    仍以“行首标题”为准，避免把正文里的“前3集 / 第41-50集”误判为真实剧集。
    兼容中文“第 1 集”、英文“Episode 1 / EP1 / EP 1 / EP.1”。
    """
    text = content or ""
    nums = {int(n) for n in re.findall(r"(?im)^[ \t>#*\-]*第\s*(\d+)\s*集", text)}
    nums |= {int(n) for n in re.findall(r"(?im)^[ \t>#*\-]*(?:Episode|EP\.?)\s*(\d+)(?!\d)", text)}
    return nums


def outline_episode_count(content):
    """按大纲实际标题推导最大集数，用于任务8批次目标兜底。"""
    nums = outline_episode_markers(content)
    return max(nums) if nums else 0


def storyboard_episode_markers(content):
    """提取任务8分镜里的集数标记。

    兼容两类常见输出：
    - 单独成行的“第 45 集” / “Episode 45”；
    - CSV 单元格里的“第45集：标题”，例如 `"45","第45集：真正的赛场",...`。
    """
    text = content or ""
    nums = {int(n) for n in re.findall(r"(?im)^[ \t>#*\-]*第\s*(\d+)\s*集", text)}
    nums |= {int(n) for n in re.findall(r"(?im)^[ \t>#*\-]*Episode\s*(\d+)(?!\d)", text)}
    nums |= {int(n) for n in re.findall(r"第\s*(\d+)\s*集", text)}
    nums |= {int(n) for n in re.findall(r"(?i)\bEpisode\s*(\d+)(?!\d)", text)}
    return nums


def task8_target_episodes(store):
    """任务8的目标集数：以项目设置为基础，若任务7大纲实际集数更多，则自动扩展。"""
    configured = _coerce_positive_int(getattr(store, "total_episodes", 1), default=1)
    outputs = getattr(store, "outputs", {}) or {}
    outline_count = outline_episode_count(outputs.get(7) if isinstance(outputs, dict) else "")
    return max(configured, outline_count)


def task8_expected_batch_labels(store):
    """按任务8目标集数计算自动分批生成应覆盖的全部批次名。"""
    total = task8_target_episodes(store)
    return {f"{a}-{b}集" for a, b in get_batches(total)}


def task8_batch_status(store):
    """汇总任务8分镜批次状态，供完成判定和恢复日志共用。"""
    batches = store.outputs.get(8)
    if not isinstance(batches, dict):
        batches = {}

    target_episodes = task8_target_episodes(store)
    expected_order = [f"{a}-{b}集" for a, b in get_batches(target_episodes)]
    expected_labels = set(expected_order)
    labels = set(batches.keys())

    review_failed_labels = [
        label for label in batches
        if _manager_review_failed(store, f"8:{label}")
    ]
    success_labels = [
        label for label, value in batches.items()
        if task8_batch_success(value) and label not in review_failed_labels
    ]
    failed_labels = [
        label for label, value in batches.items()
        if label in labels and (not task8_batch_success(value) or label in review_failed_labels)
    ]
    missing_labels = [label for label in expected_order if label not in labels]
    manual_labels = [label for label in batches.keys() if label not in expected_labels]

    merged = "\n".join(value for value in batches.values() if isinstance(value, str))
    nums = storyboard_episode_markers(merged)

    return {
        "expected_labels": expected_order,
        "success_labels": success_labels,
        "failed_labels": failed_labels,
        "missing_labels": missing_labels,
        "manual_labels": manual_labels,
        "episode_count": len(nums),
        "target_episodes": target_episodes,
        "has_batches": len(batches) > 0,
        "has_manual_batch": len(manual_labels) > 0,
        "all_batches_success": bool(batches) and len(failed_labels) == 0,
        "auto_batches_complete": expected_labels.issubset(labels),
    }


def task_done(store, tid):
    v = store.outputs.get(tid)
    if task_manager_review_failed(store, tid):
        return False
    if tid == 8:
        status = task8_batch_status(store)
        if not status["has_batches"]:
            return False
        if not status["all_batches_success"]:
            return False
        if not status["has_manual_batch"] and not status["auto_batches_complete"]:
            return False
        return status["episode_count"] >= status["target_episodes"]
    return isinstance(v, str) and v.strip() != "" and not v.startswith("❌")


def is_ready(store, tid):
    return all(task_done(store, d) for d in TASK_MAP[tid]["deps"])


def get_batches(total, size=10):
    out = []
    for i in range(1, total + 1, size):
        out.append((i, min(i + size - 1, total)))
    return out


def make_service(store, emp_key):
    """按运行参数快照里的配置创建 LLMService（后台线程可用，不依赖 session_state）。"""
    cfg = store.emp_configs.get(emp_key) or {
        "provider": "Mock (演示)", "key": "", "model": "mock-studio-model",
    }
    svc = LLMService()
    svc.set_config(cfg["provider"], cfg["key"], cfg["model"])
    return svc
