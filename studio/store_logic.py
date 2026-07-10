# -*- coding: utf-8 -*-
"""会话无关的运行状态纯逻辑工具。

这些函数只依赖传入的 store 对象字段，不依赖 Streamlit，可同时供 Streamlit 版与
FastAPI 单体版复用。
"""

import re

from .tasks import TASK_MAP
from .llm_service import LLMService


def task8_batch_success(value):
    """任务8单个分镜批次是否为可用产出。"""
    return isinstance(value, str) and value.strip() != "" and not value.strip().startswith("❌")


def task8_batch_passed(store, label):
    """任务8单个批次是否既有有效内容，又通过了管理者最新验收。"""
    batches = (getattr(store, "outputs", {}) or {}).get(8)
    value = batches.get(label) if isinstance(batches, dict) else None
    return task8_batch_success(value) and not _manager_review_failed(store, f"8:{label}")


def _latest_manager_review(store, key):
    reviews = (getattr(store, "manager_reviews", {}) or {}).get(str(key), [])
    if not reviews:
        return None
    return reviews[-1] if isinstance(reviews[-1], dict) else None


def _manager_review_failed(store, key):
    latest = _latest_manager_review(store, key)
    return bool(latest and latest.get("passed") is False)


def task_manager_review_failed(store, tid):
    """某任务是否已有管理者最终未通过记录。"""
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
    """提取分集大纲里的集数标题。"""
    text = content or ""
    nums = {int(n) for n in re.findall(r"(?im)^[ \t>#*\-]*第\s*(\d+)\s*集", text)}
    nums |= {int(n) for n in re.findall(r"(?im)^[ \t>#*\-]*(?:Episode|EP\.?)\s*(\d+)(?!\d)", text)}
    return nums


def outline_episode_count(content):
    """按大纲实际标题推导最大集数，用于任务8批次目标兜底。"""
    nums = outline_episode_markers(content)
    return max(nums) if nums else 0


def storyboard_episode_markers(content):
    """提取任务8分镜里的集数标记。"""
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


def get_batches(total, size=10):
    out = []
    for i in range(1, total + 1, size):
        out.append((i, min(i + size - 1, total)))
    return out


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
        label for label in batches
        if task8_batch_passed(store, label)
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


def make_service(store, emp_key):
    """按运行参数快照里的配置创建 LLMService。"""
    cfg = (getattr(store, "emp_configs", {}) or {}).get(emp_key) or {
        "provider": "Mock (演示)", "key": "", "model": "mock-studio-model",
    }
    svc = LLMService()
    svc.set_config(
        cfg.get("provider", "Mock (演示)"),
        cfg.get("key", ""),
        cfg.get("model", "mock-studio-model"),
    )
    return svc
