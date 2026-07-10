# -*- coding: utf-8 -*-
"""任务编排执行、飞书文档编排、分镜 CSV 解析、集数校验。

重要：本模块所有执行函数都只读写 RunStore（store），不访问 st.* / st.session_state，
因此可以在「后台线程」里安全运行（浏览器断连也不会中断），见 run_pipeline。
"""

import re
import csv
import json
import time
import urllib.error
import urllib.request

import pandas as pd

from .prompts import Prompts
from .tasks import (
    TASK_MAP,
    TASK_ORDER,
    MEM_NOTE,
    get_task_method,
    build_instruction,
    extract_outline_word_count_requirement,
)
from .employees import (
    EMPLOYEES,
    persona_from_store,
)
from .store_logic import (
    make_service,
    task_done,
    is_ready,
    get_batches,
    task8_batch_passed,
    task8_batch_status,
    task8_target_episodes,
    storyboard_episode_markers,
)

# 网络/API 临时异常的长重试间隔。总计最多重试 15 次，累计等待约 47 分钟。
# 永久性错误（鉴权、参数、权限、内容拦截等）不会进入长重试，避免无意义消耗。
RETRY_DELAYS = (2, 5, 10, 20, 30, 60, 120, 180, 300, 300, 300, 300, 300, 300, 600)
TASK_MAX_RETRIES = len(RETRY_DELAYS)

# 管理者最多发起 3 轮返工。保留上限是为了避免真实模型异常时无限消耗 API。
MANAGER_MAX_REVISIONS = 3

# 仅任务 1-4 提高管理者返工上限；任务 5-9 继续沿用默认 3 轮，避免改变长文本与分镜流程成本。
TASK_MANAGER_MAX_REVISIONS = {
    1: 5,
    2: 5,
    3: 5,
    4: 5,
}

# 任务5 / 任务7 分集大纲批次大小。单批过大时模型容易漏集或压缩细节；
# 每批 30 集能降低单次输出压力，同时保留完整剧集上下文给后续批次。
OUTLINE_BATCH_SIZE = 30

# 网络闪断后用于提前恢复的轻量探测。只用于判断端点是否重新可达，不替代正式 API 调用。
NETWORK_PROBE_TIMEOUT_SECONDS = 5
NETWORK_PROBE_INTERVAL_SECONDS = 3
PROVIDER_PROBE_URLS = {
    "Azure OpenAI (ByteDance)": "https://aidp.bytedance.net/api/modelhub/online/v2/crawl",
    "OpenRouter": "https://openrouter.ai/api/v1/models",
    "Google Gemini": "https://generativelanguage.googleapis.com",
    "OpenAI (GPT)": "https://api.openai.com/v1/models",
    "Anthropic (Claude)": "https://api.anthropic.com",
}


def _manager_max_revisions(tid):
    """读取某任务的管理者返工上限；未配置的任务保持全局默认。"""
    return TASK_MANAGER_MAX_REVISIONS.get(tid, MANAGER_MAX_REVISIONS)


def count_episodes(text):
    """统计分集大纲的「主线集数」，纠正过去把序章/番外/ACT区间/正文内嵌引用误计的问题。

    做法：
    1) 只匹配「行首的分集标题」，且中文必须形如「第 N 集」（数字后带“集”），英文形如
       「EPISODE N / EP N / EP1」，
       从而排除“第41-50集”这类区间、正文里的数字引用等假阳性；
    2) 取「从 1 开始的最长连续编号」作为主线集数 —— 即使模型多写了第0集/番外/跳号，也不会被算进去。
    """
    t = text or ""
    nums = set()
    # 中文集标题：行首（可带 # / * / > / - / 空格 / 制表符）+ 第 + 数字 + 集
    nums |= {int(n) for n in re.findall(r"(?im)^[ \t>#*\-]*第\s*(\d+)\s*集", t)}
    # 英文集标题：行首 Episode/EP + 数字；兼容 EP1、EP 1、EP.1（数字后非数字，避免 500 之类）
    nums |= {int(n) for n in re.findall(r"(?im)^[ \t>#*\-]*(?:Episode|EP\.?)\s*(\d+)(?!\d)", t)}
    if not nums:
        # 兜底：宽松匹配（无“集”的「第N」或行首「EPISODE/EP N」）
        nums = {int(n) for n in re.findall(r"(?im)^[ \t>#*\-]*(?:Episode|EP\.?|第)\s*(\d+)", t)}
    if not nums:
        return 0
    # 主线集数 = 从 1 开始的最长连续序列长度（忽略 第0集 / 番外 / 杂项编号）
    k = 0
    while (k + 1) in nums:
        k += 1
    return k if k > 0 else len(nums)


def build_io(store, tid):
    """为任务 1-7 构建 (system, user, mock_key)。任务 8/9 单独处理。

    系统提示（数字员工设定）与工作方法（任务协作方式）均在此「执行时」从 store 动态读取，
    因此用户在产品里自定义的设定会真正作用于生成。
    """
    o = store.outputs
    ep = store.total_episodes
    method = lambda i: get_task_method(i, store.task_methods)
    if tid == 1:
        seed = (store.seed or "").strip()
        seed_block = (
            f"\n【创作方向 / 赛道参考（用户提供）】\n{seed}\n"
            if seed
            else "\n（用户未指定方向，请你自主选择当前最具爆款潜力的赛道）\n"
        )
        return (
            persona_from_store(store, "researcher"),
            build_instruction(1, method(1)) + seed_block,
            "researcher_idea",
        )
    if tid == 2:
        system = persona_from_store(store, "creative") + "\n\n" + Prompts.ACT_GEN_SYSTEM
        user = (
            "【任务2：生成三幕式创意】根据任务1的原始创意写出一个三幕式创意。\n\n【工作方法】\n"
            + method(2)
            + "\n\n"
            + Prompts.ACT_GEN_TASK
            + f"\n[原始创意]\n{o[1]}"
        )
        return system, user, "three_act_v1"
    if tid == 3:
        return (
            persona_from_store(store, "reviewer"),
            build_instruction(3, method(3)) + f"\n\n[待审核 · 三幕式创意]\n{o[2]}",
            "review_3act",
        )
    if tid == 4:
        system = persona_from_store(store, "creative") + "\n\n" + Prompts.ACT_GEN_SYSTEM
        user = build_instruction(4, method(4)) + f"\n\n[原始三幕式创意]\n{o[2]}\n\n[审核员修改建议]\n{o[3]}"
        return system, user, "three_act_final"
    if tid == 5:
        system = persona_from_store(store, "writer") + "\n\n" + Prompts.OUTLINE_SYSTEM
        user = (
            "【任务5：生成分集大纲】根据任务4修改后的三幕式创意，调用分集大纲生成工具生成大纲。\n\n【工作方法】\n"
            + method(5)
            + "\n\n"
            + Prompts.OUTLINE_TASK.format(total_episodes=ep)
            + f"\n[三幕式创意]\n{o[4]}"
        )
        return system, user, f"outline:{ep}"
    if tid == 6:
        user = (
            build_instruction(6, method(6))
            + f"\n\n[原始创意]\n{o[1]}\n\n[三幕式创意 · 最终版]\n{o[4]}\n\n[待审核 · {ep} 集分集大纲]\n{o[5]}"
        )
        return persona_from_store(store, "reviewer"), user, "review_outline"
    if tid == 7:
        user = (
            build_instruction(7, method(7), total_episodes=ep)
            + f"\n\n[三幕式创意]\n{o[4]}\n\n[原始 {ep} 集大纲]\n{o[5]}\n\n[审核员修改建议]\n{o[6]}"
        )
        return persona_from_store(store, "writer"), user, f"outline_final:{ep}"
    raise ValueError(f"build_io 不支持任务 {tid}")


def _is_transient_error(msg):
    """判断报错是否为临时性错误（值得重试）。
    临时性：超时 / 429 限流 / 5xx 服务端错误 / 网络连接问题 / 传输层闪断。
    永久性（鉴权 401/403、参数 400、余额/配额不足、内容拦截等）一律返回 False，不重试。
    """
    t = (msg or "").lower()
    permanent_phrases = (
        "unauthorized", "forbidden", "permission denied",
        "invalid api key", "invalid_api_key", "incorrect api key",
        "bad request", "invalid request", "invalid_request",
        "model not found", "not found", "unsupported parameter",
        "insufficient quota", "insufficient_quota", "billing",
        "content policy", "policy violation", "blocked", "safety",
        "certificate verify failed", "self signed certificate",
        "context length", "maximum context", "request too large",
    )
    if any(p in t for p in permanent_phrases):
        return False
    if re.search(r"\b(400|401|402|403|404|407|413|422)\b", t):
        return False
    # HTTP 状态码用单词边界匹配，避免把 50000、60000 之类数字误判为 500
    if re.search(r"\b(408|409|425|429|500|502|503|504|520|521|522|523|524|529)\b", t):
        return True
    transient_phrases = (
        "apiconnectionerror", "api connection error",
        "connecterror", "connection error",
        "remoteprotocolerror", "remote protocol", "server disconnected",
        "readerror", "writeerror", "closedresourceerror", "pooltimeout",
        "protocolerror", "protocol error", "localprotocolerror",
        "rate limit", "rate_limit", "ratelimit", "too many requests",
        "timeout", "timed out", "read timed out", "readtimeout", "operation timed out",
        "overload", "overloaded", "temporar", "try again", "again later",
        "connection", "network", "network is unreachable",
        "network changed", "internet connection appears to be offline",
        "name resolution", "temporary failure in name resolution", "nodename nor servname", "dns",
        "econnreset", "connection reset", "connection aborted", "connection refused",
        "connection closed", "connection lost", "connection terminated",
        "remote end closed connection", "broken pipe", "software caused connection abort",
        "sslerror", "ssl error", "ssleoferror", "tls", "eof occurred",
        "curl error 28", "curl error 35", "curl error 52", "curl error 56",
        "service unavailable", "internal server error", "bad gateway", "gateway timeout",
    )
    return any(p in t for p in transient_phrases)


def _is_connectivity_error(msg):
    """判断是否更像本机网络/传输层闪断，而不是限流或服务端繁忙。

    这类错误在退避等待期间会做轻量端点探测；网络一恢复就提前进入下一次 API 调用。
    """
    t = (msg or "").lower()
    if any(p in t for p in ("rate limit", "rate_limit", "ratelimit", "too many requests", "overload")):
        return False
    if re.search(r"\b(408|520|521|522|523|524)\b", t):
        return True
    connectivity_phrases = (
        "apiconnectionerror", "api connection error",
        "connecterror", "connection error",
        "remoteprotocolerror", "remote protocol", "server disconnected",
        "readerror", "writeerror", "closedresourceerror", "pooltimeout",
        "protocolerror", "protocol error", "localprotocolerror",
        "timeout", "timed out", "read timed out", "readtimeout", "operation timed out",
        "connection", "network", "network is unreachable", "network changed",
        "internet connection appears to be offline",
        "name resolution", "temporary failure in name resolution", "nodename nor servname", "dns",
        "econnreset", "connection reset", "connection aborted", "connection refused",
        "connection closed", "connection lost", "connection terminated",
        "remote end closed connection", "broken pipe", "software caused connection abort",
        "sslerror", "ssl error", "ssleoferror", "tls", "eof occurred",
        "curl error 28", "curl error 35", "curl error 52", "curl error 56",
    )
    return any(p in t for p in connectivity_phrases)


def _format_retry_delay(seconds):
    if seconds < 60:
        return f"{seconds} 秒"
    minutes, rest = divmod(seconds, 60)
    return f"{minutes} 分 {rest} 秒" if rest else f"{minutes} 分钟"


def _log_retry(store, msg):
    if store is not None and hasattr(store, "log_line"):
        store.log_line(msg)


def _sleep_with_cancel(store, seconds):
    """可被 store.cancel 打断的等待；返回 False 表示等待中被取消。"""
    if seconds <= 0:
        return not getattr(store, "cancel", False)
    deadline = time.time() + seconds
    while time.time() < deadline:
        if getattr(store, "cancel", False):
            return False
        time.sleep(min(1.0, max(0.0, deadline - time.time())))
    return not getattr(store, "cancel", False)


def _provider_probe_url(svc):
    provider = getattr(svc, "provider", "")
    return PROVIDER_PROBE_URLS.get(provider)


def _probe_provider_endpoint(url):
    """轻量探测模型服务端点是否可达。

    HTTP 401/403/404/405 也说明网络路径已经恢复；正式鉴权与参数校验仍由后续 API 调用完成。
    """
    if not url:
        return False, "未配置探测端点"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ScriptStudio-Network-Probe/1.0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=NETWORK_PROBE_TIMEOUT_SECONDS) as resp:
            resp.read(1)
            return True, f"HTTP {getattr(resp, 'status', 'OK')}"
    except urllib.error.HTTPError as exc:
        return True, f"HTTP {exc.code}"
    except Exception as exc:
        return False, f"{exc.__class__.__name__}: {exc}"


def _wait_for_retry_delay(store, seconds, svc=None, task_label=None, probe_early=False):
    """等待下一次重试。

    - 非网络类临时错误：保持原有退避等待；
    - 网络/传输层闪断：等待期间短间隔探测端点，确认恢复后提前重试；
    - 探测失败不会阻断原有重试计划，避免个别端点拒绝探测导致任务卡死。
    """
    if seconds <= 0:
        return not getattr(store, "cancel", False)
    url = _provider_probe_url(svc)
    if not (probe_early and url):
        return _sleep_with_cancel(store, seconds)

    label = task_label or "模型调用"
    deadline = time.time() + seconds
    next_probe_at = 0.0
    last_detail = ""
    probe_logged = False
    while time.time() < deadline:
        if getattr(store, "cancel", False):
            return False
        now = time.time()
        if now >= next_probe_at:
            ok, detail = _probe_provider_endpoint(url)
            if ok:
                _log_retry(store, f"✅ {label} 网络/API端点已恢复可达（{detail}），立即重试。")
                return True
            last_detail = detail
            if not probe_logged:
                _log_retry(store, f"📡 {label} 等待网络/API端点恢复（最近探测：{detail}）。")
                probe_logged = True
            next_probe_at = now + NETWORK_PROBE_INTERVAL_SECONDS
        time.sleep(min(1.0, max(0.0, deadline - time.time())))

    if last_detail:
        _log_retry(
            store,
            f"📡 {label} 等待窗口结束，端点探测仍未确认恢复（最近探测：{last_detail}），按原计划继续重试。",
        )
    return not getattr(store, "cancel", False)


def _generate_with_retry(svc, system_prompt, user_prompt, mock_key=None, max_retries=TASK_MAX_RETRIES,
                         store=None, task_label=None):
    """带断网恢复能力的生成。
    - 仅对临时性错误重试，最多 max_retries 次，使用 RETRY_DELAYS 长退避；
    - 网络/传输层闪断会在退避期间探测端点，恢复后提前继续当前任务；
    - 永久性错误或重试用尽，返回 ❌ 错误串（不再重试）；
    - 停止 / 重置会通过 store.cancel 打断重试等待；
    - Mock 或配置类错误（如未填 Key，这些是普通返回而非异常）不会触发重试。
    """
    label = task_label or "模型调用"
    delays = list(RETRY_DELAYS[:max(0, max_retries)])
    attempts = max_retries + 1
    for attempt in range(attempts):
        if getattr(store, "cancel", False):
            return f"❌ 已取消：{label} 在 API 调用前收到停止信号。"
        try:
            res = svc.generate(system_prompt, user_prompt, mock_key=mock_key, raise_on_error=True)
            if attempt > 0:
                _log_retry(store, f"✅ {label} 网络/API 已恢复，已在第 {attempt} 次重试后继续执行。")
            return res
        except Exception as e:
            err = f"{e.__class__.__name__}: {e}"
            if _is_transient_error(err) and attempt < attempts - 1:
                delay = delays[attempt] if attempt < len(delays) else delays[-1] if delays else 0
                _log_retry(
                    store,
                    f"🌐 {label} 临时网络/API异常：{err}；"
                    f"将在 {_format_retry_delay(delay)} 后进行第 {attempt + 1}/{max_retries} 次重试。",
                )
                if not _wait_for_retry_delay(
                    store,
                    delay,
                    svc=svc,
                    task_label=label,
                    probe_early=_is_connectivity_error(err),
                ):
                    return f"❌ 已取消：{label} 在等待第 {attempt + 1} 次重试时收到停止信号。"
                continue
            if _is_transient_error(err):
                _log_retry(store, f"❌ {label} 临时网络/API异常重试耗尽：{err}")
            else:
                _log_retry(store, f"❌ {label} API调用失败（非临时错误，不进入长重试）：{err}")
            return f"❌ API 调用异常（已重试 {attempt} 次）: {err}"


def _review_key(tid, batch_label=None):
    return f"{tid}:{batch_label}" if batch_label else str(tid)


def _trim_for_review(content, max_chars=30000):
    """控制管理者验收输入长度，避免任务9等超长文档触发模型上下文超限。"""
    text = content or ""
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.65)]
    tail = text[-int(max_chars * 0.25):]
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n\n【中间内容因过长省略 {omitted} 字；硬性完整性校验已由程序执行】\n\n{tail}"


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "pass", "passed", "通过", "合格"}:
            return True
        if v in {"false", "no", "fail", "failed", "不通过", "不合格"}:
            return False
    return None


def _extract_json_obj(text):
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _parse_manager_review(raw):
    data = _extract_json_obj(raw)
    if isinstance(data, dict):
        passed = None
        for key in ("passed", "pass", "is_passed", "qualified", "通过", "合格"):
            if key in data:
                passed = _coerce_bool(data.get(key))
                break
        score = data.get("score", data.get("评分", data.get("quality_score")))
        summary = data.get("summary", data.get("结论", data.get("reason", "")))
        suggestions = data.get(
            "suggestions",
            data.get("revision_suggestions", data.get("修改建议", data.get("返工意见", ""))),
        )
        if isinstance(suggestions, list):
            suggestions = "\n".join(f"- {s}" for s in suggestions)
        return {
            "passed": passed,
            "score": score,
            "summary": str(summary or "").strip(),
            "suggestions": str(suggestions or "").strip(),
        }

    t = (raw or "").strip()
    low = t.lower()
    if "不通过" in t or "不合格" in t or "failed" in low or re.search(r"\bfail\b", low):
        passed = False
    elif "通过" in t or "合格" in t or "passed" in low or re.search(r"\bpass\b", low):
        passed = True
    else:
        passed = None
    return {"passed": passed, "score": None, "summary": t[:500], "suggestions": t}


def _episode_markers(content):
    return storyboard_episode_markers(content)


def _task9_hard_validate(store, content):
    """任务9专用硬校验。

    任务9是归档任务，核心不是再创作，而是确认任务4/7/8完整进入最终文档；
    因此用程序硬校验摘要作为主判断，避免把超长飞书全文交给管理者模型后产生主观误判。
    """
    text = content if isinstance(content, str) else ""
    outputs = getattr(store, "outputs", {}) or {}
    target = task8_target_episodes(store)
    issues = []

    task4 = outputs.get(4)
    if not isinstance(task4, str) or not task4.strip() or task4.startswith("❌"):
        issues.append("任务4三幕式创意缺失或为错误内容。")
    elif not task_done(store, 4):
        issues.append("任务4尚未达到完成状态或管理者验收未通过。")

    task7 = outputs.get(7)
    outline_count = count_episodes(task7 if isinstance(task7, str) else "")
    if not isinstance(task7, str) or not task7.strip() or task7.startswith("❌"):
        issues.append("任务7分集大纲缺失或为错误内容。")
    elif not task_done(store, 7):
        issues.append("任务7尚未达到完成状态或管理者验收未通过。")
    if outline_count < target:
        issues.append(f"任务7分集大纲集数不足：检测到 {outline_count} 集，目标 {target} 集。")

    task8_status = task8_batch_status(store)
    if not task8_status["has_batches"]:
        issues.append("任务8分镜脚本缺失。")
    if task8_status["failed_labels"]:
        issues.append(f"任务8存在失败/未通过批次：{task8_status['failed_labels']}。")
    if task8_status["missing_labels"]:
        issues.append(f"任务8缺失批次：{task8_status['missing_labels']}。")
    if not task8_status["has_manual_batch"] and not task8_status["auto_batches_complete"]:
        issues.append("任务8自动批次尚未完整覆盖目标集数。")
    if task8_status["episode_count"] < task8_status["target_episodes"]:
        issues.append(
            f"任务8分镜集数标记不足：检测到 {task8_status['episode_count']} 集，"
            f"目标 {task8_status['target_episodes']} 集。"
        )

    required_markers = ("## 一、三幕式创意", "## 二、", "## 三、分镜脚本表格")
    for marker in required_markers:
        if marker not in text:
            issues.append(f"最终飞书文档缺少章节：{marker}。")
    if "（缺失）" in text:
        issues.append("最终飞书文档仍包含缺失占位内容。")

    summary = (
        f"任务4：{'完成' if task_done(store, 4) else '未完成'}；"
        f"任务7：{outline_count}/{target} 集；"
        f"任务8：成功批次 {len(task8_status['success_labels'])}/{len(task8_status['expected_labels'])}，"
        f"缺失批次 {task8_status['missing_labels'] or '无'}，"
        f"失败批次 {task8_status['failed_labels'] or '无'}，"
        f"集数标记 {task8_status['episode_count']}/{task8_status['target_episodes']}；"
        f"最终文档：长度 {len(text)} 字，三章结构 "
        f"{'齐全' if all(marker in text for marker in required_markers) else '不齐全'}。"
    )
    return len(issues) == 0, "\n".join(issues), summary


def _hard_validate_result(store, tid, content, batch_label=None, start=None, end=None):
    """程序级硬性验收：管理者可做内容质量判断，硬性结构问题由代码兜底。"""
    text = content if isinstance(content, str) else ""
    issues = []
    if not text.strip():
        issues.append("产出为空。")
        return False, "\n".join(issues)
    if text.startswith("❌"):
        issues.append("模型调用返回错误，不能进入后续任务。")
        return False, "\n".join(issues)
    if len(text.strip()) < 80 and tid not in (8,):
        issues.append("产出内容过短，无法支撑后续创作。")

    if tid == 1:
        required = ("剧名", "核心", "标签")
        for item in required:
            if item not in text:
                issues.append(f"任务1缺少关键字段：{item}。")
    elif tid in (2, 4):
        if "剧名" not in text:
            issues.append(f"任务{tid}缺少剧名。")
        for act_no in (1, 2, 3):
            if not re.search(rf"\bact\s*{act_no}\b", text, re.IGNORECASE):
                act = f"Act {act_no}"
                issues.append(f"任务{tid}缺少三幕式字段：{act}。")
    elif tid in (3, 6):
        if "建议" not in text and "审核" not in text:
            issues.append(f"任务{tid}应输出结构化审核报告和修改建议。")
    elif tid in (5, 7):
        detected = count_episodes(text)
        if detected < store.total_episodes:
            issues.append(
                f"分集大纲集数不足：检测到 {detected} 集，目标 {store.total_episodes} 集。"
            )
        if "【开头高光】" not in text or "【正片】" not in text or "【结尾悬念】" not in text:
            issues.append("分集大纲必须包含【开头高光】【正片】【结尾悬念】三段。")
    elif tid == 8:
        try:
            df = parse_script_to_df(text)
        except Exception as exc:
            df = None
            issues.append(f"分镜 CSV 解析异常：{exc}")
        if df is None or len(df) == 0:
            issues.append("分镜脚本无法解析为有效表格。")
        if start is not None and end is not None:
            expected = set(range(start, end + 1))
            missing = sorted(expected - _episode_markers(text))
            if missing:
                issues.append(f"分镜批次 {batch_label or ''} 缺少集数标记：{missing[:10]}。")
    elif tid == 9:
        task9_passed, task9_suggestions, _summary = _task9_hard_validate(store, text)
        if not task9_passed:
            issues.append(task9_suggestions)

    return len(issues) == 0, "\n".join(issues)


def _task_standard(store, tid, batch_label=None):
    t = TASK_MAP[tid]
    method = get_task_method(tid, store.task_methods)
    hard = {
        1: "必须输出1个完整原始创意，包含剧名、核心设定、核心爽点/痛点、目标受众、黄金标签组合。",
        2: "必须输出格式工整的三幕式创意，包含剧名、受众、故事类型、故事背景、Act 1/2/3。",
        3: "必须输出结构化审核报告，只给审核和修改建议，不直接改写剧本。",
        4: "必须输出新版三幕式创意最终版，并附本次优化对应说明。",
        5: f"必须生成完整 {store.total_episodes} 集分集大纲，每集包含【开头高光】【正片】【结尾悬念】。",
        6: "必须输出分集大纲结构化审核报告，重点检查一致性、商业化、留存和悬念。",
        7: f"必须输出优化版 {store.total_episodes} 集分集大纲，集数完整，每集三段结构完整。",
        8: f"必须输出可解析 CSV 分镜脚本；{batch_label or '当前批次'} 每集都要有单独集数标题和镜号。",
        9: "必须汇总任务4、任务7、任务8，形成飞书可粘贴 Markdown 文档，不能遗漏章节和内容。",
    }
    return (
        f"任务目标：{t['desc']}\n"
        f"当前协作方式 / 工作方法：\n{method}\n\n"
        f"硬性验收标准：{hard.get(tid, '产出必须完整、清晰、可支撑后续任务。')}"
    )


def _append_manager_dispatch(store, tid, user_prompt, feedback_notes=None, batch_label=None):
    notes = "\n\n".join([n for n in (feedback_notes or []) if n])
    feedback = f"\n\n【上轮管理者验收返工意见 · 必须逐条修正】\n{notes}" if notes else ""
    return (
        user_prompt
        + "\n\n【工作室管理者派发指令】\n"
        + "本任务由工作室管理者根据当前项目阶段、数字员工能力和上下游产出派发。"
        + "你必须严格遵守以下任务标准；产出后会由工作室管理者验收，未通过将自动返工。\n\n"
        + _task_standard(store, tid, batch_label=batch_label)
        + feedback
    )


def _employees_brief(store):
    lines = []
    for k, e in EMPLOYEES.items():
        if k == "manager":
            continue
        overrides = (getattr(store, "emp_settings", None) or {}).get(k) or {}
        skills = (overrides.get("skills") or "").strip()
        if not skills:
            skills = "；".join(e["skills"][:2])
        lines.append(f"- {e['name']}：{e['title']}；核心能力：{skills[:260]}")
    return "\n".join(lines)


def _manager_review_prompt(store, tid, content, round_no, hard_passed, hard_suggestions,
                           batch_label=None):
    label = f"任务{tid}" + (f" · {batch_label}" if batch_label else "")
    hard_block = "通过" if hard_passed else f"未通过：\n{hard_suggestions}"
    return (
        f"你正在以工作室管理者身份验收【{label}】第 {round_no} 轮结果。\n\n"
        "【你掌握的数字员工背景】\n"
        + _employees_brief(store)
        + "\n\n【本任务标准】\n"
        + _task_standard(store, tid, batch_label=batch_label)
        + "\n\n【程序硬性结构校验】\n"
        + hard_block
        + "\n\n【待验收产出】\n"
        + _trim_for_review(content)
        + "\n\n请严格判断该产出能否支撑后续任务。只输出 JSON，不要输出 Markdown：\n"
          "{\n"
          '  "passed": true 或 false,\n'
          '  "score": 0-100,\n'
          '  "summary": "一句话验收结论",\n'
          '  "suggestions": "若不通过，给出可直接并入下一轮Prompt的具体返工意见；若通过，写通过原因"\n'
          "}"
    )


def _validate_by_manager(store, tid, content, round_no, batch_label=None, start=None, end=None,
                         max_retries=TASK_MAX_RETRIES):
    hard_passed, hard_suggestions = _hard_validate_result(
        store, tid, content, batch_label=batch_label, start=start, end=end
    )
    svc = make_service(store, "manager")
    raw = _generate_with_retry(
        svc,
        persona_from_store(store, "manager"),
        _manager_review_prompt(store, tid, content, round_no, hard_passed, hard_suggestions, batch_label),
        mock_key=f"manager_review:{tid}",
        max_retries=max_retries,
        store=store,
        task_label=f"工作室管理者验收任务{tid}" + (f" · {batch_label}" if batch_label else ""),
    )
    if isinstance(raw, str) and raw.startswith("❌"):
        return {
            "time": time.strftime("%H:%M:%S"),
            "round": round_no,
            "passed": False,
            "fatal": True,
            "score": 0,
            "summary": "工作室管理者验收调用失败。",
            "suggestions": raw,
            "raw": raw,
        }

    parsed = _parse_manager_review(raw)
    manager_passed = parsed["passed"]
    if manager_passed is None:
        manager_passed = hard_passed

    suggestions = []
    if not hard_passed and hard_suggestions:
        suggestions.append("【程序硬性结构问题】\n" + hard_suggestions)
    if parsed["suggestions"]:
        suggestions.append(parsed["suggestions"])

    passed = bool(hard_passed and manager_passed)
    return {
        "time": time.strftime("%H:%M:%S"),
        "round": round_no,
        "passed": passed,
        "fatal": False,
        "score": parsed["score"],
        "summary": parsed["summary"] or ("验收通过。" if passed else "验收不通过，需要返工。"),
        "suggestions": "\n\n".join(suggestions).strip(),
        "raw": raw,
    }


def _validate_task9_by_manager(store, content, round_no):
    """任务9专用管理者验收：只看硬校验摘要，不看超长飞书全文。"""
    hard_passed, hard_suggestions, summary = _task9_hard_validate(store, content)
    if hard_passed:
        return {
            "time": time.strftime("%H:%M:%S"),
            "round": round_no,
            "passed": True,
            "fatal": False,
            "score": 100,
            "summary": "任务9归档硬性完整性校验通过，最终飞书文档可交付。",
            "suggestions": "通过：任务4、任务7、任务8均完整汇总，章节齐全，无缺失占位。",
            "raw": json.dumps(
                {"passed": True, "hard_validation_summary": summary},
                ensure_ascii=False,
            ),
        }
    return {
        "time": time.strftime("%H:%M:%S"),
        "round": round_no,
        "passed": False,
        "fatal": False,
        "score": 0,
        "summary": "任务9归档硬性完整性校验未通过，需要重新归档或补齐前置内容。",
        "suggestions": "【程序硬性结构问题】\n" + hard_suggestions + "\n\n【验收摘要】\n" + summary,
        "raw": json.dumps(
            {"passed": False, "hard_validation_summary": summary, "issues": hard_suggestions},
            ensure_ascii=False,
        ),
    }


def _failed_review_result(advice, preserved_output, max_revisions=None):
    """管理者多轮验收失败时返回精简错误原因；最后一轮产出由调用方单独保留到输出区。"""
    revisions = MANAGER_MAX_REVISIONS if max_revisions is None else max_revisions
    return (
        f"❌ 工作室管理者验收未通过：已自动返工 {revisions} 轮。\n\n"
        f"{advice}"
    )


def validate_manual_output(store, tid, content, batch_label=None, start=None, end=None,
                           max_retries=TASK_MAX_RETRIES):
    """验收用户手动保存/粘贴的产出。

    与 AI 生成路径共用同一套管理者验收和硬性结构校验；不自动返工，只记录本次验收结果。
    """
    review_key = _review_key(tid, batch_label)
    store.clear_manager_review(review_key)
    with store.lock:
        store.running_employee = "manager"
    label = f"任务{tid}" + (f" · {batch_label}" if batch_label else "")
    store.log_line(f"🧭 工作室管理者验收手动保存产出：{label}（第 1 轮）…")
    if tid == 9:
        review = _validate_task9_by_manager(store, content, 1)
    else:
        review = _validate_by_manager(
            store,
            tid,
            content,
            1,
            batch_label=batch_label,
            start=start,
            end=end,
            max_retries=max_retries,
        )
    if not getattr(store, "cancel", False):
        store.add_manager_review(review_key, review)
        if review.get("fatal"):
            with store.lock:
                store.failed_task = tid
        if review.get("passed"):
            store.add_memory("manager", f"已验收手动保存产出：{label}，确认达标。")
        else:
            store.log_line(f"🔁 手动保存产出未通过管理者验收：{label}。")
    with store.lock:
        store.running_employee = None
    return review


def _outline_batch_ranges(total):
    """任务5 / 任务7 分集大纲批次：每批最多 OUTLINE_BATCH_SIZE 集。"""
    return get_batches(total, OUTLINE_BATCH_SIZE)


def _merge_outline_batches(tid, total_episodes, parts):
    """把已生成的大纲批次拼接成任务5/7对外唯一产出，供评审、验收、任务8继续使用。"""
    clean_parts = [p.strip() for p in parts if isinstance(p, str) and p.strip()]
    body = "\n\n".join(clean_parts).strip()
    if not body:
        return ""
    if re.match(r"\s*【.*分集大纲", body):
        return body
    tag = "优化版" if tid == 7 else "初版"
    return f"【{tag} · {total_episodes} 集分集大纲（ACT 划分）】\n\n{body}"


def _build_outline_batch_io(store, tid, start, end, previous_outline, feedback_notes):
    """为任务5/7构建单个分集大纲批次的模型输入。

    关键约束：
    - 本批只生成 start..end 集，避免单次生成过长导致漏集；
    - 每批都输入此前已生成的全部批次，保证人物、伏笔、冲突和悬念连续；
    - 批次全部完成后再统一拼接为 store.outputs[tid]，保持下游接口不变。
    """
    total = store.total_episodes
    method = get_task_method(tid, store.task_methods)
    previous = (previous_outline or "").strip() or "（无，当前为首批，请从第 1 集开始建立完整故事节奏。）"
    notes = "\n\n".join([n for n in (feedback_notes or []) if n])
    feedback = f"\n\n【上轮管理者验收返工意见 · 必须逐条修正】\n{notes}" if notes else ""
    word_count_requirement = extract_outline_word_count_requirement(method)
    batch_rule = (
        f"\n\n【分批生成硬性要求】\n"
        f"1. 本任务总集数为 {total} 集，本次只生成第 {start}-{end} 集，禁止输出其他集数。\n"
        f"2. 每集必须使用独立集标题，例如“第 {start} 集”或“EPISODE {start}”，编号必须连续、不得跳号、不得合并集数。\n"
        "3. 每集仍严格包含【开头高光】【正片】【结尾悬念】三段，并标明 ACT1/ACT2/ACT3 所属阶段。\n"
        f"4. {word_count_requirement}剧情要具体、充实、有动作、场景、情绪转折和悬念。\n"
        "5. 生成前必须阅读【此前已生成批次分集大纲】，承接人物状态、剧情伏笔、冲突升级、上一批结尾悬念和世界观设定，保证连续性和一致性。\n"
        "6. 只输出本批次分集大纲正文，不要输出寒暄、解释、总结或与本批无关的内容。"
    )
    previous_block = f"\n\n【此前已生成批次分集大纲 · 必须承接】\n{previous}"

    if tid == 5:
        system = persona_from_store(store, "writer") + "\n\n" + Prompts.OUTLINE_SYSTEM
        user = (
            "【任务5：生成分集大纲】根据任务4修改后的三幕式创意，调用分集大纲生成工具分批生成大纲。\n\n【工作方法】\n"
            + method
            + "\n\n【完整任务要求参考】\n"
            + Prompts.OUTLINE_TASK.format(total_episodes=total)
            + batch_rule
            + previous_block
            + f"\n\n[三幕式创意]\n{store.outputs[4]}"
            + feedback
        )
        return system, user, f"outline:{total}:{start}-{end}"

    if tid == 7:
        system = persona_from_store(store, "writer")
        user = (
            "【完整任务要求参考 · 本次按分批方式执行，实际输出范围以“分批生成硬性要求”为准】\n"
            + build_instruction(7, method, total_episodes=total)
            + batch_rule
            + previous_block
            + f"\n\n[三幕式创意]\n{store.outputs[4]}"
            + f"\n\n[原始 {total} 集大纲]\n{store.outputs[5]}"
            + f"\n\n[审核员修改建议]\n{store.outputs[6]}"
            + feedback
        )
        return system, user, f"outline_final:{total}:{start}-{end}"

    raise ValueError(f"_build_outline_batch_io 不支持任务 {tid}")


def run_outline_task_batched(store, tid, max_retries=TASK_MAX_RETRIES):
    """任务5/7专用：每 30 集一批生成分集大纲，最终拼接成完整任务产出。

    对外行为保持不变：成功后 store.outputs[tid] 仍是一份完整大纲文本，下游任务继续读取同一字段。
    管理者验收仍在完整拼接结果上执行；若不通过，带返工意见重新分批生成。
    """
    owner = TASK_MAP[tid]["owner"]
    review_key = _review_key(tid)
    store.clear_manager_review(review_key)
    feedback_notes = []
    last_review = None
    last_output = None
    ranges = _outline_batch_ranges(store.total_episodes)

    for round_no in range(1, MANAGER_MAX_REVISIONS + 2):
        parts = []
        for start, end in ranges:
            if getattr(store, "cancel", False):
                with store.lock:
                    store.running_employee = None
                return f"❌ 已取消：任务{tid} 在生成第 {start}-{end} 集前收到停止信号。"

            previous_outline = _merge_outline_batches(tid, store.total_episodes, parts)
            system, user, mkey = _build_outline_batch_io(
                store, tid, start, end, previous_outline, feedback_notes
            )
            svc = make_service(store, owner)
            with store.lock:
                store.running_employee = owner
            label = f"任务{tid}「{TASK_MAP[tid]['title']}」第 {start}-{end} 集批次"
            store.log_line(f"🧩 {label} 开始生成（每批最多 {OUTLINE_BATCH_SIZE} 集）。")
            res = _generate_with_retry(
                svc,
                system,
                user,
                mock_key=mkey,
                max_retries=max_retries,
                store=store,
                task_label=label,
            )
            if getattr(store, "cancel", False):
                with store.lock:
                    store.running_employee = None
                return f"❌ 已取消：任务{tid} 第 {start}-{end} 集批次返回后收到停止信号，已丢弃本轮结果。"
            if isinstance(res, str) and res.startswith("❌"):
                store.set_output(tid, res)
                with store.lock:
                    store.running_employee = None
                return res

            parts.append(res)
            merged = _merge_outline_batches(tid, store.total_episodes, parts)
            store.set_output(tid, merged)
            store.log_line(f"✅ 任务{tid} 第 {start}-{end} 集批次完成，已拼接 {count_episodes(merged)}/{store.total_episodes} 集。")

        final_output = _merge_outline_batches(tid, store.total_episodes, parts)
        store.set_output(tid, final_output)
        last_output = final_output

        with store.lock:
            store.running_employee = "manager"
        store.log_line(f"🧭 工作室管理者验收任务{tid}完整拼接大纲（第 {round_no} 轮）…")
        review = _validate_by_manager(store, tid, final_output, round_no, max_retries=max_retries)
        last_review = review
        if getattr(store, "cancel", False):
            with store.lock:
                store.running_employee = None
            return f"❌ 已取消：任务{tid} 在管理者验收返回后收到停止信号，已丢弃本轮验收结果。"
        store.add_manager_review(review_key, review)

        if review["fatal"]:
            err = f"❌ 工作室管理者验收失败：{review['suggestions']}"
            store.set_output(tid, err)
            with store.lock:
                store.running_employee = None
            return err
        if review["passed"]:
            store.add_memory(owner, MEM_NOTE[tid])
            store.add_memory("manager", f"已验收任务{tid}「{TASK_MAP[tid]['title']}」分批拼接结果并确认达标。")
            with store.lock:
                store.running_employee = None
            return final_output

        if round_no <= MANAGER_MAX_REVISIONS:
            advice = review["suggestions"] or review["summary"] or "请提升分集完整性、上下批次连续性和三段格式规范。"
            feedback_notes.append(advice)
            store.log_line(f"🔁 任务{tid} 完整大纲未通过管理者验收，自动返工第 {round_no} 次，并重新分批生成。")

    advice = (last_review or {}).get("suggestions") or "多轮返工后仍未达到管理者验收标准。"
    err = _failed_review_result(advice, last_output)
    store.set_output(tid, last_output if isinstance(last_output, str) and last_output.strip() else err)
    with store.lock:
        store.running_employee = None
    return err


def run_generic_task(store, tid, max_retries=TASK_MAX_RETRIES):
    if tid in (5, 7):
        return run_outline_task_batched(store, tid, max_retries=max_retries)

    owner = TASK_MAP[tid]["owner"]
    review_key = _review_key(tid)
    store.clear_manager_review(review_key)
    feedback_notes = []
    last_review = None
    last_output = None
    max_revisions = _manager_max_revisions(tid)

    for round_no in range(1, max_revisions + 2):
        system, user, mkey = build_io(store, tid)
        user = _append_manager_dispatch(store, tid, user, feedback_notes)
        svc = make_service(store, owner)
        with store.lock:
            store.running_employee = owner
        res = _generate_with_retry(
            svc,
            system,
            user,
            mock_key=mkey,
            max_retries=max_retries,
            store=store,
            task_label=f"任务{tid}「{TASK_MAP[tid]['title']}」",
        )
        if getattr(store, "cancel", False):
            with store.lock:
                store.running_employee = None
            return f"❌ 已取消：任务{tid} 在模型调用返回后收到停止信号，已丢弃本轮结果。"
        store.set_output(tid, res)
        if isinstance(res, str) and res.startswith("❌"):
            with store.lock:
                store.running_employee = None
            return res
        last_output = res

        with store.lock:
            store.running_employee = "manager"
        store.log_line(f"🧭 工作室管理者验收任务{tid}（第 {round_no} 轮）…")
        review = _validate_by_manager(store, tid, res, round_no, max_retries=max_retries)
        last_review = review
        if getattr(store, "cancel", False):
            with store.lock:
                store.running_employee = None
            return f"❌ 已取消：任务{tid} 在管理者验收返回后收到停止信号，已丢弃本轮验收结果。"
        store.add_manager_review(review_key, review)

        if review["fatal"]:
            err = f"❌ 工作室管理者验收失败：{review['suggestions']}"
            store.set_output(tid, err)
            with store.lock:
                store.running_employee = None
            return err
        if review["passed"]:
            store.add_memory(owner, MEM_NOTE[tid])
            store.add_memory("manager", f"已验收任务{tid}「{TASK_MAP[tid]['title']}」并确认达标。")
            with store.lock:
                store.running_employee = None
            return res

        if round_no <= max_revisions:
            advice = review["suggestions"] or review["summary"] or "请提升内容完整性、格式规范性和上下游一致性。"
            feedback_notes.append(advice)
            store.log_line(f"🔁 任务{tid} 未通过管理者验收，自动返工第 {round_no} 次。")

    advice = (last_review or {}).get("suggestions") or "多轮返工后仍未达到管理者验收标准。"
    err = _failed_review_result(advice, last_output, max_revisions=max_revisions)
    store.set_output(tid, last_output if isinstance(last_output, str) and last_output.strip() else err)
    with store.lock:
        store.running_employee = None
    return err


def run_task8_batch(store, start, end, max_retries=TASK_MAX_RETRIES):
    mode = store.script_mode
    if mode == "comic":
        base, mtag = Prompts.COMIC_SCRIPT_TASK_TEMPLATE, "comic"
    else:
        base, mtag = Prompts.SCRIPT_TASK_TEMPLATE, "standard"
    user = (
        "【任务8：生成分镜脚本表格】根据任务7修改后的分集大纲，逐批生成分镜脚本并严格审核。\n\n【工作方法】\n"
        + get_task_method(8, store.task_methods)
        + "\n\n"
        + base.format(episode_range=f"{start}-{end}")
        + f"\n[大纲]\n{store.outputs[7]}"
    )
    label = f"{start}-{end}集"
    review_key = _review_key(8, label)
    store.clear_manager_review(review_key)
    feedback_notes = []
    last_review = None
    last_output = None

    for round_no in range(1, MANAGER_MAX_REVISIONS + 2):
        round_user = _append_manager_dispatch(store, 8, user, feedback_notes, batch_label=label)
        svc = make_service(store, "reviewer")
        with store.lock:
            store.running_employee = "reviewer"
        res = _generate_with_retry(
            svc, Prompts.SCRIPT_SYSTEM, round_user,
            mock_key=f"script:{mtag}:{start}-{end}", max_retries=max_retries,
            store=store,
            task_label=f"任务8分镜批次 {label}",
        )
        if getattr(store, "cancel", False):
            with store.lock:
                store.running_employee = None
            return f"❌ 已取消：任务8分镜批次 {label} 在模型调用返回后收到停止信号，已丢弃本轮结果。"
        store.set_batch(label, res)
        if isinstance(res, str) and res.startswith("❌"):
            with store.lock:
                store.running_employee = None
            return res
        last_output = res

        with store.lock:
            store.running_employee = "manager"
        store.log_line(f"🧭 工作室管理者验收任务8 · {label}（第 {round_no} 轮）…")
        review = _validate_by_manager(
            store, 8, res, round_no, batch_label=label, start=start, end=end,
            max_retries=max_retries,
        )
        last_review = review
        if getattr(store, "cancel", False):
            with store.lock:
                store.running_employee = None
            return f"❌ 已取消：任务8分镜批次 {label} 在管理者验收返回后收到停止信号，已丢弃本轮验收结果。"
        store.add_manager_review(review_key, review)

        if review["fatal"]:
            err = f"❌ 工作室管理者验收失败：{review['suggestions']}"
            store.set_batch(label, err)
            with store.lock:
                store.running_employee = None
            return err
        if review["passed"]:
            store.add_memory("reviewer", MEM_NOTE[8])
            store.add_memory("manager", f"已验收任务8「分镜脚本」{label} 并确认达标。")
            with store.lock:
                store.running_employee = None
            return res

        if round_no <= MANAGER_MAX_REVISIONS:
            advice = review["suggestions"] or review["summary"] or "请修复分镜格式、集数标记和剧情一致性。"
            feedback_notes.append(advice)
            store.log_line(f"🔁 任务8 · {label} 未通过管理者验收，自动返工第 {round_no} 次。")

    advice = (last_review or {}).get("suggestions") or "多轮返工后仍未达到管理者验收标准。"
    err = _failed_review_result(advice, last_output)
    store.set_batch(label, last_output if isinstance(last_output, str) and last_output.strip() else err)
    with store.lock:
        store.running_employee = None
    return err


def _df_to_markdown(df):
    """把分镜 DataFrame 转成飞书可识别的 Markdown 表格（与任务8表格列一致）。"""
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            cell = str(row[c]) if row[c] is not None else ""
            cell = cell.replace("\\", "").replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def compile_feishu_doc(store):
    o = store.outputs
    ep = task8_target_episodes(store)
    parts = [
        "# 📕 短剧剧本工作室 · 最终交付文档（飞书格式）",
        f"> 由工作室管理者调度 5 位创作数字员工协作产出 · 共 {ep} 集 · 文档助理已逐项校对，核对无误。",
        "",
        "## 一、三幕式创意（任务 4 · 最终版）",
        o.get(4) or "（缺失）",
        "",
        f"## 二、{ep} 集分集大纲（任务 7 · 优化版）",
        o.get(7) or "（缺失）",
        "",
        "## 三、分镜脚本表格（任务 8）",
    ]
    scripts = o.get(8) or {}
    if scripts:
        for label, content in scripts.items():
            parts.append(f"\n### 分镜 · {label}\n")
            df = None
            try:
                df = parse_script_to_df(content or "")
            except Exception:
                df = None
            if df is not None and len(df) > 0:
                parts.append(_df_to_markdown(df))
            else:
                parts.append("```")
                parts.append((content or "").strip())
                parts.append("```")
    else:
        parts.append("（缺失）")
    return "\n".join(parts)


def _extract_drama_title(store):
    """从已有产出里提取剧名（用于文档库归档标题）；取不到则用占位名。"""
    for tid in (4, 2, 1):
        s = store.outputs.get(tid)
        if isinstance(s, str):
            m = re.search(r"剧名[:：]\s*([^\n]+)", s)
            if m:
                return m.group(1).strip()[:40]
    return "未命名短剧"


def run_task9(store):
    review_key = _review_key(9)
    store.clear_manager_review(review_key)
    last_review = None
    last_doc = None

    for round_no in range(1, MANAGER_MAX_REVISIONS + 2):
        with store.lock:
            store.running_employee = "assistant"
        doc = compile_feishu_doc(store)
        last_doc = doc
        store.set_output(9, doc)

        with store.lock:
            store.running_employee = "manager"
        store.log_line(f"🧭 工作室管理者验收任务9 · 飞书最终交付文档（第 {round_no} 轮）…")
        review = _validate_task9_by_manager(store, doc, round_no)
        last_review = review
        store.add_manager_review(review_key, review)

        if getattr(store, "cancel", False):
            with store.lock:
                store.running_employee = None
            return "❌ 已取消：任务9 在管理者验收返回后收到停止信号，已保留当前飞书文档。"

        if review["passed"]:
            store.add_memory("assistant", MEM_NOTE[9])
            store.add_memory("manager", "已验收任务9「飞书归档」并确认最终交付文档达标。")
            store.add_doc_history(_extract_drama_title(store), doc)
            with store.lock:
                store.running_employee = None
            return doc

        if round_no <= MANAGER_MAX_REVISIONS:
            store.log_line(f"🔁 任务9 未通过管理者验收，文档助理自动重新归档第 {round_no} 次。")

    advice = (last_review or {}).get("suggestions") or "多轮重新归档后仍未达到任务9硬性完整性标准。"
    err = _failed_review_result(advice, last_doc)
    store.set_output(9, last_doc if isinstance(last_doc, str) and last_doc.strip() else err)
    with store.lock:
        store.running_employee = None
    return err


# ==========================================
# 后台流水线（在线程中运行，只读写 store，不触碰 st.*）
# ==========================================
def _mark_failed(store, tid, res):
    with store.lock:
        store.failed_task = tid
    store.log_line(f"❌ 任务{tid} 失败（已完成 API 重试 / 管理者返工校验）：{res}")
    store.log_line("⛔ 任务失败，再次点击生成或者调整手动模式")


def _format_task8_status(status):
    """把任务8批次状态压缩成一行日志，方便恢复/排障。"""
    return (
        f"成功批次 {len(status['success_labels'])}/{len(status['expected_labels'])}；"
        f"失败批次 {status['failed_labels'] or '无'}；"
        f"缺失批次 {status['missing_labels'] or '无'}；"
        f"本地批次 {status['manual_labels'] or '无'}；"
        f"集数标记 {status['episode_count']}/{status['target_episodes']}"
    )


def run_pipeline(store, from_progress=False):
    """后台执行尚未完成的任务，直到全部完成或失败/被取消。仅读写 store。

    from_progress=True（手动“自动执行后续剩余任务”）：仅从「当前已完成的最高任务」的下一个
    任务开始往后执行，不回头补跑前序未完成任务。
    """
    try:
        with store.lock:
            store.is_running = True
            store.cancel = False
            store.failed_task = None
            store.interrupted_task = None
            store.interrupted_at = ""
        store.save_state()

        start = 1
        if from_progress:
            done = [t for t in TASK_ORDER if task_done(store, t)]
            start = (max(done) + 1) if done else 1

        ok = True
        for tid in [1, 2, 3, 4, 5, 6, 7]:
            if store.cancel:
                ok = False
                break
            if tid < start or task_done(store, tid):
                continue
            # 不因依赖未满足而中止：从当前进度往后执行所有未完成任务
            # （正常顺序执行时前置自然会先产出；若用户跳过了某前置，则按空内容处理，绝不卡住）
            e = EMPLOYEES[TASK_MAP[tid]["owner"]]
            with store.lock:
                store.running_task = tid
            store.log_line(f"▶️ 任务{tid}「{TASK_MAP[tid]['title']}」开始 · {e['name']}")
            res = run_generic_task(store, tid)
            if store.cancel:
                ok = False
                break
            if isinstance(res, str) and res.startswith("❌"):
                _mark_failed(store, tid, res)
                ok = False
                break
            store.log_line(f"✅ 任务{tid} 完成 · {e['name']}")

        # 任务 8（分批，只跳过已成功生成的批次；失败批次恢复后必须重试）
        if ok and not store.cancel and 8 >= start and not task_done(store, 8):
            with store.lock:
                store.running_task = 8
            store.log_line("⚖️ 犀利的短剧剧本审核员开始生成分镜脚本（分批）…")
            store.log_line(f"📊 任务8恢复检查：{_format_task8_status(task8_batch_status(store))}")
            for (a, b) in get_batches(task8_target_episodes(store)):
                if store.cancel:
                    ok = False
                    break
                label = f"{a}-{b}集"
                if task8_batch_passed(store, label):
                    store.log_line(f"⏭️ 跳过已成功分镜批次：{label}")
                    continue
                if label in store.outputs[8]:
                    store.log_line(f"🔁 重新生成失败/无效分镜批次：{label}")
                else:
                    store.log_line(f"🎬 生成缺失分镜批次：{label}")
                res = run_task8_batch(store, a, b)
                if store.cancel:
                    ok = False
                    break
                if isinstance(res, str) and res.startswith("❌"):
                    _mark_failed(store, 8, res)
                    ok = False
                    break
            if ok and not store.cancel and task_done(store, 8):
                store.log_line(f"✅ 任务8 完成 · 全部分镜脚本已生成（{_format_task8_status(task8_batch_status(store))}）")
            elif ok and not store.cancel:
                _mark_failed(
                    store,
                    8,
                    "❌ 任务8 未完成：分镜批次不完整或存在失败批次，已阻止进入任务9。"
                    f"\n{_format_task8_status(task8_batch_status(store))}",
                )
                ok = False

        # 任务 9（编排归档）
        if ok and not store.cancel and 9 >= start and not task_done(store, 9):
            with store.lock:
                store.running_task = 9
            store.log_line("📋 文档助理整理飞书交付文档…")
            res = run_task9(store)
            if isinstance(res, str) and res.startswith("❌"):
                _mark_failed(store, 9, res)
                ok = False
            elif not store.cancel:
                store.log_line("✅ 任务9 完成 · 飞书文档已生成")

        if ok and not store.cancel:
            store.log_line("🎉 全流程执行完毕！可在下方查看每位数字员工的任务产出。")
    except Exception as ex:
        store.log_line(f"❌ 运行异常：{ex}")
    finally:
        with store.lock:
            store.running_task = None
            store.running_employee = None
            store.is_running = False
        store.save_state()


def parse_script_to_df(content):
    """复用原工具的鲁棒 CSV 解析逻辑，返回 DataFrame（解析失败返回 None）。"""
    match = re.search(r"((第\s*\d+\s*集|Episode|镜号).*$)", content, re.DOTALL)
    if not match:
        return None
    csv_text = match.group(1).strip()
    csv_text = re.sub(r"```\w*\n?", "", csv_text).replace("```", "").strip()

    data_rows = []
    reader = csv.reader(csv_text.splitlines())
    for row in reader:
        if not row:
            continue
        row = [str(x).strip() for x in row]
        row_str = "".join(row)

        # 逻辑 A：识别分集标题行
        if (len(row) == 1 or (len(row) < 3 and len(row_str) < 20)) and (
            "集" in row_str or "Episode" in row_str
        ):
            title = row[0].replace(",", "")
            data_rows.append([f"🎬 {title} 🎬", "", "", ""])
            continue

        # 逻辑 B：处理表头
        if "镜号" in row[0]:
            continue

        # 逻辑 C：数据行格式化（智能分离画面与台词）
        processed_row = []
        if len(row) >= 3:
            if len(row) == 3:
                row.append("")
            rest_text = ",".join(row[2:])
            match_dialogue = re.search(
                r'(?:^|[,。！？”\s])\s*([A-Za-z0-9\s\(\)\-]{2,25}:\s*\S)', rest_text
            )
            if match_dialogue:
                idx = match_dialogue.start(1)
                visual_part = rest_text[:idx].strip(' ,"')
                dialogue_part = rest_text[idx:].strip(' ,"')
                processed_row = [row[0], row[1], visual_part, dialogue_part]
            else:
                if len(row) == 4:
                    processed_row = row
                else:
                    processed_row = [row[0], row[1], ",".join(row[2:-1]), row[-1]]
        elif len(row) < 3:
            row.extend([""] * (4 - len(row)))
            processed_row = row

        # 逻辑 E：清洗景别关键词
        if processed_row and len(processed_row) == 4:
            clean_visual = re.sub(r"【.*?】|\[.*?\]", "", processed_row[2]).strip()
            processed_row[2] = clean_visual

        # 逻辑 D：隐式分集检测
        if processed_row and processed_row[0] == "1" and len(data_rows) > 0:
            if "🎬" not in data_rows[-1][0]:
                data_rows.append(["🎬 下一集 / Next Episode 🎬", "", "", ""])

        if processed_row:
            data_rows.append(processed_row)

    header_list = ["镜号", "场景", "画面内容 (Visual)", "台词/解说 (Dialogue/Commentary)"]
    if len(data_rows) > 0:
        return pd.DataFrame(data_rows, columns=header_list)
    return pd.DataFrame(columns=header_list)
