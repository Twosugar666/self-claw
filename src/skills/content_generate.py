# -*- coding: utf-8 -*-
"""
Skill: content_generate (V1)
基于笔记内容的两阶段智能生成 — 用户不只是找内容，而是要基于已有笔记生成新内容。

工作方式（两阶段模型调用）：
1. 用户发送"帮我基于 xxx 写一篇 yyy"/"根据会议记录总结一下"/"把这些笔记整理成文章"
2. LLM 识别为 content.generate，提取 task + source_keywords + output_format
3. Skill 先收集所有相关笔记内容（全量读取，不截断）
4. 第一阶段 — GLM-5-turbo（快模型）：
   - 强制读完所有收集到的内容
   - 理解用户真正想要什么
   - 提取关键信息、结构化要点、标注哪些内容与生成任务相关
5. 第二阶段 — GLM-5（强模型）：
   - 基于第一阶段的理解结果
   - 生成高质量的最终输出
6. 返回生成结果，可选保存到文件

与 deep.dive 的区别：
- deep.dive 是分析型（时间线+趋势+洞察）
- content.generate 是生成型（基于笔记产出新内容）
"""
import sys
import json
import requests
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor

BEIJING_TZ = timezone(timedelta(hours=8))


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def generate(params, state, ctx):
    """
    content.generate — 基于笔记内容进行两阶段智能生成。

    params:
        task: str — 生成任务描述（如"写一篇读书总结"、"整理成会议纪要"）
        source_keywords: list[str] — 源内容搜索关键词
        output_format: str — 期望输出格式（如"文章"、"总结"、"大纲"、"邮件"）
        source_paths: list[str] — 可选，直接指定源文件路径
        save: bool — 是否保存到文件（默认 false）
    """
    task = params.get("task", "")
    source_keywords = params.get("source_keywords", [])
    output_format = params.get("output_format", "文章")
    source_paths = params.get("source_paths", [])
    save = params.get("save", False)

    if not task:
        return {"success": False, "reply": "想让我基于什么内容生成什么呢？描述一下你的需求~"}

    if not source_keywords and not source_paths:
        # 从 task 自动提取关键词
        source_keywords = _extract_keywords(task)
        if not source_keywords:
            return {"success": False, "reply": "没找到相关内容的线索，你能告诉我要基于哪些笔记或话题来生成吗？"}

    _log(f"[content_generate] 开始生成: task={task}, keywords={source_keywords}, format={output_format}")

    # ====== 阶段 0：收集所有相关内容（全量不截断） ======
    raw_content = _collect_all_content(source_keywords, source_paths, state, ctx)
    if not raw_content.get("has_data"):
        return {
            "success": True,
            "reply": f"翻了翻笔记，没找到和「{task}」相关的内容。你可以先记录一些笔记，或者告诉我更具体的关键词~"
        }

    _log(f"[content_generate] 收集到 {raw_content['total_chars']} 字符内容，来源 {len(raw_content['sources'])} 个")

    # ====== 阶段 1：GLM-5-turbo 理解阶段 ======
    understanding = _phase1_understand(task, output_format, raw_content)
    if not understanding:
        return {"success": True, "reply": "理解内容时出了点问题，稍后再试试~"}

    _log(f"[content_generate] 阶段1完成: 理解结果 {len(understanding)} 字符")

    # ====== 阶段 2：GLM-5 生成阶段 ======
    result = _phase2_generate(task, output_format, understanding, raw_content)

    # GLM-5 结果为空/无效 → 用 GLM-5-turbo 兜底重新生成（不走"好的～"兜底）
    if not result or len(result.strip()) < 10:
        _log(f"[content_generate] GLM-5 生成结果为空或过短，启动 GLM-5-turbo 兜底生成")
        result = _phase2_fallback_turbo(task, output_format, understanding, raw_content)
        if not result:
            return {"success": True, "reply": "内容生成遇到了困难，两个模型都没能给出满意的结果。你可以换个说法再试试~"}
        _log(f"[content_generate] GLM-5-turbo 兜底生成完成: {len(result)} 字符")
    else:
        _log(f"[content_generate] 阶段2完成: 生成结果 {len(result)} 字符")

    # 可选保存
    if save:
        _save_result(task, output_format, result, ctx)

    return {"success": True, "reply": result}


def _extract_keywords(task):
    """从任务描述中自动提取搜索关键词"""
    # 去除常见的生成动词和格式词
    stop_words = {"帮我", "帮", "我", "写", "生成", "整理", "总结", "基于", "根据",
                  "一篇", "一份", "一个", "把", "这些", "那些", "所有", "的",
                  "文章", "报告", "大纲", "邮件", "总结", "纪要", "成", "为",
                  "给我", "做", "出来", "下", "一下"}
    words = []
    for w in task.replace("，", " ").replace("、", " ").replace("。", " ").split():
        w = w.strip()
        if len(w) >= 2 and w not in stop_words:
            words.append(w)
    return words[:5] if words else []


def _collect_all_content(keywords, source_paths, state, ctx):
    """
    收集所有相关笔记内容 — 全量读取，不截断。
    与 deep_dive 不同，这里尽可能多读取完整内容供后续生成使用。
    """
    try:
        from brain import _executor
        executor = _executor
    except Exception:
        executor = ThreadPoolExecutor(max_workers=6)

    files_to_read = {}

    # 直接指定的路径
    for p in source_paths[:10]:
        if p.startswith("/"):
            if p.startswith(ctx.base_dir):
                files_to_read[f"direct_{p}"] = p
        else:
            files_to_read[f"direct_{p}"] = f"{ctx.base_dir}/{p}"

    # 通过关键词搜索的文件
    # 核心规则：用户说"笔记""日记"时，默认数据源是 03-WeChat
    import os
    _note_diary_kw = {"笔记", "日记"}
    _is_note_diary = any(kw in " ".join(keywords) for kw in _note_diary_kw)

    if _is_note_diary:
        # "笔记""日记" → 优先全量扫描 03-WeChat（主数据源）
        _log("[content_generate] 检测到'笔记/日记'关键词，优先扫描 03-WeChat")
        for subdir_name, subdir_path in [
            ("wechat_notes", getattr(ctx, 'wechat_notes_dir', '')),
            ("wechat_favorites", getattr(ctx, 'wechat_favorites_dir', '')),
            ("wechat_articles", getattr(ctx, 'wechat_articles_dir', '')),
        ]:
            if subdir_path and os.path.isdir(subdir_path):
                for fname in os.listdir(subdir_path):
                    if fname.startswith(".") or fname.startswith("_"):
                        continue
                    ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
                    if ext in ("md", "txt", "html"):
                        files_to_read[f"{subdir_name}_{fname}"] = os.path.join(subdir_path, fname)
        # 也扫描 wechat_dir 根目录下散落的文件
        wechat_root = getattr(ctx, 'wechat_dir', '')
        if wechat_root and os.path.isdir(wechat_root):
            for fname in os.listdir(wechat_root):
                fpath = os.path.join(wechat_root, fname)
                if os.path.isfile(fpath) and not fname.startswith(".") and not fname.startswith("_"):
                    ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
                    if ext in ("md", "txt", "html"):
                        files_to_read[f"wechat_root_{fname}"] = fpath
        # 02-Notes 作为补充（仅 quick_notes + memory）
        files_to_read["quick_notes"] = ctx.quick_notes_file
        files_to_read["memory"] = ctx.memory_file
    else:
        # 非"笔记/日记"场景 → 全量扫描所有目录
        files_to_read["quick_notes"] = ctx.quick_notes_file
        files_to_read["misc"] = ctx.misc_file
        files_to_read["memory"] = ctx.memory_file

        # 最近 30 天的归档笔记
        today = datetime.now(BEIJING_TZ).date()
        for i in range(30):
            d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            files_to_read[f"emotion_{d}"] = f"{ctx.emotion_notes_dir}/{d}.md"
            files_to_read[f"work_{d}"] = f"{ctx.work_notes_dir}/{d}.md"
            files_to_read[f"fun_{d}"] = f"{ctx.fun_notes_dir}/{d}.md"

        # 也搜索 WeChat 目录
        for subdir_name, subdir_path in [
            ("wechat_notes", getattr(ctx, 'wechat_notes_dir', '')),
            ("wechat_favorites", getattr(ctx, 'wechat_favorites_dir', '')),
            ("wechat_articles", getattr(ctx, 'wechat_articles_dir', '')),
        ]:
            if subdir_path and os.path.isdir(subdir_path):
                for fname in os.listdir(subdir_path):
                    if fname.startswith(".") or fname.startswith("_"):
                        continue
                    ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
                    if ext in ("md", "txt", "html"):
                        files_to_read[f"{subdir_name}_{fname}"] = os.path.join(subdir_path, fname)

    # 并发读取
    futures = {k: executor.submit(ctx.IO.read_text, v) for k, v in files_to_read.items()}
    results = {}
    for k, fut in futures.items():
        try:
            results[k] = fut.result(timeout=30) or ""
        except Exception:
            results[k] = ""

    # 按关键词过滤，但保留完整段落/完整文件（不截断到 200 字）
    keyword_lower = [kw.lower() for kw in keywords]
    matched_blocks = []
    sources = set()

    for source_key, text in results.items():
        if not text:
            continue

        # 直接指定的路径 — 全量读取
        if source_key.startswith("direct_"):
            matched_blocks.append({
                "source": source_key.replace("direct_", ""),
                "content": text,
                "relevance": "direct"
            })
            sources.add(source_key)
            continue

        # 检查整个文件是否包含关键词
        text_lower = text.lower()
        if not any(kw in text_lower for kw in keyword_lower):
            continue

        # 按段落分割，保留包含关键词的段落及其上下文
        paragraphs = text.split("\n\n")
        relevant_paragraphs = []
        for idx, para in enumerate(paragraphs):
            if any(kw in para.lower() for kw in keyword_lower):
                # 保留上下文段落
                start = max(0, idx - 1)
                end = min(len(paragraphs), idx + 2)
                for j in range(start, end):
                    if paragraphs[j] not in relevant_paragraphs:
                        relevant_paragraphs.append(paragraphs[j])

        if relevant_paragraphs:
            matched_blocks.append({
                "source": _friendly_source_name(source_key),
                "content": "\n\n".join(relevant_paragraphs),
                "relevance": "keyword"
            })
            sources.add(source_key)

    total_chars = sum(len(b["content"]) for b in matched_blocks)

    return {
        "has_data": bool(matched_blocks),
        "blocks": matched_blocks,
        "total_chars": total_chars,
        "sources": list(sources),
    }


def _friendly_source_name(key):
    """将内部 key 转换为友好的来源名称"""
    if key == "quick_notes":
        return "速记"
    if key == "misc":
        return "碎碎念"
    if key == "memory":
        return "长期记忆"
    if key.startswith("emotion_"):
        return f"情感日记({key[8:]})"
    if key.startswith("work_"):
        return f"工作笔记({key[5:]})"
    if key.startswith("fun_"):
        return f"生活趣事({key[4:]})"
    if key.startswith("wechat_notes_"):
        return f"微信笔记({key[13:]})"
    if key.startswith("wechat_favorites_"):
        return f"微信收藏({key[17:]})"
    if key.startswith("wechat_articles_"):
        return f"公众号文章({key[16:]})"
    return key


def _phase1_understand(task, output_format, raw_content):
    """
    第一阶段：GLM-5-turbo 理解用户需求，强制读完所有内容。

    目标：
    1. 完整理解用户想要什么
    2. 读完全部源内容，标注重点
    3. 提取结构化要点供第二阶段使用
    """
    from config import LLM2_API_KEY, LLM2_BASE_URL, LLM2_MODEL
    from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

    # 决定使用哪个模型做理解阶段
    # 优先用 LLM2（GLM-5-turbo），不可用则降级到主模型
    if LLM2_API_KEY and LLM2_BASE_URL and LLM2_MODEL:
        api_key = LLM2_API_KEY
        base_url = LLM2_BASE_URL
        model = LLM2_MODEL
        tier_label = "Phase1-Turbo"
    else:
        api_key = DEEPSEEK_API_KEY
        base_url = DEEPSEEK_BASE_URL
        model = DEEPSEEK_MODEL
        tier_label = "Phase1-Main"

    _log(f"[content_generate][{tier_label}] 使用模型: {model}")

    # 组装所有源内容（不截断，让模型全部读完）
    content_parts = []
    for block in raw_content["blocks"]:
        content_parts.append(f"=== 来源：{block['source']} ===\n{block['content']}")
    all_content = "\n\n".join(content_parts)

    # 如果内容超过模型上下文窗口（保守估计 28K 字符 ≈ 7K token），分批处理
    MAX_CHARS_PER_CALL = 28000
    if len(all_content) > MAX_CHARS_PER_CALL:
        return _phase1_chunked(task, output_format, raw_content, api_key, base_url, model, tier_label)

    import prompts
    system_prompt = prompts.CONTENT_GEN_PHASE1_SYSTEM
    user_prompt = prompts.get(
        "CONTENT_GEN_PHASE1_USER",
        task=task,
        output_format=output_format,
        source_count=len(raw_content["sources"]),
        total_chars=raw_content["total_chars"],
        all_content=all_content,
    )

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 2000,
        "temperature": 0.3
    }

    import time as _time
    t0 = _time.time()
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=90)
        t1 = _time.time()
        if resp.status_code == 200:
            result = resp.json()
            usage = result.get("usage", {})
            understanding = result["choices"][0]["message"]["content"].strip()
            _log(f"[content_generate][{tier_label}] 理解完成: {t1-t0:.1f}s, "
                 f"prompt_tokens={usage.get('prompt_tokens')}, "
                 f"completion_tokens={usage.get('completion_tokens')}, "
                 f"output={len(understanding)}字符")

            # 记录用量
            try:
                from brain import _log_llm_usage
                _log_llm_usage("phase1", model, usage, t1 - t0)
            except Exception:
                pass

            return understanding
        _log(f"[content_generate][{tier_label}] API 错误: {resp.status_code} - {resp.text[:200]}")
    except Exception as e:
        _log(f"[content_generate][{tier_label}] 异常: {e}")
    return None


def _phase1_chunked(task, output_format, raw_content, api_key, base_url, model, tier_label):
    """
    当内容超出单次 context 窗口时，分块让模型理解。
    每块读完后提取要点，最后合并。
    """
    import prompts
    import time as _time

    MAX_CHARS = 28000
    blocks = raw_content["blocks"]
    chunk_understandings = []

    # 将 blocks 分成不超过 MAX_CHARS 的 chunk
    current_chunk = []
    current_size = 0
    chunks = []

    for block in blocks:
        block_text = f"=== 来源：{block['source']} ===\n{block['content']}"
        if current_size + len(block_text) > MAX_CHARS and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [block_text]
            current_size = len(block_text)
        else:
            current_chunk.append(block_text)
            current_size += len(block_text)
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    _log(f"[content_generate][{tier_label}] 内容分为 {len(chunks)} 批次处理")

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for i, chunk in enumerate(chunks):
        user_prompt = prompts.get(
            "CONTENT_GEN_PHASE1_CHUNK_USER",
            task=task,
            output_format=output_format,
            chunk_index=i + 1,
            total_chunks=len(chunks),
            chunk_content=chunk,
        )

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": prompts.CONTENT_GEN_PHASE1_SYSTEM},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 1500,
            "temperature": 0.3
        }

        t0 = _time.time()
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=90)
            t1 = _time.time()
            if resp.status_code == 200:
                result = resp.json()
                understanding = result["choices"][0]["message"]["content"].strip()
                chunk_understandings.append(f"[批次 {i+1}/{len(chunks)}]\n{understanding}")
                _log(f"[content_generate][{tier_label}] 批次 {i+1}/{len(chunks)} 完成: {t1-t0:.1f}s")

                try:
                    from brain import _log_llm_usage
                    _log_llm_usage("phase1-chunk", model, result.get("usage", {}), t1 - t0)
                except Exception:
                    pass
            else:
                _log(f"[content_generate][{tier_label}] 批次 {i+1} API 错误: {resp.status_code}")
        except Exception as e:
            _log(f"[content_generate][{tier_label}] 批次 {i+1} 异常: {e}")

    if not chunk_understandings:
        return None

    return "\n\n".join(chunk_understandings)


def _phase2_generate(task, output_format, understanding, raw_content):
    """
    第二阶段：GLM-5（强模型）基于理解结果生成最终输出。

    输入：第一阶段的结构化理解 + 用户原始需求
    输出：高质量的生成内容
    """
    from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

    import prompts
    import time as _time

    # 组装 source 列表供参考
    source_list = ", ".join(set(b["source"] for b in raw_content["blocks"]))

    user_prompt = prompts.get(
        "CONTENT_GEN_PHASE2_USER",
        task=task,
        output_format=output_format,
        source_list=source_list,
        total_chars=raw_content["total_chars"],
        understanding=understanding,
    )

    url = f"{DEEPSEEK_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": prompts.CONTENT_GEN_PHASE2_SYSTEM},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 2000,
        "temperature": 0.5
    }

    t0 = _time.time()
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=120)
        t1 = _time.time()
        if resp.status_code == 200:
            result = resp.json()
            usage = result.get("usage", {})
            output = result["choices"][0]["message"]["content"].strip()
            _log(f"[content_generate][Phase2] 生成完成: {t1-t0:.1f}s, "
                 f"prompt_tokens={usage.get('prompt_tokens')}, "
                 f"completion_tokens={usage.get('completion_tokens')}, "
                 f"output={len(output)}字符")

            try:
                from brain import _log_llm_usage
                _log_llm_usage("phase2", DEEPSEEK_MODEL, usage, t1 - t0)
            except Exception:
                pass

            return output
        _log(f"[content_generate][Phase2] API 错误: {resp.status_code} - {resp.text[:200]}")
    except Exception as e:
        _log(f"[content_generate][Phase2] 异常: {e}")
    return None


def _phase2_fallback_turbo(task, output_format, understanding, raw_content):
    """
    GLM-5 生成结果为空时的兜底：用 GLM-5-turbo 重新理解并直接生成。
    不走"好的～"兜底，而是让快速模型自己完成最终生成。
    """
    from config import LLM2_API_KEY, LLM2_BASE_URL, LLM2_MODEL
    from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

    # 选模型：优先 LLM2（GLM-5-turbo），不可用则用主模型重试
    if LLM2_API_KEY and LLM2_BASE_URL and LLM2_MODEL:
        api_key = LLM2_API_KEY
        base_url = LLM2_BASE_URL
        model = LLM2_MODEL
        tier_label = "Phase2-Fallback-Turbo"
    else:
        api_key = DEEPSEEK_API_KEY
        base_url = DEEPSEEK_BASE_URL
        model = DEEPSEEK_MODEL
        tier_label = "Phase2-Fallback-Main"

    _log(f"[content_generate][{tier_label}] GLM-5 生成为空，启动兜底: {model}")

    import prompts
    import time as _time

    source_list = ", ".join(set(b["source"] for b in raw_content["blocks"]))

    # 用 Phase2 的 prompt，但交给 turbo 模型来做
    user_prompt = prompts.get(
        "CONTENT_GEN_PHASE2_USER",
        task=task,
        output_format=output_format,
        source_list=source_list,
        total_chars=raw_content["total_chars"],
        understanding=understanding,
    )

    # 补充提示：告诉 turbo 这是兜底生成，需要更认真
    fallback_system = prompts.CONTENT_GEN_PHASE2_SYSTEM + "\n\n注意：前一个模型未能成功生成内容，你是兜底模型。请务必认真生成有价值的内容，不要返回空内容。"

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": fallback_system},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 2500,
        "temperature": 0.6
    }

    t0 = _time.time()
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=120)
        t1 = _time.time()
        if resp.status_code == 200:
            result = resp.json()
            usage = result.get("usage", {})
            output = result["choices"][0]["message"]["content"].strip()
            _log(f"[content_generate][{tier_label}] 兜底生成完成: {t1-t0:.1f}s, "
                 f"prompt_tokens={usage.get('prompt_tokens')}, "
                 f"completion_tokens={usage.get('completion_tokens')}, "
                 f"output={len(output)}字符")

            try:
                from brain import _log_llm_usage
                _log_llm_usage("phase2-fallback", model, usage, t1 - t0)
            except Exception:
                pass

            # 二次检查：如果 turbo 也返回空，则确实失败
            if output and len(output.strip()) >= 10:
                return output
            _log(f"[content_generate][{tier_label}] 兜底模型也返回了空/过短内容")
        else:
            _log(f"[content_generate][{tier_label}] API 错误: {resp.status_code} - {resp.text[:200]}")
    except Exception as e:
        _log(f"[content_generate][{tier_label}] 异常: {e}")
    return None


def _save_result(task, output_format, content, ctx):
    """将生成结果保存到文件"""
    now = datetime.now(BEIJING_TZ)
    date_str = now.strftime("%Y-%m-%d")
    safe_task = task.replace("/", "-").replace("\\", "-").replace(" ", "")[:20]
    file_path = f"{ctx.base_dir}/02-Notes/生成内容/{date_str}-{safe_task}.md"

    md_content = f"""---
date: {date_str}
type: content-generate
task: {task}
format: {output_format}
tags: [generated]
---

{content}
"""
    try:
        ok = ctx.IO.write_text(file_path, md_content)
        if ok:
            _log(f"[content_generate] 结果已保存: {file_path}")
        else:
            _log(f"[content_generate] 保存失败: {file_path}")
    except Exception as e:
        _log(f"[content_generate] 保存异常: {e}")


# ============ Skill 热加载注册表 ============
SKILL_REGISTRY = {
    "content.generate": generate,
}
