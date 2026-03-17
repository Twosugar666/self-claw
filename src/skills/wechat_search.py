# -*- coding: utf-8 -*-
"""
Skill: wechat.search
微信离线数据检索 — 从用户 03-WeChat 目录中检索笔记、收藏、公众号内容。

参考 internal_ops.search_files 的并发模式重构：
- 并发读取文件（ThreadPoolExecutor）
- 递归扫描目录收集文件列表
- 段落级 + 文件名级关键词匹配
- 返回 agent_context 供 LLM 生成自然语言回复

目录结构:
  03-WeChat/
    笔记/       — 微信笔记导出
    收藏/       — 微信收藏导出
    公众号/     — 公众号文章存档
"""
import os
import sys
from datetime import datetime, timezone, timedelta

BEIJING_TZ = timezone(timedelta(hours=8))


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


# 关键词 → 子目录映射
_CATEGORY_KEYWORDS = {
    "笔记": "笔记",
    "日记": "笔记",
    "记录": "笔记",
    "note": "笔记",
    "收藏": "收藏",
    "收藏夹": "收藏",
    "favorite": "收藏",
    "公众号": "公众号",
    "文章": "公众号",
    "推文": "公众号",
    "article": "公众号",
}


def _detect_category(query: str) -> str:
    """从用户查询中检测微信数据类别，返回子目录名或空字符串（全量搜索）"""
    query_lower = query.lower()
    for keyword, category in _CATEGORY_KEYWORDS.items():
        if keyword in query_lower:
            return category
    return ""


def _collect_file_paths(directories: list) -> dict:
    """递归扫描目录，收集所有 .md/.txt/.html 文件路径。
    返回 dict: {display_key: full_path}
    """
    files_to_read = {}
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        dir_name = os.path.basename(directory)
        for root, dirs, files in os.walk(directory):
            for fname in sorted(files):
                if fname.startswith(".") or fname.startswith("_"):
                    continue
                ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
                if ext not in ("md", "txt", "html"):
                    continue
                fpath = os.path.join(root, fname)
                # key 用于显示来源：子目录名_文件名
                rel = os.path.relpath(fpath, directory)
                key = f"{dir_name}/{rel}"
                files_to_read[key] = fpath
    return files_to_read


def _concurrent_read(files_to_read: dict, io_backend) -> dict:
    """并发读取所有文件，返回 {key: content}"""
    from concurrent.futures import ThreadPoolExecutor
    try:
        from brain import _executor
        executor = _executor
    except Exception:
        executor = ThreadPoolExecutor(max_workers=6)

    futures = {k: executor.submit(io_backend.read_text, v) for k, v in files_to_read.items()}
    results = {}
    for k, fut in futures.items():
        try:
            content = fut.result(timeout=15) or ""
            if content.strip():
                results[k] = content
        except Exception:
            pass
    return results


def _search_in_contents(contents: dict, keywords: list, max_results: int = 15) -> list:
    """在已读取的文件内容中搜索关键词，返回匹配列表。

    匹配策略：
    1. 段落级匹配（精确度高）
    2. 文件名匹配（补充）
    """
    keyword_lower = [kw.lower() for kw in keywords if kw]
    matches = []

    if not keyword_lower:
        # 无关键词 → 返回所有文件摘要
        for source, content in contents.items():
            matches.append({
                "source": source,
                "content": content.strip()[:400],
            })
            if len(matches) >= max_results:
                break
        return matches

    # 已匹配过的文件（避免段落+文件名重复匹配）
    matched_sources = set()

    for source, content in contents.items():
        # 段落级搜索
        paragraphs = content.split("\n\n")
        for para in paragraphs:
            para_stripped = para.strip()
            if not para_stripped:
                continue
            if any(kw in para_stripped.lower() for kw in keyword_lower):
                matches.append({
                    "source": source,
                    "content": para_stripped[:400],
                })
                matched_sources.add(source)
                if len(matches) >= max_results:
                    return matches

    # 文件名级补充匹配
    for source, content in contents.items():
        if source in matched_sources:
            continue
        fname = os.path.basename(source)
        if any(kw in fname.lower() for kw in keyword_lower):
            matches.append({
                "source": source,
                "content": content.strip()[:400],
            })
            if len(matches) >= max_results:
                return matches

    return matches


def execute(params, state, ctx):
    """
    wechat.search — 在微信离线数据中检索内容。

    params:
        query: str — 用户的搜索意图/问题
        keywords: list[str] — 搜索关键词
        category: str — 可选，强制指定类别 (笔记/收藏/公众号)，不传则自动检测
        max_results: int — 最大返回条数（默认 15）
    """
    query = (params.get("query") or "").strip()
    keywords = params.get("keywords") or []
    forced_category = (params.get("category") or "").strip()
    max_results = params.get("max_results", 15)

    if not query and not keywords:
        return {"success": False, "reply": "请告诉我你想查找什么内容~"}

    # 如果没有明确关键词，从 query 中提取
    if not keywords and query:
        _stopwords = {"我", "的", "了", "在", "有", "吗", "呢", "是", "把", "给",
                      "看看", "帮我", "找", "找找", "搜", "搜索", "查", "查找",
                      "微信", "里", "中", "那个", "这个", "之前", "以前"}
        keywords = [w for w in query if len(w) > 1 and w not in _stopwords]
        if not keywords:
            keywords = [query]

    # 检测类别
    category = forced_category or _detect_category(query)

    # 确定搜索目录
    if category:
        category_dir_map = {
            "笔记": ctx.wechat_notes_dir,
            "收藏": ctx.wechat_favorites_dir,
            "公众号": ctx.wechat_articles_dir,
        }
        search_dirs = [category_dir_map.get(category, ctx.wechat_dir)]
        _log(f"[wechat.search] 精确搜索: {category} 目录")
    else:
        search_dirs = [
            ctx.wechat_notes_dir,
            ctx.wechat_favorites_dir,
            ctx.wechat_articles_dir,
        ]
        _log(f"[wechat.search] 全量搜索: 03-WeChat 所有子目录")

    # 收集文件路径（递归）
    files_to_read = _collect_file_paths(search_dirs)
    _log(f"[wechat.search] 扫描到 {len(files_to_read)} 个文件")

    if not files_to_read:
        category_hint = f"「{category}」" if category else "微信"
        return {
            "success": True,
            "reply": f"{category_hint}目录下还没有数据哦~ 你可以把微信的{category_hint}内容导出为 .md 或 .txt 文件放到 03-WeChat 对应目录下。"
        }

    # 并发读取所有文件
    contents = _concurrent_read(files_to_read, ctx.IO)
    _log(f"[wechat.search] 成功读取 {len(contents)} 个文件")

    if not contents:
        return {
            "success": True,
            "reply": "文件读取失败或所有文件都是空的，请检查文件格式是否正确。"
        }

    # 搜索匹配
    matches = _search_in_contents(contents, keywords, max_results)

    if not matches:
        return {
            "success": True,
            "reply": f"在微信数据中没有找到和「{'、'.join(keywords[:3])}」相关的内容。",
            "agent_context": {
                "matches": [],
                "total": 0,
                "total_files": len(contents),
                "keywords": keywords,
                "category": category or "all",
            }
        }

    _log(f"[wechat.search] 找到 {len(matches)} 条匹配 (共 {len(contents)} 个文件)")

    # 构建 agent_context 供 LLM 生成回复
    context_text = ""
    for i, m in enumerate(matches, 1):
        context_text += f"\n--- [{i}] {m['source']} ---\n{m['content']}\n"

    return {
        "success": True,
        "reply": None,
        "agent_context": {
            "matches": matches,
            "total": len(matches),
            "total_files": len(contents),
            "keywords": keywords,
            "category": category or "all",
            "context_text": context_text,
        }
    }


# ============ Skill 热加载注册表 ============
SKILL_REGISTRY = {
    "wechat.search": execute,
}
