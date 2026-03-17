# -*- coding: utf-8 -*-
"""
Skill 自描述装饰器（V11）
每个 handler 通过 @skill(...) 声明完整元数据，实现自注册。
"""

_skill_metadata = {}  # 全局元数据注册表


def skill(
    name,            # skill 名称，如 "todo.add"
    description,     # 给 LLM 看的一句话描述
    params=None,     # 参数 schema {name: "说明", ...}
    rules="",        # 给 LLM 的触发/使用规则（自然语言）
    simple=True,     # 是否为简单 skill（直接返回回复，不走二次加工）
    long=False,      # 是否为长任务（需先发确认消息）
    group="core",    # 所属分组，用于条件注入（core/finance/book/habit/advanced）
    keywords=None,   # 触发关键词，用于 group != "core" 时的条件匹配
):
    """Skill 自描述装饰器，让 handler 自带完整元数据。"""
    def decorator(fn):
        _skill_metadata[name] = {
            "name": name,
            "description": description,
            "params": params or {},
            "rules": rules,
            "simple": simple,
            "long": long,
            "group": group,
            "keywords": keywords or [],
            "handler": fn,
        }
        fn._skill_name = name
        return fn
    return decorator


def get_all_metadata():
    """返回所有已注册 skill 的元数据（dict）"""
    return _skill_metadata


def get_skill_registry():
    """返回 {name: handler} 映射，兼容旧 skill_loader 接口"""
    return {name: meta["handler"] for name, meta in _skill_metadata.items()}


def get_simple_skills():
    """自动生成 _SIMPLE_SKILLS 集合"""
    return frozenset(
        name for name, meta in _skill_metadata.items()
        if meta["simple"]
    )


def get_long_tasks():
    """自动生成 LONG_TASKS 集合"""
    return frozenset(
        name for name, meta in _skill_metadata.items()
        if meta["long"]
    )


def generate_skills_prompt():
    """从装饰器元数据自动生成 SKILLS prompt 文本（与原 prompts.py SKILLS 格式一致）"""
    lines = ["# 可用 Skill（参数均为 JSON）\n"]
    for name in sorted(_skill_metadata.keys()):
        meta = _skill_metadata[name]
        desc = meta["description"]
        if meta["params"]:
            params_str = ", ".join(meta["params"].keys())
            lines.append(f"- **{name}** `{{{params_str}}}` — {desc}")
        else:
            lines.append(f"- **{name}** `{{}}` — {desc}")
    # 追加 ignore
    lines.append("- **ignore** `{reason?}` — 不处理")
    return "\n".join(lines)


def generate_skills_doc():
    """从装饰器元数据自动生成 Skill 文档（Markdown）"""
    lines = ["# Karvis Skills 一览\n"]
    groups = {}
    for name, meta in sorted(_skill_metadata.items()):
        g = meta["group"]
        if g not in groups:
            groups[g] = []
        groups[g].append(meta)

    group_labels = {"core": "核心", "finance": "财务", "book": "读书影视",
                    "habit": "习惯", "advanced": "高级功能"}
    for g in ["core", "finance", "book", "habit", "advanced"]:
        if g not in groups:
            continue
        lines.append(f"## {group_labels.get(g, g)}\n")
        for meta in groups[g]:
            params = ", ".join(meta["params"].keys()) if meta["params"] else "无"
            lines.append(f"### {meta['name']}\n")
            lines.append(f"- **描述**：{meta['description']}")
            lines.append(f"- **参数**：`{params}`")
            flags = []
            if meta["simple"]:
                flags.append("简单回复")
            if meta["long"]:
                flags.append("长任务")
            if flags:
                lines.append(f"- **标记**：{' / '.join(flags)}")
            if meta["rules"].strip():
                lines.append(f"- **触发规则**：\n{meta['rules'].strip()}")
            lines.append("")
    return "\n".join(lines)
