# -*- coding: utf-8 -*-
"""本地 Skill 选择、显示名管理与对话参数节点。"""
from __future__ import annotations

import ast
import asyncio
import json
import re

from .skill_runtime import (
    get_skill,
    list_skills,
    read_reference,
    read_skill,
    resolve_skill_id,
    set_model_display_names,
    skill_catalog,
)


CATEGORY = "🍭大炮-llama-cpp"


def _structured_values_from_text(text: str) -> list:
    value = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", value, re.S | re.I)
    candidates = [fenced.group(1), value] if fenced else [value]
    decoder = json.JSONDecoder()
    values = []
    for candidate in candidates:
        variants = [candidate, candidate.translate(str.maketrans({"“": '"', "”": '"', "：": ":"}))]
        for variant in variants:
            for start in (match.start() for match in re.finditer(r"[\[{]", variant)):
                try:
                    parsed, _end = decoder.raw_decode(variant[start:])
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, (dict, list)) and parsed not in values:
                    values.append(parsed)
            try:
                parsed = ast.literal_eval(variant.strip())
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, (dict, list)) and parsed not in values:
                values.append(parsed)
    return values


def _name_pairs_from_value(value) -> list[tuple[str, str]]:
    if isinstance(value, list):
        pairs = []
        for item in value:
            pairs.extend(_name_pairs_from_value(item))
        return pairs
    if not isinstance(value, dict):
        return []
    if "names" in value:
        return _name_pairs_from_value(value["names"])
    skill_id = value.get("id") or value.get("skill_id") or value.get("skill-id")
    display_name = value.get("display_name") or value.get("display-name") or value.get("title")
    if skill_id and not display_name and value.get("name") != skill_id:
        display_name = value.get("name")
    if isinstance(skill_id, str) and isinstance(display_name, str):
        return [(skill_id, display_name)]
    return [(str(key), item) for key, item in value.items() if isinstance(item, str)]


def _clean_model_name(value: str, skill_id: str) -> str:
    name = str(value or "").strip().strip("`|,，;；").strip('"\'')
    name = re.sub(r"^\*\*(.*?)\*\*$", r"\1", name).strip()
    name = re.sub(rf"\s*[\[（(]{re.escape(skill_id)}[\]）)]\s*$", "", name, flags=re.I).strip()
    if not re.search(r"[\u3400-\u9fff]", name) or not 1 <= len(name) <= 60:
        return ""
    if re.search(r"[\r\n]", name) or (skill_id and skill_id.lower() in name.lower()):
        return ""
    return name


def _skill_names_from_reply(text: str, skill_ids: list[str]) -> dict[str, str]:
    allowed = set(skill_ids)
    names: dict[str, str] = {}
    for value in _structured_values_from_text(text):
        for skill_id, raw_name in _name_pairs_from_value(value):
            skill_id = str(skill_id).strip()
            if skill_id in allowed:
                name = _clean_model_name(raw_name, skill_id)
                if name:
                    names[skill_id] = name
    unkeyed = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```") or re.fullmatch(r"[|:\-\s]+", line):
            continue
        matched = False
        for skill_id in sorted(allowed, key=len, reverse=True):
            if skill_id in names:
                continue
            position = line.find(skill_id)
            if position < 0:
                continue
            tail = line[position + len(skill_id):]
            tail = re.sub(r"^[\s\]）)'\"`|:=：—–-]+", "", tail)
            tail = re.sub(r"[|,，}\]]+\s*$", "", tail).strip()
            name = _clean_model_name(tail, skill_id)
            if name:
                names[skill_id] = name
                matched = True
                break
        if matched:
            continue
        candidate = re.sub(r"^\s*(?:[-*•]|\d+[.)、])\s*", "", line)
        candidate = _clean_model_name(candidate, "")
        if candidate and len(candidate) <= 32:
            unkeyed.append(candidate)
    unresolved = [skill_id for skill_id in skill_ids if skill_id not in names]
    if unresolved and len(unkeyed) == len(skill_ids):
        for skill_id, name in zip(skill_ids, unkeyed):
            names.setdefault(skill_id, name)
    return names


def _optimize_display_names(model, scope: str, selected_id: str) -> dict:
    from .multi_turn_chat_node import (
        _clean_reply,
        _create_completion,
        _extract_reply,
        _release_model,
        _should_unload,
        _sync_model,
        _tokens,
    )

    all_skills = list(list_skills())
    candidates = []
    for item in all_skills:
        if item.get("display_source") == "manual":
            continue
        if scope == "selected" and item["id"] != selected_id:
            continue
        candidates.append({
            "id": item["id"],
            "source_name": item.get("source_name") or item["id"],
            "current_name": item.get("display_name") or item["id"],
            "function_description": str(item.get("description") or "")[:500],
        })
    if scope == "selected" and not selected_id:
        raise ValueError("请先选择一个具体Skill，再优化当前技能名称。")
    if not candidates:
        return {"requested": 0, "updated": 0, "catalog": skill_catalog()}

    synced = None
    try:
        synced = _sync_model(model)
        system = (
            "你是Skill界面命名整理器。必须根据每项function_description概括技能真实用途，"
            "不能只翻译ID或原名称。只优化用户可见的简体中文显示名，不修改Skill功能。"
            "名称准确简洁，建议4到16个中文字符；所有名称不得重复，不含Skill ID、路径、括号说明或宣传口号。"
            "每行只返回：skill-id、一个制表符、中文显示名。不要标题、解释、编号、Markdown或代码块。"
        )
        n_ctx = max(1024, int(synced.settings.get("n_ctx", 8192)))
        output_tokens = min(4096, max(256, len(candidates) * 32), max(256, n_ctx // 3))
        description_limit = 500
        while True:
            packed = [
                {**item, "function_description": item["function_description"][:description_limit]}
                for item in candidates
            ]
            user = "请整理以下Skill显示名称：\n" + json.dumps(packed, ensure_ascii=False, separators=(",", ":"))
            required = _tokens(synced.llm, system) + _tokens(synced.llm, user) + output_tokens + 128
            if required <= n_ctx or description_limit <= 32:
                break
            description_limit = max(32, description_limit // 2)
        if required > n_ctx:
            raise RuntimeError(
                f"当前{len(candidates)}个Skill的命名请求无法放入{n_ctx}上下文；"
                "请提高本地模型上下文长度，或先使用“优化当前技能”。"
            )
        response, _ = _create_completion(
            synced.llm,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            {
                "max_tokens": output_tokens,
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 20,
                "repeat_penalty": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "seed": 0,
                "stream": False,
            },
        )
        reply = _clean_reply(_extract_reply(response))
        candidate_ids = [item["id"] for item in candidates]
        proposed = _skill_names_from_reply(reply, candidate_ids)
        reserved = {item["display_name"] for item in all_skills if item["id"] not in candidate_ids}
        cleaned = {}
        for skill_id in candidate_ids:
            name = proposed.get(skill_id)
            if name and name not in reserved:
                cleaned[skill_id] = name
                reserved.add(name)
        if not cleaned:
            raise RuntimeError("本地模型返回内容无法匹配任何Skill ID或中文显示名；请重试或手动修改。")
        return {
            "requested": len(candidates),
            "updated": len(cleaned),
            "catalog": set_model_display_names(cleaned, overwrite_manual=False),
        }
    finally:
        if _should_unload(model):
            _release_model(model)


class DapaoSkillLoader:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("DAPAO_SKILL_CONFIG",)
    RETURN_NAMES = ("🧩Skill配置",)
    FUNCTION = "load_skill"
    OUTPUT_NODE = True
    DESCRIPTION = "扫描、安装和管理本地Skills；支持稳定ID、手动显示名和本地模型一键优化名称。"

    @classmethod
    def INPUT_TYPES(cls):
        skills = list_skills()
        choices = ["自动选择"] + [item["label"] for item in skills]
        return {
            "required": {
                "🧩Skill选择": (choices, {
                    "default": "自动选择",
                    "tooltip": "自动选择会额外进行一次本地模型路由；手动选择可避免这次推理。",
                }),
                "🧭管理动作": ("STRING", {"default": "idle"}),
                "🎯管理Skill": ("STRING", {"default": ""}),
                "🆔管理请求": ("STRING", {"default": ""}),
            },
            "optional": {
                "🤖本地模型": ("DAPAO_LOCAL_MODEL", {
                    "tooltip": "仅供“优化Skill显示名”按钮使用；手动改名和上传不需要连接。",
                }),
            },
        }

    async def load_skill(self, **kwargs):
        return await asyncio.to_thread(self._load_sync, **kwargs)

    def _load_sync(self, **kwargs):
        selected = str(kwargs.get("🧩Skill选择") or "自动选择")
        selected_id = resolve_skill_id(selected)
        if selected not in ("自动选择", "自动匹配") and not selected_id:
            raise ValueError("所选 Skill 不存在，请刷新节点后重试。")
        action = str(kwargs.get("🧭管理动作") or "idle").strip().lower()
        management = {"action": action, "requested": 0, "updated": 0}
        if action in {"optimize_selected", "optimize_all"}:
            model = kwargs.get("🤖本地模型")
            if model is None:
                raise ValueError("请先把“大炮本地模型加载器”连接到Skill加载器的“🤖本地模型”接口。")
            target = resolve_skill_id(kwargs.get("🎯管理Skill") or selected_id)
            management = _optimize_display_names(
                model,
                "selected" if action == "optimize_selected" else "all",
                target,
            )
            management["action"] = action
        skills = list(list_skills())
        result = {"selected": selected_id, "skills": skills, "version": 3}
        return {
            "ui": {"🛠️Skill管理结果": [json.dumps(management, ensure_ascii=False, separators=(",", ":"))]},
            "result": (result,),
        }


class DapaoChatSettings:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("DAPAO_CHAT_SETTINGS",)
    RETURN_NAMES = ("⚙️对话设置",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "📝系统提示词": ("STRING", {"default": "", "multiline": True, "placeholder": "可选。连接 Skill 时由 Skill 负责系统规则。"}),
            "🔢最大历史轮数": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1}),
            "📤最大生成token": ("INT", {"default": 1024, "min": 20, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
            "🌡️温度": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
            "🎯top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
            "🔝top_k": ("INT", {"default": 20, "min": 0, "max": 200, "step": 1}),
            "🔁重复惩罚": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
            "📈频率惩罚": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
            "📍存在惩罚": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
            "🎲随机种子": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1, "control_after_generate": True}),
            "🧠输出think块": ("BOOLEAN", {"default": False}),
            "📐图片最大边长": ("INT", {"default": 2048, "min": 128, "max": 2048, "step": 64, "tooltip": "图片在请求边界逐张Lanczos缩放并以PNG发送；默认2K。"}),
        }}

    def run(self, **kwargs):
        return ({
            "系统提示词": str(kwargs.get("📝系统提示词") or ""),
            "最大历史轮数": int(kwargs.get("🔢最大历史轮数", 100)),
            "最大生成token": int(kwargs.get("📤最大生成token", 1024)),
            "温度": float(kwargs.get("🌡️温度", 0.7)),
            "top_p": float(kwargs.get("🎯top_p", 0.9)),
            "top_k": int(kwargs.get("🔝top_k", 20)),
            "重复惩罚": float(kwargs.get("🔁重复惩罚", 1.0)),
            "频率惩罚": float(kwargs.get("📈频率惩罚", 0.0)),
            "存在惩罚": float(kwargs.get("📍存在惩罚", 0.0)),
            "seed": int(kwargs.get("🎲随机种子", 0)),
            "输出think块": bool(kwargs.get("🧠输出think块", False)),
            "最大边长": min(2048, max(128, int(kwargs.get("📐图片最大边长", 2048)))),
        },)


NODE_CLASS_MAPPINGS = {"DapaoSkillLoader": DapaoSkillLoader, "DapaoChatSettings": DapaoChatSettings}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DapaoSkillLoader": "🧩大炮Skill加载器@炮老师的小课堂",
    "DapaoChatSettings": "⚙️对话增强设置@炮老师的小课堂",
}
NODE_CLASS_MAPPINGS_SKILL = NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS_SKILL = NODE_DISPLAY_NAME_MAPPINGS

__all__ = [
    "DapaoSkillLoader",
    "DapaoChatSettings",
    "get_skill",
    "list_skills",
    "read_reference",
    "read_skill",
    "NODE_CLASS_MAPPINGS_SKILL",
    "NODE_DISPLAY_NAME_MAPPINGS_SKILL",
]
