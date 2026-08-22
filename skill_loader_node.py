# -*- coding: utf-8 -*-
"""本地 Skill 目录索引。Skill 文件由使用者放入插件根目录的 skills/ 下。"""
from __future__ import annotations

import os
import re


SKILLS_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "skills")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_TEXT_EXTENSIONS = {".md", ".txt", ".yaml", ".yml", ".json"}
_CHINESE_RE = re.compile(r"[\u3400-\u9fff]")


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return handle.read()


def _frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    values: dict[str, str] = {}
    lines = text[3:end].splitlines()
    index = 0
    while index < len(lines):
        match = re.match(r"^([A-Za-z0-9_-]+):\s*(.*?)\s*$", lines[index])
        if not match:
            index += 1
            continue
        key, value = match.groups()
        if value in ("|", ">"):
            index += 1
            parts = []
            while index < len(lines) and (not lines[index].strip() or lines[index][:1].isspace()):
                parts.append(lines[index].strip())
                index += 1
            values[key] = " ".join(part for part in parts if part)
            continue
        values[key] = value.strip("\"'")
        index += 1
    return values


def _meta(skill_dir: str, key: str) -> str:
    path = os.path.join(skill_dir, "meta.yaml")
    if not os.path.isfile(path):
        return ""
    for line in _read_text(path).splitlines():
        match = re.match(rf"^{re.escape(key)}:\s*(.*?)\s*$", line)
        if match:
            return match.group(1).strip("\"'")
    return ""


def _heading(text: str, chinese_only: bool = False) -> str:
    for line in text.splitlines():
        match = re.match(r"^\s*#\s+(.+?)\s*$", line)
        if not match:
            continue
        value = re.sub(r"[`*_]", "", match.group(1)).strip()
        if value and (not chinese_only or _CHINESE_RE.search(value)):
            return value
    return ""


def _references(skill_dir: str) -> list[str]:
    root = os.path.join(skill_dir, "references")
    found: list[str] = []
    if not os.path.isdir(root):
        return found
    for current, _, names in os.walk(root):
        for name in names:
            if os.path.splitext(name)[1].lower() not in _TEXT_EXTENSIONS:
                continue
            found.append(os.path.relpath(os.path.join(current, name), skill_dir).replace("\\", "/"))
    return sorted(found)


def list_skills() -> tuple[dict, ...]:
    os.makedirs(SKILLS_ROOT, exist_ok=True)
    result: list[dict] = []
    for skill_id in sorted(os.listdir(SKILLS_ROOT)):
        if not _ID_RE.fullmatch(skill_id):
            continue
        skill_dir = os.path.join(SKILLS_ROOT, skill_id)
        if not os.path.isdir(skill_dir):
            continue
        candidates = ["SKILL.cn.md", "SKILL.md"]
        skill_file = next((name for name in candidates if os.path.isfile(os.path.join(skill_dir, name))), None)
        if not skill_file:
            continue
        body = _read_text(os.path.join(skill_dir, skill_file))
        metadata = _frontmatter(body)
        name = (
            _meta(skill_dir, "display-name-zh")
            or metadata.get("display-name-zh")
            or metadata.get("name-zh")
            or _heading(body, chinese_only=True)
            or _meta(skill_dir, "name")
            or metadata.get("name")
            or _heading(body)
            or skill_id
        )
        description = (
            _meta(skill_dir, "summary-cn")
            or metadata.get("summary-cn")
            or _meta(skill_dir, "description")
            or metadata.get("description")
            or ""
        )
        result.append({
            "id": skill_id,
            "name": name[:100],
            "label": f"{name} [{skill_id}]" if name != skill_id else skill_id,
            "description": description[:500],
            "skill_file": skill_file,
            "references": _references(skill_dir),
        })
    return tuple(result)


def refresh_skills() -> None:
    # 保留公开函数，方便前端或后续节点主动刷新目录。
    return None


def get_skill(skill_id: str) -> dict | None:
    return next((item for item in list_skills() if item["id"] == skill_id), None)


def read_skill(skill: dict) -> str:
    path = os.path.join(SKILLS_ROOT, skill["id"], skill["skill_file"])
    return _read_text(path)


def read_reference(skill: dict, relative_path: str) -> str:
    normalized = str(relative_path or "").replace("\\", "/").strip("/")
    if normalized not in skill.get("references", []):
        raise ValueError(f"Skill资料不存在：{normalized}")
    root = os.path.realpath(os.path.join(SKILLS_ROOT, skill["id"]))
    path = os.path.realpath(os.path.join(root, normalized))
    if os.path.commonpath([root, path]) != root:
        raise ValueError("Skill资料路径无效。")
    return _read_text(path)


class DapaoSkillLoader:
    CATEGORY = "🍭大炮-llama-cpp"
    RETURN_TYPES = ("DAPAO_SKILL_CONFIG",)
    RETURN_NAMES = ("🧩Skill配置",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        skills = list_skills()
        choices = ["自动选择"] + [item["label"] for item in skills]
        return {"required": {"🧩Skill选择": (choices, {
            "default": "自动选择",
            "tooltip": "自动选择会让本地模型根据首次任务挑选唯一匹配的 Skill；也可以固定选择一个 Skill。",
        })}}

    def run(self, **kwargs):
        selected = str(kwargs.get("🧩Skill选择") or "自动选择")
        selected_id = ""
        if selected not in ("自动选择", "自动匹配"):
            selected_id = next(
                (item["id"] for item in list_skills() if item["label"] == selected or item["id"] == selected),
                "",
            )
            if not selected_id:
                raise ValueError("所选 Skill 不存在，请刷新节点后重试。")
        return ({"selected": selected_id, "skills": list(list_skills()), "version": 1},)


class DapaoChatSettings:
    CATEGORY = "🍭大炮-llama-cpp"
    RETURN_TYPES = ("DAPAO_CHAT_SETTINGS",)
    RETURN_NAMES = ("⚙️对话设置",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "📝系统提示词": ("STRING", {"default": "", "multiline": True, "placeholder": "可选。连接 Skill 时由 Skill 负责系统规则。"}),
            "🔢最大历史轮数": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1}),
            "📤最大生成token": ("INT", {"default": 1024, "min": 20, "max": 0xFFFFFFFFFFFFFFFF, "step": 1, "tooltip": "实际长度受模型上下文长度约束。"}),
            "🌡️温度": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
            "🎯top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
            "🔝top_k": ("INT", {"default": 20, "min": 0, "max": 200, "step": 1}),
            "🔁重复惩罚": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
            "📈频率惩罚": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            "📍存在惩罚": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            "🎲随机种子": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1, "control_after_generate": True}),
            "🧠输出think块": ("BOOLEAN", {"default": False}),
            "📐图片最大边长": ("INT", {"default": 1024, "min": 128, "max": 16384, "step": 64, "tooltip": "对话中插入的图片按最长边缩放。"}),
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
            "最大边长": int(kwargs.get("📐图片最大边长", 1024)),
        },)


NODE_CLASS_MAPPINGS = {"DapaoSkillLoader": DapaoSkillLoader, "DapaoChatSettings": DapaoChatSettings}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DapaoSkillLoader": "🧩大炮Skill加载器@炮老师的小课堂",
    "DapaoChatSettings": "⚙️对话增强设置@炮老师的小课堂",
}
NODE_CLASS_MAPPINGS_SKILL = NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS_SKILL = NODE_DISPLAY_NAME_MAPPINGS
