# -*- coding: utf-8 -*-
"""大炮本地多轮聊天节点：会话状态由工作流控件保存，模型实例复用现有缓存。"""
from __future__ import annotations

import base64
import asyncio
import gc
import io
import json
import os
import re
import time
import inspect
from dataclasses import dataclass
import folder_paths
import comfy.model_management as mm
from PIL import Image

from .nodes import (
    DapaoLlamaStorage, Dapao_LlamaChat, QWEN38_REASONING_OPTIONS,
    KV_CACHE_DEFAULT, KV_CACHE_OPTIONS, normalize_qwen38_reasoning_effort,
)
from .skill_runtime import (
    api_messages,
    build_skill_prompt,
    current_message_content,
    get_skill,
    list_skills,
    normalize_history,
    normalize_image_refs,
    normalize_state,
    parse_skill_reply,
    read_reference,
    read_skill,
    select_material_mentions,
)


STATE_TAG = re.compile(r"<dapao_local_skill_state>\s*(\{.*?\})\s*</dapao_local_skill_state>", re.S)
CATEGORY = "🍭大炮-llama-cpp"
MODEL_FAMILIES = ["Qwen3-VL", "Qwen3.5-VL", "Qwen3.6-VL", "Qwen3.8-VL"]


def _reset_llm(llm):
    try:
        context = getattr(llm, "_ctx", None)
        if context is not None and hasattr(context, "memory_clear"):
            context.memory_clear(True)
    except Exception:
        pass
    try:
        hybrid_cache = getattr(llm, "_hybrid_cache_mgr", None)
        if hybrid_cache is not None and hasattr(hybrid_cache, "clear"):
            hybrid_cache.clear()
    except Exception:
        pass
    try:
        batch = getattr(llm, "_batch", None)
        if batch is not None and hasattr(batch, "reset"):
            batch.reset()
    except Exception:
        pass
    try:
        input_ids = getattr(llm, "input_ids", None)
        if input_ids is not None and hasattr(input_ids, "fill"):
            input_ids.fill(0)
    except Exception:
        pass
    try:
        reset = getattr(llm, "reset", None)
        if callable(reset):
            reset()
        elif hasattr(llm, "n_tokens"):
            llm.n_tokens = 0
    except Exception:
        pass


def _clean_reply(text: str) -> str:
    value = str(text or "").strip()
    match = re.search(r"<think>.*?</think>(.*)", value, re.S)
    return match.group(1).strip() if match else value


def _latest_assistant_reply(history: list[dict]) -> str:
    for item in reversed(history):
        if item.get("role") == "assistant" and isinstance(item.get("content"), str):
            return item["content"]
    return ""


def _json(raw, fallback):
    try:
        value = json.loads(raw or "")
        return value
    except (TypeError, json.JSONDecodeError):
        return fallback


def _history(raw) -> list[dict]:
    value = _json(raw, [])
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if not isinstance(item, dict) or item.get("role") not in ("user", "assistant"):
            continue
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        message = {"role": item["role"], "content": content.strip()}
        if item["role"] == "user" and isinstance(item.get("images"), list):
            images = []
            for ref in item["images"][:12]:
                if not isinstance(ref, dict):
                    continue
                filename = os.path.basename(str(ref.get("filename") or "").strip())
                subfolder = str(ref.get("subfolder") or "").replace("\\", "/").strip("/")
                if filename and (not subfolder or all(part not in ("", ".", "..") for part in subfolder.split("/"))):
                    images.append({"filename": filename, "subfolder": subfolder, "type": "input"})
            if images:
                message["images"] = images
        try:
            token_count = int(item.get("token_count"))
        except (TypeError, ValueError):
            token_count = -1
        if token_count >= 0:
            message["token_count"] = token_count
        try:
            created_at = int(item.get("created_at"))
        except (TypeError, ValueError):
            created_at = 0
        if created_at > 0:
            message["created_at"] = created_at
        if item["role"] == "assistant" and isinstance(item.get("flow_before"), dict):
            flow_before = item["flow_before"]
            message["flow_before"] = {
                "skill": str(flow_before.get("skill") or ""),
                "skill_name": str(flow_before.get("skill_name") or "")[:100],
                "stage": str(flow_before.get("stage") or "未开始")[:80],
                "loaded_references": [
                    str(x)
                    for x in flow_before.get("loaded_references", flow_before.get("loaded", []))
                    if isinstance(x, str)
                ][:100],
                "final_result": str(flow_before.get("final_result", flow_before.get("final", "")) or ""),
            }
        result.append(message)
    return result


def _image_refs(raw) -> list[dict]:
    value = _json(raw, raw if isinstance(raw, list) else [])
    if not isinstance(value, list):
        return []
    result = []
    for ref in value[:12]:
        if not isinstance(ref, dict):
            continue
        filename = os.path.basename(str(ref.get("filename") or "").strip())
        subfolder = str(ref.get("subfolder") or "").replace("\\", "/").strip("/")
        if filename and (not subfolder or all(part not in ("", ".", "..") for part in subfolder.split("/"))):
            result.append({"filename": filename, "subfolder": subfolder, "type": "input"})
    return result


def _state(raw) -> dict:
    value = _json(raw, {})
    if not isinstance(value, dict):
        value = {}
    return {
        "version": 2,
        "skill": str(value.get("skill") or ""),
        "skill_name": str(value.get("skill_name") or "")[:100],
        "stage": str(value.get("stage") or "未开始")[:80],
        "loaded_references": [
            str(x)
            for x in value.get("loaded_references", value.get("loaded", []))
            if isinstance(x, str)
        ][:100],
        "final_result": str(value.get("final_result", value.get("final", "")) or ""),
    }


def _image_uri(ref: dict, max_edge: int) -> str:
    root = os.path.realpath(folder_paths.get_input_directory())
    path = os.path.realpath(os.path.join(root, ref.get("subfolder", ""), ref["filename"]))
    if os.path.commonpath([root, path]) != root or not os.path.isfile(path):
        raise ValueError(f"找不到对话图片：{ref.get('filename', '')}")
    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        if max(width, height) > max_edge:
            scale = max_edge / max(width, height)
            image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _content(text: str, images: list[dict], max_edge: int):
    if not images:
        return text
    parts = [{"type": "text", "text": text}]
    for ref in images:
        parts.append({"type": "image_url", "image_url": {"url": _image_uri(ref, max_edge)}})
    return parts


def _messages(history: list[dict], max_edge: int):
    result = []
    for item in history:
        result.append({"role": item["role"], "content": _content(item["content"], item.get("images", []), max_edge)})
    return result


def _tokens(llm, text: str) -> int:
    try:
        return max(1, len(llm.tokenize(text.encode("utf-8"), add_bos=False)))
    except Exception:
        return max(1, len(text) // 2)


def _create_completion(llm, messages, params):
    active = dict(params)
    active["messages"] = messages
    try:
        signature = inspect.signature(llm.create_chat_completion)
        allowed = signature.parameters
        has_var_kw = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in allowed.values()
        )
    except Exception:
        allowed = {}
        has_var_kw = True
    if "presence_penalty" in active and "presence_penalty" not in allowed and "present_penalty" in allowed:
        active["present_penalty"] = active.pop("presence_penalty")
    if not has_var_kw:
        active = {key: value for key, value in active.items() if key in allowed}
    _reset_llm(llm)
    return llm.create_chat_completion(**active), int(params["max_tokens"])


def _extract_reply(result) -> str:
    try:
        content = result["choices"][0]["message"]["content"]
    except Exception:
        return str(result)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
            if isinstance(item, str) or (isinstance(item, dict) and isinstance(item.get("text"), str))
        )
    return str(content)


def _trim(llm, history, system_text, user_text, max_tokens, n_ctx, image_count, image_tokens):
    context_limit = max(512, int(n_ctx))
    requested_output = min(max(32, int(max_tokens)), max(32, context_limit - 512))
    safety_margin = 128

    def estimate(items):
        total = _tokens(llm, system_text) + 16
        for item in items:
            total += _tokens(llm, item.get("content", "")) + 8
            total += len(item.get("images", [])) * image_tokens
        return total

    current = list(history)
    trimmed_count = 0
    output_reserve = requested_output
    budget = max(256, context_limit - output_reserve - safety_margin)
    probe = current + [{"role": "user", "content": user_text, "images": [{}] * image_count}]

    # 先删除最旧的完整对话轮次，尽量保留用户设置的输出长度。
    while current and estimate(probe) > budget:
        current.pop(0)
        trimmed_count += 1
        if current and current[0]["role"] == "assistant":
            current.pop(0)
            trimmed_count += 1
        probe = current + [{"role": "user", "content": user_text, "images": [{}] * image_count}]

    required_tokens = estimate(probe)
    if required_tokens > budget:
        # 即使没有历史仍放不下时，自动缩小本轮输出预留，而不是直接报错。
        available_output = context_limit - required_tokens - safety_margin
        output_reserve = min(requested_output, max(32, available_output))
        budget = max(256, context_limit - output_reserve - safety_margin)

    if required_tokens > budget:
        recommended_context = ((required_tokens + safety_margin + 32 + 511) // 512) * 512
        image_note = f"，其中包含 {image_count} 张图片的视觉 token 预算" if image_count else ""
        raise ValueError(
            f"当前系统提示词、Skill 和本轮消息约需 {required_tokens} tokens{image_note}，"
            f"已经超过 {context_limit} 上下文能够容纳的输入空间。"
            f"节点已尝试清理历史并把输出压缩到最低值，仍然无法放入。"
            f"建议把模型加载器的上下文长度提高到至少 {recommended_context}，"
            "或者缩短 Skill/消息、减少图片后重试。"
        )
    return current, budget, required_tokens, output_reserve, trimmed_count


def _auto_select_skill(llm, skills: list[dict], text: str) -> str:
    if not skills:
        raise ValueError("Skill加载器没有发现可用 Skill，请把 Skill 放入节点根目录的 skills 文件夹。")
    catalogue = "\n".join(
        f'- {item["id"]}: {item["name"]}；{str(item.get("description") or "")[:500]}'
        for item in skills
    )
    result, _ = _create_completion(
        llm,
        [
            {"role": "system", "content": "根据用户任务选择唯一最匹配的 Skill。只输出 Skill ID，不解释，不添加标点。"},
            {"role": "user", "content": f"可用 Skills：\n{catalogue}\n\n用户任务：\n{text}"},
        ],
        {"max_tokens": 80, "temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": 0, "stream": False},
    )
    selected = _clean_reply(_extract_reply(result)).strip().strip("`'\".,，。 ")
    valid_ids = {str(item.get("id") or "") for item in skills}
    if selected in valid_ids:
        return selected
    for skill_id in valid_ids:
        if skill_id and skill_id in selected:
            return skill_id
    raise ValueError(
        f"自动选择 Skill 失败，模型返回：{selected[:120]}。"
        "请在 Skill加载器中手动选择后重试。"
    )


def _pick_skill(llm, config, text: str, previous_id: str = ""):
    if not isinstance(config, dict):
        return None
    skills = config.get("skills") if isinstance(config, dict) else []
    if not isinstance(skills, list):
        skills = list(list_skills())
    selected = str(config.get("selected") or "") if isinstance(config, dict) else ""
    if not selected and previous_id:
        selected = previous_id
    if selected:
        return get_skill(selected)
    return get_skill(_auto_select_skill(llm, skills, text))


def _skill_prompt(base: str, skill: dict, state: dict) -> str:
    loaded = []
    for path in state["loaded_references"]:
        if path in skill["references"]:
            loaded.append(f"\n--- 已载入资料：{path} ---\n{read_reference(skill, path)}")
    catalogue = "\n".join(f"- {path}" for path in skill["references"]) or "- 无"
    loaded_names = "、".join(state["loaded_references"]) or "无"
    protocol = (
        "你正在通过 ComfyUI 的本地 Skill 执行器工作。只完成当前对话中能完成的内容，"
        "不得声称已调用联网、画布、媒体生成或其他未连接工具。信息不足或到达确认门时先提问，"
        "每次只推进当前阶段。回复正文之后必须追加且最后只能出现一个状态标记："
        "<dapao_local_skill_state>{\"stage\":\"当前阶段\",\"options\":[],"
        "\"load_references\":[],\"final\":false}</dapao_local_skill_state>。"
        "options 最多6个；load_references 只能填写可用资料中的相对路径；"
        "只有交付完整最终产物时 final 才能为 true。使用简体中文交流。"
    )
    return "\n\n".join(x for x in [
        base.strip(),
        f"当前工作 Skill：{skill['name']} ({skill['id']})\n当前阶段：{state['stage']}"
        f"\n可用资料：\n{catalogue}\n已加载资料：{loaded_names}",
        read_skill(skill), *loaded, protocol,
    ] if x)


def _parse_reply(raw: str):
    match = list(STATE_TAG.finditer(raw or ""))
    if not match:
        return raw.strip(), {}
    last = match[-1]
    state = _json(last.group(1), {})
    return (raw[:last.start()] + raw[last.end():]).strip(), state if isinstance(state, dict) else {}


def _ensure_model(params):
    cfg = DapaoLlamaStorage.current_config or {}
    family = params["family"]
    handler = {
        "Qwen3-VL": "Qwen3-VL",
        "Qwen3.5-VL": "Qwen3.5",
        "Qwen3.6-VL": "Qwen3.5",
        "Qwen3.8-VL": "Qwen3.8",
    }[family]
    same = all((
        cfg.get("model_file") == params["model_file"],
        cfg.get("handler_name") == handler,
        cfg.get("mmproj_file") == params["mmproj_file"],
        int(cfg.get("n_ctx", -1)) == params["n_ctx"],
        int(cfg.get("n_gpu_layers", -9999)) == params["n_gpu_layers"],
        cfg.get("cache_type_k", KV_CACHE_DEFAULT) == params["cache_type_k"],
        cfg.get("cache_type_v", KV_CACHE_DEFAULT) == params["cache_type_v"],
        bool(cfg.get("think_mode", False)) == params["think"],
        bool(cfg.get("preserve_thinking", False)) == params["preserve_thinking"],
        cfg.get("reasoning_effort") == params["reasoning"],
        bool(cfg.get("cpu_moe", False)) == params["cpu_moe"],
        int(cfg.get("n_cpu_moe", 0)) == params["n_cpu_moe"],
        cfg.get("model_family") == family,
    ))
    if DapaoLlamaStorage.llm is None or not same:
        DapaoLlamaStorage.clean()
        Dapao_LlamaChat()._load_model(
            params["model_file"], handler, params["mmproj_file"], params["n_ctx"],
            -1, 256, 1344, params["think"], params["reasoning"],
            n_gpu_layers_override=params["n_gpu_layers"],
            cache_type_k=params["cache_type_k"],
            cache_type_v=params["cache_type_v"],
            preserve_thinking=params["preserve_thinking"],
            cpu_moe=params["cpu_moe"],
            n_cpu_moe=params["n_cpu_moe"],
            model_family=family,
        )
    return DapaoLlamaStorage.llm


@dataclass
class DapaoLocalModel:
    llm: object
    settings: dict
    chat_handler: object | None = None


def _bool_setting(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "开启", "是")
    return bool(value)


def _model_settings(model) -> dict:
    """Accept current, cached pre-reload, and mapping-based loader outputs."""
    if isinstance(model, dict):
        raw = model.get("settings", model)
    else:
        raw = getattr(model, "settings", None)
    if not isinstance(raw, dict):
        received = "空值" if model is None else type(model).__name__
        raise RuntimeError(
            "没有收到有效的本地模型配置。"
            f"当前收到：{received}。请确认“大炮本地模型加载器”的绿色输出已连接到本节点；"
            "如果刚更新过节点，请重启 ComfyUI 后重新执行工作流。"
        )

    model_file = str(raw.get("model_file") or "").strip()
    if not model_file:
        raise RuntimeError(
            "本地模型配置缺少模型文件信息。请重新执行“大炮本地模型加载器”；"
            "如果问题仍存在，请删除旧加载器后重新添加。"
        )

    think = bool(raw.get("think", raw.get("think_mode", False)))
    reasoning = normalize_qwen38_reasoning_effort(
        raw.get("reasoning", raw.get("reasoning_effort", "xhigh" if think else "off"))
    )
    try:
        return {
            "family": str(raw.get("family", raw.get("model_family", "Qwen3.8-VL"))),
            "model_file": model_file,
            "mmproj_file": str(raw.get("mmproj_file") or "None"),
            "n_ctx": int(raw.get("n_ctx", 8192)),
            "n_gpu_layers": int(raw.get("n_gpu_layers", -1)),
            "cache_type_k": str(raw.get("cache_type_k", KV_CACHE_DEFAULT)),
            "cache_type_v": str(raw.get("cache_type_v", KV_CACHE_DEFAULT)),
            "think": think,
            "preserve_thinking": bool(raw.get("preserve_thinking", False)),
            "reasoning": reasoning,
            "cpu_moe": bool(raw.get("cpu_moe", False)),
            "n_cpu_moe": int(raw.get("n_cpu_moe", 0)),
            "unload_after_run": _bool_setting(
                raw.get("unload_after_run", raw.get("force_offload", False))
            ),
        }
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "本地模型配置中的上下文、显存或图像 token 参数无效。"
            "请重新执行“大炮本地模型加载器”；如果问题仍存在，请删除旧加载器后重新添加。"
        ) from error


def _sync_model(model):
    # ComfyUI may retain an output object created before a node module reload.
    # Its class identity is stale even though its settings are still valid, so
    # validate the data contract instead of requiring an exact isinstance match.
    settings = _model_settings(model)
    llm = _ensure_model(settings)
    return DapaoLocalModel(
        llm=llm,
        settings=settings,
        chat_handler=DapaoLlamaStorage.chat_handler,
    )


def _should_unload(model) -> bool:
    if isinstance(model, dict):
        raw = model.get("settings", model)
    else:
        raw = getattr(model, "settings", None)
    return isinstance(raw, dict) and _bool_setting(
        raw.get("unload_after_run", raw.get("force_offload", False))
    )


def _release_model(model):
    if isinstance(model, dict):
        held_llm = model.get("llm")
    else:
        held_llm = getattr(model, "llm", None)
    active_llm = DapaoLlamaStorage.llm
    DapaoLlamaStorage.clean()
    if held_llm is not None and held_llm is not active_llm and hasattr(held_llm, "close"):
        try:
            held_llm.close()
        except Exception as error:
            print(f"[大炮-llama] 关闭缓存模型对象时出现警告：{error}")
    if isinstance(model, dict):
        if "llm" in model:
            model["llm"] = None
    else:
        try:
            model.llm = None
        except Exception:
            pass
    del held_llm, active_llm
    gc.collect()
    mm.soft_empty_cache()
    print("[大炮-llama] 已按设置卸载模型与多模态处理器，并释放显存")


def _material_aliases(raw) -> dict[str, str]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        value = raw
    else:
        text = str(raw).strip()
        try:
            value = json.loads(text or "{}")
        except json.JSONDecodeError:
            value = {}
            for line in text.splitlines():
                if not line.strip():
                    continue
                key, separator, label = line.partition("=")
                if not separator:
                    raise ValueError("素材别名格式错误，请使用JSON，或每行填写“图片1=产品正面”。")
                value[key.strip()] = label.strip()
    if not isinstance(value, dict):
        raise ValueError("素材别名必须是JSON对象。")
    result = {}
    for key, label in value.items():
        normalized_key = str(key or "").strip().lstrip("@")
        normalized_label = str(label or "").strip().lstrip("@")
        if normalized_key and normalized_label:
            result[normalized_key] = normalized_label[:80]
    if len(set(result.values())) != len(result):
        raise ValueError("素材别名不能重复，否则@菜单无法区分。")
    return result


class DapaoLocalChatMaterialLibrary:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("DAPAO_LOCAL_CHAT_MATERIAL_LIBRARY",)
    RETURN_NAMES = ("📦多轮对话素材库",)
    FUNCTION = "build_library"
    DESCRIPTION = "登记最多20图、5视频、5音频；只有聊天框本轮明确@引用的素材才会被处理并传给本地模型。"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for index in range(1, 21):
            optional[f"🖼️图片{index}"] = ("IMAGE", {"tooltip": f"待引用图片{index}；每个接口只连接单张IMAGE。"})
        for index in range(1, 6):
            optional[f"🎞️视频{index}"] = ("VIDEO", {"tooltip": f"只有本轮@视频{index}时才抽取代表帧。"})
            optional[f"🎵音频{index}"] = ("AUDIO", {"tooltip": f"只有本轮@音频{index}时才压缩为16kHz单声道WAV。"})
        return {
            "required": {
                "🏷️素材别名": ("STRING", {
                    "default": "{}",
                    "multiline": True,
                    "tooltip": "可选。JSON示例：{\"图片1\":\"产品正面\",\"视频1\":\"开场镜头\"}。固定编号始终是内部稳定ID。",
                }),
            },
            "optional": optional,
        }

    def build_library(self, **kwargs):
        aliases = _material_aliases(kwargs.get("🏷️素材别名"))
        items = []
        specs = (("image", "图片", "🖼️图片", 20), ("video", "视频", "🎞️视频", 5), ("audio", "音频", "🎵音频", 5))
        for kind, chinese, prefix, limit in specs:
            for slot in range(1, limit + 1):
                value = kwargs.get(f"{prefix}{slot}")
                if value is None:
                    continue
                token = f"@{chinese}{slot}"
                if kind == "image":
                    if not hasattr(value, "shape") or len(value.shape) != 4:
                        raise ValueError(f"{token}必须连接ComfyUI IMAGE。")
                    if int(value.shape[0]) != 1:
                        raise ValueError(f"{token}包含{int(value.shape[0])}张图片；请先拆分批次，每个素材接口只接一张。")
                key = f"{chinese}{slot}"
                items.append({
                    "kind": kind,
                    "slot": slot,
                    "token": token,
                    "label": aliases.get(key, token),
                    "value": value,
                })
        return ({"version": 1, "items": items},)


class DapaoLocalModelLoader:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("DAPAO_LOCAL_MODEL",)
    RETURN_NAMES = ("🤖本地模型",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        files = folder_paths.get_filename_list("LLM")
        models = [
            name for name in files
            if "mmproj" not in name.lower() and os.path.splitext(name)[1].lower() == ".gguf"
        ]
        projections = ["无"] + [
            name for name in files
            if "mmproj" in name.lower() and os.path.splitext(name)[1].lower() == ".gguf"
        ]
        if not models:
            models = ["（请把GGUF模型放入ComfyUI/models/LLM）"]
        return {"required": {
            "🧬模型系列": (MODEL_FAMILIES, {"default": "Qwen3.8-VL"}),
            "🤖主模型": (models, {"tooltip": "GGUF 主模型，放入 ComfyUI/models/LLM。"}),
            "🖼️视觉投影mmproj": (projections, {"default": "无", "tooltip": "图片对话必须选择与主模型匹配的 mmproj；纯文本可选“无”。"}),
            "🧠启用思考": ("BOOLEAN", {"default": False}),
            "🧾保留历史think": ("BOOLEAN", {"default": False, "tooltip": "关闭可减少多轮对话的上下文占用。"}),
            "📐上下文长度": ("INT", {"default": 8192, "min": 1024, "max": 327680, "step": 256}),
            "🎮GPU层数": ("INT", {"default": -1, "min": -1, "max": 9999, "step": 1, "tooltip": "-1=尽可能全部放入GPU，速度最快；0=纯CPU；正数=仅指定层数放入GPU。"}),
            "🗜️KV缓存K类型": (KV_CACHE_OPTIONS, {"default": KV_CACHE_DEFAULT, "tooltip": "F16质量与兼容性优先；Q8_0更省显存，部分大模型可能更快。"}),
            "🗜️KV缓存V类型": (KV_CACHE_OPTIONS, {"default": KV_CACHE_DEFAULT, "tooltip": "F16质量与兼容性优先；Q8_0更省显存，部分大模型可能更快。"}),
            "🧩MoE专家上CPU": ("BOOLEAN", {"default": False, "tooltip": "仅Qwen3.6-VL生效。显存不足时使用，通常会变慢。"}),
            "🔢前N层专家上CPU": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1, "tooltip": "仅Qwen3.6-VL生效；开启全部专家上CPU时忽略。"}),
            "🧠Qwen3.8推理强度": (QWEN38_REASONING_OPTIONS, {"default": "关闭", "tooltip": "仅Qwen3.8生效；关闭时不输出思考。"}),
            "🧹推理后卸载模型": ("BOOLEAN", {
                "default": False,
                "tooltip": "关闭=模型常驻显存，后续对话更快；开启=每次完成或报错后立即卸载GGUF和mmproj并释放显存，下一轮需要重新加载模型。",
            }),
        }}

    def load(self, **kwargs):
        model_file = str(kwargs["🤖主模型"])
        if model_file.startswith("（请把GGUF模型"):
            raise RuntimeError("没有找到 GGUF 模型，请把模型放入 ComfyUI/models/LLM 后刷新节点。")
        family = str(kwargs.get("🧬模型系列") or "Qwen3.8-VL")
        thinking = bool(kwargs.get("🧠启用思考", False))
        if family == "Qwen3.8-VL":
            reasoning = normalize_qwen38_reasoning_effort(
                kwargs.get("🧠Qwen3.8推理强度", "关闭") if thinking else "关闭"
            )
        else:
            reasoning = "xhigh" if thinking else "off"
        mmproj_file = str(kwargs.get("🖼️视觉投影mmproj") or "无")
        settings = {
            "family": family,
            "model_file": model_file,
            "mmproj_file": "None" if mmproj_file == "无" else mmproj_file,
            "n_ctx": int(kwargs["📐上下文长度"]),
            "n_gpu_layers": int(kwargs["🎮GPU层数"]),
            "cache_type_k": str(kwargs["🗜️KV缓存K类型"]),
            "cache_type_v": str(kwargs["🗜️KV缓存V类型"]),
            "think": thinking and (family != "Qwen3.8-VL" or reasoning != "off"),
            "preserve_thinking": bool(kwargs.get("🧾保留历史think", False)),
            "reasoning": reasoning,
            "cpu_moe": bool(kwargs.get("🧩MoE专家上CPU", False)),
            "n_cpu_moe": int(kwargs.get("🔢前N层专家上CPU", 0)),
            "unload_after_run": _bool_setting(kwargs.get("🧹推理后卸载模型", False)),
        }
        llm = _ensure_model(settings)
        return (DapaoLocalModel(llm=llm, settings=settings, chat_handler=DapaoLlamaStorage.chat_handler),)


class DapaoMultiTurnChat:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("💬助手回复", "📚会话历史JSON", "🧩Skill最终结果")
    FUNCTION = "run"
    OUTPUT_NODE = True
    DESCRIPTION = "本地llama多轮对话工作台：支持Skill、历史和@素材库；素材只在本轮明确@时进入推理。"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "💬本轮消息": ("STRING", {"default": "", "multiline": True}),
                "📚会话历史": ("STRING", {"default": "[]", "multiline": True}),
                "🖼️图片引用": ("STRING", {"default": "[]", "multiline": True}),
                "🧩流程状态": ("STRING", {"default": "{}", "multiline": True}),
                "🧩选项": ("STRING", {"default": "[]", "multiline": True}),
                "🆔请求标识": ("STRING", {"default": ""}),
                "🧭执行动作": ("STRING", {"default": "chat"}),
            },
            "optional": {
                "🤖本地模型": ("DAPAO_LOCAL_MODEL",),
                "⚙️对话设置": ("DAPAO_CHAT_SETTINGS",),
                "🧩Skill配置": ("DAPAO_SKILL_CONFIG",),
                "📦素材库": ("DAPAO_LOCAL_CHAT_MATERIAL_LIBRARY",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    async def run(self, **kwargs):
        return await asyncio.to_thread(self._run_with_cleanup, kwargs)

    def _run_with_cleanup(self, kwargs):
        source_model = kwargs.get("🤖本地模型")
        action = str(kwargs.get("🧭执行动作") or "chat").strip().lower()
        try:
            return self._run(**kwargs)
        finally:
            if action != "publish_final" and _should_unload(source_model):
                _release_model(source_model)

    def _run(self, **kwargs):
        display_history = normalize_history(kwargs.get("📚会话历史", "[]"))
        user_text = str(kwargs.get("💬本轮消息") or "").strip()
        state = normalize_state(kwargs.get("🧩流程状态", "{}"))
        flow_before = {**state, "loaded_references": list(state["loaded_references"])}
        action = str(kwargs.get("🧭执行动作") or "chat").strip().lower()
        if action == "publish_final":
            reply = _latest_assistant_reply(display_history)
            raw_options = _json(kwargs.get("🧩选项", "[]"), [])
            options = [str(item)[:240] for item in raw_options[:6] if str(item).strip()] if isinstance(raw_options, list) else []
            return self._result(display_history, reply, state, options, False, {})

        current_images = normalize_image_refs(kwargs.get("🖼️图片引用", "[]"))
        if not user_text and current_images:
            user_text = "请分析上传的图片，并根据当前对话或Skill继续处理。"
        if not user_text:
            return self._result(display_history, _latest_assistant_reply(display_history), state, [], False, {})

        source_settings = _model_settings(kwargs.get("🤖本地模型"))
        settings = kwargs.get("⚙️对话设置") or {}
        settings = settings if isinstance(settings, dict) else {}
        max_rounds = min(100, max(1, int(settings.get("最大历史轮数", 100))))
        max_tokens = int(settings.get("最大生成token", 1024))
        max_edge = min(2048, max(128, int(settings.get("最大边长", 2048))))
        system_base = str(settings.get("系统提示词") or "")
        selected_materials = select_material_mentions(user_text, kwargs.get("📦素材库"))
        visual_materials = current_images or any(item["kind"] in {"image", "video"} for item in selected_materials)
        if visual_materials and source_settings.get("mmproj_file") in ("", "None", "无"):
            raise RuntimeError("本轮引用了图片或视频，但没有选择mmproj；请加载与主模型匹配的视觉投影。")

        capabilities = {
            "supports_images": source_settings.get("mmproj_file") not in ("", "None", "无"),
            "supports_video": False,
            "supports_audio": True,
        }
        current_content, material_stats = current_message_content(
            user_text,
            current_images,
            selected_materials,
            max_edge,
            capabilities,
        )
        media_image_equivalents = (
            int(material_stats["image_parts"])
            + int(material_stats["video_frames"])
            + (1 if material_stats["audio_seconds"] else 0)
        )

        model = _sync_model(kwargs.get("🤖本地模型"))
        llm = model.llm
        model_settings = model.settings
        n_ctx = int(model_settings.get("n_ctx", 8192))
        if visual_materials and model.chat_handler is None:
            raise RuntimeError("本轮引用了图片或视频，但当前mmproj处理器没有成功加载；请检查视觉投影是否与主模型匹配。")

        skill_config = kwargs.get("🧩Skill配置")
        selected_skill = str(skill_config.get("selected") or "") if isinstance(skill_config, dict) else ""
        if selected_skill and state.get("skill") and selected_skill != state["skill"]:
            state = normalize_state({"context_cutoff": state.get("context_cutoff", 0)})
        skill = _pick_skill(llm, skill_config, user_text, state.get("skill", ""))
        if skill:
            state["skill"], state["skill_name"] = skill["id"], skill["name"]
            system = build_skill_prompt(system_base, skill, state)
        else:
            system = system_base or "你是一个专业、友好、准确的AI助手。"

        cutoff = int(state.get("context_cutoff") or 0)
        context_history = [
            {"role": item["role"], "content": item["content"]}
            for item in display_history
            if not cutoff or int(item.get("created_at", -1)) > cutoff
        ][-max_rounds * 2:]
        context_history, budget, used, output_reserve, trimmed_messages = _trim(
            llm,
            context_history,
            system,
            user_text,
            max_tokens,
            n_ctx,
            media_image_equivalents,
            1536,
        )
        if output_reserve < max_tokens:
            print(f"[大炮-llama] 自动适配上下文：本轮最大输出 {max_tokens} -> {output_reserve} tokens")

        temperature = float(settings.get("温度", 0.7))
        top_p = float(settings.get("top_p", 0.9))
        top_k = int(settings.get("top_k", 20))
        if model_settings.get("family") == "Qwen3.8-VL":
            if bool(model_settings.get("think", False)):
                if abs(temperature - 0.7) < 1e-9:
                    temperature = 1.0
                if abs(top_p - 0.9) < 1e-9:
                    top_p = 0.95
            elif abs(top_p - 0.9) < 1e-9:
                top_p = 0.8
        params = {
            "max_tokens": output_reserve,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": float(settings.get("重复惩罚", 1.0)),
            "frequency_penalty": float(settings.get("频率惩罚", 0.0)),
            "presence_penalty": float(settings.get("存在惩罚", 0.0)),
            "seed": int(settings.get("seed", 0)),
            "stream": False,
            "stop": ["</s>"],
        }
        if model_settings.get("family") == "Qwen3.8-VL":
            params["min_p"] = 0.0

        def make_messages():
            messages = ([{"role": "system", "content": system}] if system else []) + api_messages(context_history, max_edge)
            messages.append({"role": "user", "content": current_content})
            return messages

        reply, skill_state, options = "", {}, []
        for attempt in range(2):
            try:
                response, actual_max_tokens = _create_completion(llm, make_messages(), params)
                params["max_tokens"] = actual_max_tokens
                output_reserve = min(output_reserve, actual_max_tokens)
                raw = _extract_reply(response)
            except Exception as error:
                has_audio = any(item["kind"] == "audio" for item in selected_materials)
                audio_hint = (
                    " 当前本地模型或llama.cpp聊天处理器可能不支持input_audio；"
                    "请换用支持音频的本地多模态模型，或取消@音频后重试。"
                    if has_audio else ""
                )
                raise RuntimeError(f"本地模型推理失败：{error}.{audio_hint}") from error
            cleaned = raw if bool(settings.get("输出think块", False)) else _clean_reply(raw)
            reply, skill_state = parse_skill_reply(cleaned.lstrip().removeprefix(": ").strip())
            if not skill:
                break
            requested = [
                path
                for path in skill_state.get("load_references", [])
                if isinstance(path, str)
                and path in skill["references"]
                and path not in state["loaded_references"]
            ]
            if requested and attempt == 0:
                state["loaded_references"].extend(requested)
                system = build_skill_prompt(system_base, skill, state)
                context_history, budget, used, output_reserve, trimmed_messages = _trim(
                    llm,
                    context_history,
                    system,
                    user_text,
                    max_tokens,
                    n_ctx,
                    media_image_equivalents,
                    1536,
                )
                params["max_tokens"] = output_reserve
                continue
            state["stage"] = str(skill_state.get("stage") or "进行中")[:80]
            raw_options = skill_state.get("options")
            options = [str(item)[:240] for item in raw_options[:6] if str(item).strip()] if isinstance(raw_options, list) else []
            if skill_state.get("final"):
                state["final_result"] = reply
            break

        created_at = _request_time(kwargs.get("🆔请求标识"))
        user_message = {
            "role": "user",
            "content": user_text,
            "token_count": _tokens(llm, user_text) + 8 + media_image_equivalents * 1536,
            "created_at": created_at,
            **({"images": current_images} if current_images else {}),
            **({
                "materials": [
                    {key: item[key] for key in ("kind", "slot", "token", "label")}
                    for item in selected_materials
                ],
            } if selected_materials else {}),
        }
        assistant_message = {
            "role": "assistant",
            "content": reply,
            "token_count": _tokens(llm, reply) + 8,
            "created_at": int(time.time() * 1000),
            "flow_before": flow_before,
        }
        display_history.extend((user_message, assistant_message))
        conversation_used = min(n_ctx, used + assistant_message["token_count"])
        context = {
            "used_tokens": conversation_used,
            "input_used_tokens": used,
            "assistant_tokens": assistant_message["token_count"],
            "prompt_budget": budget,
            "context_limit": n_ctx,
            "output_reserve": output_reserve,
            "requested_output": max_tokens,
            "output_auto_adjusted": output_reserve < max_tokens,
            "remaining_tokens": max(0, budget - used),
            "input_remaining_tokens": max(0, budget - used),
            "total_remaining_tokens": max(0, n_ctx - conversation_used),
            "percent": round(conversation_used / max(1, n_ctx) * 100, 1),
            "input_percent": round(used / max(1, budget) * 100, 1),
            "current_rounds": sum(item["role"] == "user" for item in context_history) + 1,
            "max_rounds": max_rounds,
            "trimmed_messages": trimmed_messages,
            "usage_source": "local_tokenizer",
            "model": model_settings.get("model_file", ""),
            "material_count": len(selected_materials),
            "material_image_parts": int(material_stats["image_parts"]),
            "material_video_frames": int(material_stats["video_frames"]),
            "material_audio_seconds": round(float(material_stats["audio_seconds"]), 3),
        }
        return self._result(display_history, reply, state, options, True, context)

    @staticmethod
    def _result(history, reply, state, options, sent, context):
        history_json = json.dumps(history, ensure_ascii=False, separators=(",", ":"))
        final_result = str(state.get("final_result") or reply or "")
        return {
            "ui": {
                "📚会话历史": [history_json],
                "💬助手回复": [reply],
                "🧩流程状态": [json.dumps(state, ensure_ascii=False, separators=(",", ":"))],
                "🧩选项": [json.dumps(options, ensure_ascii=False)],
                "📊上下文": [json.dumps(context, ensure_ascii=False)],
                "✅已发送": [bool(sent)],
            },
            "result": (reply, history_json, final_result),
        }


class DapaoMultiTurnChatV2(DapaoMultiTurnChat):
    """独立节点类型，避免已保存工作流的旧控件影响当前聊天界面。"""

    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("💬助手回复", "📚会话历史JSON", "🧩Skill最终结果")
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return DapaoMultiTurnChat.INPUT_TYPES()


def _request_time(request_id: str) -> int:
    now = int(time.time() * 1000)
    try:
        value = int(str(request_id or "").split("-", 1)[0])
    except (TypeError, ValueError):
        return now
    return value if 946684800000 <= value <= now + 300000 else now


NODE_CLASS_MAPPINGS = {
    "DapaoLocalModelLoader": DapaoLocalModelLoader,
    "DapaoLocalChatMaterialLibrary": DapaoLocalChatMaterialLibrary,
    "DapaoMultiTurnChatV2": DapaoMultiTurnChatV2,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DapaoLocalModelLoader": "🤖大炮本地模型加载器@炮老师的小课堂",
    "DapaoLocalChatMaterialLibrary": "📦大炮本地多轮对话素材库@炮老师的小课堂",
    "DapaoMultiTurnChatV2": "💬大炮本地多轮对话@炮老师的小课堂",
}
NODE_CLASS_MAPPINGS_MULTI = NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS_MULTI = NODE_DISPLAY_NAME_MAPPINGS
