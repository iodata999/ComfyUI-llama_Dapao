"""Qwen-Image-2.1 prompt rewriting with the project's local llama.cpp model loader.

The reference rules in resources/qwen_image_2_1 are copied from
iamyoki/qwen-image-2.1-skill (Apache-2.0). This adapter changes their chat
interaction into explicit ComfyUI controls and machine-readable outputs.
"""

import asyncio
import json
import re
import threading
from pathlib import Path

from .multi_turn_chat_node import (
    _create_completion, _extract_reply, _release_model, _should_unload,
    _sync_model,
)
from .nodes import image2base64, scale_image, tensor2pil


RULE_DIR = Path(__file__).parent / "resources" / "qwen_image_2_1"
# llama.cpp uses a shared model/context in this plugin; queued node runs must
# not decode through the same context concurrently.
_QWEN_INFERENCE_LOCK = threading.Lock()
RATIOS = ["自动判断", "继承基准图", "3:2", "2:3", "1:1", "16:9", "9:16", "1:2", "3:4", "2:1", "21:9", "4:3", "9:21", "4:5", "3:1", "5:4", "1:3", "18:39", "9:20", "7:3", "9:5", "5:7", "7:5"]
TASKS = ["自动识别", "文生图", "图像编辑与局部修改", "多图合成", "风格迁移", "背景替换", "换脸或换装", "扩图", "参考主体生成新场景"]
OUTPUT_STYLES = ["标准展示（解析+提示词+建议）", "仅提示词", "严格JSON"]


def _read_rules(mode):
    names = ("t2i_rules.md", "cheat_sheet.md") if mode == "t2i" else ("edit_rules.md", "cheat_sheet.md")
    return "\n\n".join((RULE_DIR / name).read_text(encoding="utf-8") for name in names)


def _extract_json(raw):
    text = re.sub(r"<think>.*?</think>", "", str(raw), flags=re.S).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text[index:])
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                pass
    raise ValueError("本地模型没有返回可解析的JSON；请提高最大输出token，或更换指令遵循能力更强的模型。")


def _images(kwargs):
    images = []
    for number in range(1, 9):
        value = kwargs.get(f"🖼️参考图{number}")
        if value is None:
            continue
        if not hasattr(value, "shape") or len(value.shape) != 4 or int(value.shape[0]) != 1:
            raise ValueError(f"参考图{number}必须是单张IMAGE；请先拆分批次。")
        if int(value.shape[-1]) not in (3, 4):
            raise ValueError(f"参考图{number}必须是RGB或RGBA图像。")
        images.append(value)
    return images


def _normalize_payload(data, mode, ratio, canvas, count):
    prompt = str(data.get("rewritten_prompt") or "").replace("\r", " ").replace("\n", " ").strip()
    prompt = re.sub(r"\s{2,}", " ", prompt)
    if not prompt:
        raise ValueError("本地模型没有生成有效提示词。")
    if mode == "t2i":
        wh_ratio = ratio if ratio not in ("自动判断", "继承基准图") else str(data.get("wh_ratio") or "3:2")
        ratio_follow = ""
    elif ratio not in ("自动判断", "继承基准图"):
        wh_ratio, ratio_follow = ratio, ""
    elif ratio == "继承基准图":
        wh_ratio, ratio_follow = "", f"<image{canvas}>"
    else:
        wh_ratio = str(data.get("wh_ratio") or "").strip()
        ratio_follow = str(data.get("ratio_follow") or "").strip()
        if not wh_ratio and not ratio_follow:
            ratio_follow = f"<image{canvas}>"
        if wh_ratio and ratio_follow:
            ratio_follow = ""
    if wh_ratio and not re.fullmatch(r"\d+:\d+", wh_ratio):
        raise ValueError(f"模型返回的画幅比例无效：{wh_ratio}")
    if ratio_follow and (not re.fullmatch(r"<image\d+>", ratio_follow) or int(ratio_follow[6:-1]) > count):
        raise ValueError(f"模型返回的基准图标签无效：{ratio_follow}")
    if mode == "edit" and count >= 2:
        missing = [f"<image{n}>" for n in range(1, count + 1) if f"<image{n}>" not in prompt]
        if missing:
            raise ValueError("多图提示词缺少图像引用：" + "、".join(missing))
    return {"rewritten_prompt": prompt, "wh_ratio": wh_ratio, "ratio_follow": ratio_follow}


class DapaoQwenImage21Prompt:
    CATEGORY = "🍭大炮-llama-cpp"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("📋可用提示词", "📐wh_ratio", "🖼️ratio_follow", "🧾JSON结果", "🧩展示结果")
    DESCRIPTION = "把 Qwen-Image-2.1 参考 skill 的文生图、改图、多图合成与画幅规则变成可选节点参数；使用大炮本地模型加载器。"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {f"🖼️参考图{n}": ("IMAGE",) for n in range(1, 9)}
        optional["🔗上游需求"] = ("STRING", {"forceInput": True})
        return {
            "required": {
                "🤖本地模型": ("DAPAO_LOCAL_MODEL",),
                "📝原始需求": ("STRING", {"default": "", "multiline": True}),
                "🧭任务模式": (TASKS, {"default": "自动识别"}),
                "📐输出画幅": (RATIOS, {"default": "自动判断", "tooltip": "文生图由模型按用途推断；改图可继承基准图，或指定比例。比例只在输出口提供。"}),
                "🖼️基准图": ([str(n) for n in range(1, 9)], {"default": "1", "tooltip": "多图编辑时作为保留原构图的画布。编号按已连接参考图的顺序排列。"}),
                "🔤画面文字": (["按用户原文与参考图判断", "禁止新增可读文字", "严格保留用户指定文字", "自定义准确文字"], {"default": "按用户原文与参考图判断"}),
                "✍️自定义准确文字": ("STRING", {"default": "", "multiline": True, "tooltip": "选择“自定义准确文字”时生效；逐字保留并使用直双引号。"}),
                "🧩输出样式": (OUTPUT_STYLES, {"default": OUTPUT_STYLES[0], "tooltip": "控制展示结果输出口；可用提示词输出口始终保持纯提示词，方便接生图节点。"}),
                "📤最大输出token": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 128}),
                "🌡️温度": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01}),
                "🎲随机种子": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "🧹为生图释放显存": ("BOOLEAN", {"default": True, "tooltip": "提示词生成后释放本地LLM显存，给后续Qwen Image生图使用；下次运行会重新加载LLM。"}),
            },
            "optional": optional,
        }

    async def generate(self, **kwargs):
        return await asyncio.to_thread(self._generate_sync, kwargs)

    def _generate_sync(self, kwargs):
        with _QWEN_INFERENCE_LOCK:
            source_model = kwargs["🤖本地模型"]
            try:
                return self._run(kwargs)
            finally:
                if kwargs.get("🧹为生图释放显存", True) or _should_unload(source_model):
                    _release_model(source_model)

    def _run(self, kwargs):
        internal = str(kwargs.get("📝原始需求") or "").strip()
        external = str(kwargs.get("🔗上游需求") or "").strip()
        brief = "\n".join(part for part in (external, internal) if part)
        if not brief:
            raise ValueError("请填写原始需求或连接上游需求。")
        images = _images(kwargs)
        count = len(images)
        task = kwargs["🧭任务模式"]
        mode = "t2i" if task == "文生图" or (task == "自动识别" and not images) else "edit"
        if mode == "edit" and not images:
            raise ValueError("改图模式至少需要连接一张参考图。")
        if mode == "t2i" and images:
            raise ValueError("文生图模式不接收参考图；请选择自动识别或改图模式。")
        canvas = int(kwargs["🖼️基准图"])
        if mode == "edit" and canvas > count:
            raise ValueError(f"已连接{count}张参考图，基准图不能选{canvas}。")
        ratio = kwargs["📐输出画幅"]
        if mode == "t2i" and ratio == "继承基准图":
            raise ValueError("文生图没有基准图，请选择自动判断或具体画幅。")
        text_policy = kwargs["🔤画面文字"]
        exact_text = str(kwargs.get("✍️自定义准确文字") or "").strip()
        if text_policy == "自定义准确文字" and not exact_text:
            raise ValueError("选择自定义准确文字后，请填写文字内容。")

        model = _sync_model(kwargs["🤖本地模型"])
        if count and model.settings.get("mmproj_file") in ("", "None", "无"):
            raise ValueError("参考图需要视觉模型，请在本地模型加载器中选择配套mmproj。")
        if count and model.chat_handler is None:
            raise ValueError("视觉处理器没有成功加载，请检查主模型与mmproj是否配套。")
        rule_text = _read_rules(mode)
        system = (
            "You are the Qwen-Image-2.1 prompt rewriting expert. Follow the supplied reference rules as the authority. "
            "Return one JSON object only, without markdown. Required keys: rewritten_prompt, wh_ratio, ratio_follow, "
            "optimization_breakdown, tweak_suggestions. The latter two are short Chinese strings (suggestions may be an array). "
            "For T2I, keep English prose and original-script quoted rendered text. For edits, apply the reference language rules. "
            "Keep rewritten_prompt a single paragraph. Ratio and resolution belong only in fields, not the prompt. "
            "Never invent readable text unless explicitly requested.\n\nREFERENCE RULES:\n" + rule_text
        )
        directives = [f"用户原始需求：{brief}", f"任务类型：{task}；实际规则模式：{mode}", f"参考图数量：{count}"]
        if count:
            directives.append(f"参考图依次命名为 <image1> 至 <image{count}>。基准画布是 <image{canvas}>。请先观察各图实际内容；多图时在提示词中逐一说明各图用途。")
        if ratio != "自动判断":
            directives.append(f"画幅选择：{ratio}。严格遵守，比例不要写进 rewritten_prompt。")
        directives.append(f"画面文字策略：{text_policy}。")
        if exact_text and text_policy == "自定义准确文字":
            directives.append(f"必须逐字使用的画面文字：{exact_text}")
        directives.append("输出 JSON；分析与建议简短，不得混入 rewritten_prompt。")
        content = []
        for image in images:
            pil = scale_image(tensor2pil(image).convert("RGB"), 1536)
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image2base64(pil)}})
        content.append({"type": "text", "text": "\n".join(directives)})
        response, _ = _create_completion(
            model.llm,
            [{"role": "system", "content": system}, {"role": "user", "content": content}],
            {"max_tokens": int(kwargs["📤最大输出token"]), "temperature": float(kwargs["🌡️温度"]),
             "top_p": 0.9, "seed": int(kwargs["🎲随机种子"])},
        )
        data = _extract_json(_extract_reply(response))
        payload = _normalize_payload(data, mode, ratio, canvas, count)
        analysis = str(data.get("optimization_breakdown") or "").strip()
        suggestions = data.get("tweak_suggestions") or ""
        if isinstance(suggestions, list):
            suggestions = "\n".join(f"- {str(item).strip()}" for item in suggestions[:3] if str(item).strip())
        notes = f"💡 提示词优化解析\n{analysis}\n\n🎨 进阶微调建议\n{suggestions}".strip()
        json_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        style = kwargs["🧩输出样式"]
        if style == "严格JSON":
            notes = json_text
        elif style == "仅提示词":
            notes = payload["rewritten_prompt"]
        else:
            notes = (f"💡 提示词优化解析\n{analysis}\n画幅：{payload['wh_ratio'] or payload['ratio_follow']}\n\n"
                     f"📋 提示词（可直接复制）\n{payload['rewritten_prompt']}\n\n🎨 进阶微调建议\n{suggestions}")
        return (payload["rewritten_prompt"], payload["wh_ratio"], payload["ratio_follow"], json_text, notes)


NODE_CLASS_MAPPINGS = {"DapaoQwenImage21Prompt": DapaoQwenImage21Prompt}
NODE_DISPLAY_NAME_MAPPINGS = {"DapaoQwenImage21Prompt": "🧳Qwen-image2.1提示词优化@炮老师的小课堂"}
