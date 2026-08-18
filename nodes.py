import os
import io
import gc
import json
import base64
import re
import sys

import numpy as np
import torch
from PIL import Image

import folder_paths
import comfy.model_management as mm

from .gguf_layers import get_layer_count, get_gguf_model_info
from .cqdm import cqdm

# ── LLM 文件夹注册 ──────────────────────────────────────────────────────────
llm_extensions = {".gguf", ".bin"}
llm_dir = os.path.join(folder_paths.models_dir, "LLM")
os.makedirs(llm_dir, exist_ok=True)
folder_paths.folder_names_and_paths["LLM"] = ([llm_dir], llm_extensions)

# ── AnyType ─────────────────────────────────────────────────────────────────
class AnyType(str):
    def __ne__(self, other):
        return False

any_type = AnyType("*")

# ── 按需 import 各 ChatHandler（兼容不同版本 llama-cpp-python）──────────────
import llama_cpp
from llama_cpp import Llama
from llama_cpp.llama_chat_format import (
    Llava15ChatHandler, Llava16ChatHandler, MoondreamChatHandler,
    NanoLlavaChatHandler, Llama3VisionAlphaChatHandler, MiniCPMv26ChatHandler,
)

try:
    from llama_cpp.llama_chat_format import MTMDChatHandler
    _MTMD = True
except Exception:
    _MTMD = False

try:
    from llama_cpp.llama_chat_format import Gemma3ChatHandler
except Exception:
    Gemma3ChatHandler = None

try:
    from llama_cpp.llama_chat_format import Gemma4ChatHandler
except Exception:
    if _MTMD:
        Gemma4ChatHandler = MTMDChatHandler
    else:
        Gemma4ChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
except Exception:
    Qwen25VLChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen3VLChatHandler
except Exception:
    Qwen3VLChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen35ChatHandler
except Exception:
    Qwen35ChatHandler = None

try:
    from llama_cpp.llama_chat_format import (
        Jinja2ChatFormatter,
        chat_formatter_to_chat_completion_handler,
    )
except Exception:
    Jinja2ChatFormatter = None
    chat_formatter_to_chat_completion_handler = None

try:
    from llama_cpp.llama_chat_format import GLM46VChatHandler, GLM41VChatHandler
except Exception:
    GLM46VChatHandler = None
    GLM41VChatHandler = None

try:
    from llama_cpp.llama_chat_format import LFM2VLChatHandler
except Exception:
    LFM2VLChatHandler = None

try:
    from llama_cpp.llama_chat_format import GraniteDoclingChatHandler
except Exception:
    GraniteDoclingChatHandler = None

# ── 可用 handler 列表（固定顺序，始终包含全部，运行时若 import 失败会报错）──
CHAT_HANDLERS = [
    "None",
    "LLaVA-1.5", "LLaVA-1.6", "Moondream2", "nanoLLaVA", "llama3-Vision-Alpha",
    "MiniCPM-v2.6", "MiniCPM-v4.5", "MiniCPM-v4.5-Thinking",
    "Gemma3", "Gemma4",
    "Qwen2.5-VL",
    "Qwen3.8",
    "Qwen3-VL", "Qwen3-VL-Thinking",
    "Qwen3.5", "Qwen3.5-Thinking",
    "GLM-4.6V", "GLM-4.6V-Thinking", "GLM-4.1V-Thinking",
    "LFM2-VL",
    "Granite-Docling",
]

_QWEN_HANDLER_ARCHITECTURES = {
    "Qwen2.5-VL": "qwen2vl",
    "Qwen3.8": "qwen35",
    "Qwen3-VL": "qwen3vl",
    "Qwen3-VL-Thinking": "qwen3vl",
    "Qwen3.5": "qwen35",
    "Qwen3.5-Thinking": "qwen35",
}

_QWEN_ARCHITECTURE_HANDLERS = {
    "qwen2vl": "Qwen2.5-VL",
    "qwen3vl": "Qwen3-VL",
    "qwen35": "Qwen3.5",
}

_QWEN38_MIN_LLAMA_CPP_VERSION = (0, 3, 47)
QWEN38_REASONING_OPTIONS = ["关闭", "自动", "低", "中等", "高"]
_QWEN38_REASONING_VALUES = {
    "关闭": "off",
    "自动": "xhigh",
    "低": "low",
    "中等": "medium",
    "高": "xhigh",
    # Older workflows and the reference node use the native English values.
    "off": "off",
    "xhigh": "xhigh",
    "medium": "medium",
    "low": "low",
}

_VRAM_TOOLTIP = (
    "这是LLM可使用的显存预算，不是预留空间。-1=尝试全部放入GPU，最快但可能因显存不足失败；"
    "填写数值=只将部分模型层放入GPU，其余使用系统内存。参考起点：8GB显卡填6，12GB填10，"
    "16GB填13，24GB填20，32GB填24-28。请为ComfyUI、mmproj和上下文缓存保留约2GB。"
)


def _llama_cpp_version():
    return str(getattr(llama_cpp, "__version__", "未知"))


def _version_tuple(version):
    numbers = re.findall(r"\d+", str(version))
    if len(numbers) < 3:
        return None
    return tuple(int(number) for number in numbers[:3])


def normalize_qwen38_reasoning_effort(value):
    if value is None:
        return "off"
    try:
        return _QWEN38_REASONING_VALUES[str(value)]
    except KeyError as error:
        raise ValueError(f"未知 Qwen3.8 推理强度：{value}") from error


def _validate_qwen38_backend():
    version = _llama_cpp_version()
    parsed = _version_tuple(version)
    if parsed is not None and parsed < _QWEN38_MIN_LLAMA_CPP_VERSION:
        raise RuntimeError(
            "Qwen3.8 需要支持 MTP/NextN 张量的新版 llama-cpp-python。"
            f"当前版本为 {version}，最低需要 0.3.47；旧版会在加载时缺少 "
            "blk.64.ssm_conv1d.weight。请按本节点 README 安装与你的 Python、"
            "操作系统和 CUDA 匹配的 JamePeng 0.3.47+ wheel，然后重启 ComfyUI。"
            f"当前 Python：{sys.executable}"
        )


def _normalize_handler_name(handler_name):
    """Migrate the temporary Qwen3-8B label used by older workflows."""
    return "Qwen3.8" if handler_name == "Qwen3-8B" else handler_name


def _should_analyze_images_individually(user_prompt):
    """Keep comparison tasks in one request; otherwise analyze static images separately."""
    comparison_keywords = (
        "比较", "对比", "区别", "差异", "相同", "不同", "关联", "关系",
        "排序", "挑选", "哪张", "所有图片", "整体", "共同",
    )
    return not any(keyword in user_prompt for keyword in comparison_keywords)


def _select_inference_strategy(frame_count, has_video_input, has_audio_input, user_prompt):
    if has_video_input:
        return "视频帧联合分析"
    if has_audio_input and frame_count:
        return "图文音联合分析"
    if frame_count > 1 and _should_analyze_images_individually(user_prompt):
        return "多图逐张分析"
    if frame_count > 1:
        return "多图联合分析"
    if frame_count:
        return "单图分析"
    if has_audio_input:
        return "音频分析"
    return "纯文本分析"


def _validate_multimodal_pair(model_path, model_file, handler_name, mmproj_path, mmproj_file):
    """Reject incompatible model/handler/mmproj combinations before native loading."""
    if not mmproj_path or handler_name == "None":
        return

    model_info = get_gguf_model_info(model_path)
    mmproj_info = get_gguf_model_info(mmproj_path)
    model_arch = model_info["architecture"]
    expected_arch = _QWEN_HANDLER_ARCHITECTURES.get(handler_name)

    if expected_arch and model_arch and model_arch != expected_arch:
        recommended = _QWEN_ARCHITECTURE_HANDLERS.get(model_arch)
        recommendation = f"，该主模型应选择“{recommended}”" if recommended else ""
        raise ValueError(
            "模型与对话处理器不匹配："
            f"“{model_file}”的 GGUF 架构是 {model_arch}，"
            f"当前处理器“{handler_name}”要求 {expected_arch}{recommendation}。"
            "请同时选择与主模型同系列、同参数规模的 mmproj 文件。"
        )

    model_dimension = model_info["dimension"]
    projection_dimension = mmproj_info["dimension"]
    if (
        model_dimension is not None
        and projection_dimension is not None
        and int(model_dimension) != int(projection_dimension)
    ):
        raise ValueError(
            "主模型与 mmproj 文件不匹配："
            f"“{model_file}”的嵌入维度是 {model_dimension}，"
            f"“{mmproj_file}”的投影维度是 {projection_dimension}。"
            "请选择与主模型同系列、同参数规模的 mmproj 文件。"
        )

# ── 模型状态管理 ─────────────────────────────────────────────────────────────
class DapaoLlamaStorage:
    llm = None
    chat_handler = None
    current_config = None
    messages = {}
    sys_prompts = {}

    @classmethod
    def clean(cls, all=False):
        if cls.llm is not None:
            del cls.llm
            cls.llm = None
        cls.chat_handler = None
        cls.current_config = None
        if all:
            cls.messages = {}
            cls.sys_prompts = {}
        gc.collect()

    @classmethod
    def clean_state(cls, uid=-1):
        if uid == -1:
            cls.messages = {}
            cls.sys_prompts = {}
        else:
            cls.messages.pop(str(uid), None)
            cls.sys_prompts.pop(str(uid), None)


# ── patch mm.unload_all_models（只 patch 一次）──────────────────────────────
if not hasattr(mm, "_dapao_llama_unload_backup"):
    mm._dapao_llama_unload_backup = mm.unload_all_models
    def _patched_unload(*args, **kwargs):
        DapaoLlamaStorage.clean(all=True)
        return mm._dapao_llama_unload_backup(*args, **kwargs)
    mm.unload_all_models = _patched_unload
    print("[大炮-llama] 模型卸载钩子已挂载")

# ── 图像工具 ─────────────────────────────────────────────────────────────────
def tensor2pil(tensor):
    """[1,H,W,C] float32 0-1 → PIL Image"""
    img = tensor.squeeze(0).cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


def scale_image(pil_img, max_size):
    w, h = pil_img.size
    if max(w, h) <= max_size:
        return pil_img
    scale = max_size / max(w, h)
    return pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def image2base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def audio2base64(audio_dict):
    """ComfyUI AUDIO dict → base64 WAV string（纯 Python wave 模块，不依赖 torchaudio）"""
    import wave
    import struct
    waveform = audio_dict["waveform"]   # [B, C, T]
    sample_rate = audio_dict["sample_rate"]
    wav = waveform[0].cpu()             # [C, T]
    # 混合为单声道或保留立体声
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav[0]  # [T]
    # 转为 16-bit PCM
    pcm = (wav.clamp(-1.0, 1.0) * 32767).short().numpy()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(pcm)}h", *pcm))
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _create_multimodal_handler(handler_class, mmproj_path, **kwargs):
    """Support both llama-cpp handler constructor spellings."""
    try:
        return handler_class(mmproj_path=mmproj_path, **kwargs)
    except TypeError as error:
        message = str(error)
        if "mmproj_path" not in message and "clip_model_path" not in message:
            raise
        return handler_class(clip_model_path=mmproj_path, **kwargs)


def _adapt_qwen38_chat_template(chat_template):
    """Adapt Qwen3.8's image_pad token to llama.cpp's MTMD image URL input."""
    if not chat_template or "<|image_pad|>" not in chat_template:
        return chat_template

    pattern = (
        r"\{\{-?\s*(['\"])<\|vision_start\|><\|image_pad\|><\|vision_end\|>\1\s*-?\}\}"
    )
    replacement = (
        "{{- '<|vision_start|>' }}"
        "{%- if item.image_url is string %}"
        "{{- item.image_url }}"
        "{%- else %}"
        "{{- item.image_url.url }}"
        "{%- endif %}"
        "{{- '<|vision_end|>' }}"
    )
    adapted, count = re.subn(pattern, replacement, chat_template)
    if count == 0:
        raise RuntimeError(
            "Qwen3.8 聊天模板包含 <|image_pad|>，但无法适配当前 llama.cpp 多模态处理器。"
        )
    return adapted


def _create_qwen38_mm_handler(
    mmproj_path, *, enable_thinking, preserve_thinking, reasoning_effort,
    chat_template_override, image_min_tokens, image_max_tokens,
):
    if Qwen35ChatHandler is None:
        raise RuntimeError("Qwen3.8 需要 Qwen35ChatHandler，请升级 llama-cpp-python。")

    shared = {
        "verbose": False,
        "extra_template_arguments": {"reasoning_effort": reasoning_effort},
        "chat_template_override": chat_template_override,
    }
    if _MTMD:
        shared.update(
            image_min_tokens=image_min_tokens,
            image_max_tokens=image_max_tokens,
        )
    candidates = [
        {"enable_thinking": enable_thinking, "preserve_thinking": preserve_thinking,
         "add_vision_id": True, **shared},
        {"enable_thinking": enable_thinking, "preserve_thinking": preserve_thinking, **shared},
        {"enable_thinking": enable_thinking, "add_vision_id": True, **shared},
        {"enable_thinking": enable_thinking, **shared},
    ]
    last_error = None
    for kwargs in candidates:
        try:
            return _create_multimodal_handler(Qwen35ChatHandler, mmproj_path, **kwargs)
        except TypeError as error:
            last_error = error
    raise last_error or RuntimeError("创建 Qwen3.8 多模态处理器失败。")


def _create_qwen38_text_handler(llm, *, enable_thinking, preserve_thinking, reasoning_effort):
    if Jinja2ChatFormatter is None or chat_formatter_to_chat_completion_handler is None:
        raise RuntimeError("当前 llama-cpp-python 不支持 Qwen3.8 聊天模板，请升级 llama-cpp-python。")

    metadata = getattr(llm, "metadata", {}) or {}
    chat_template = metadata.get("tokenizer.chat_template")
    if not chat_template:
        raise RuntimeError("Qwen3.8 GGUF 缺少 tokenizer.chat_template。")

    model = getattr(llm, "_model", None)

    def token_text(token_id):
        if token_id == -1 or model is None or not hasattr(model, "token_get_text"):
            return ""
        return model.token_get_text(token_id)

    stop_token_ids = [
        token_id
        for token_id in (llm.token_eos(), llm.token_eot())
        if token_id != -1
    ] or None
    formatter = Jinja2ChatFormatter(
        template=chat_template,
        eos_token=token_text(llm.token_eos()),
        bos_token=token_text(llm.token_bos()),
        stop_token_ids=stop_token_ids,
    )

    def qwen38_formatter(*, messages, **kwargs):
        kwargs.update(
            enable_thinking=enable_thinking,
            preserve_thinking=preserve_thinking,
            reasoning_effort=reasoning_effort,
        )
        return formatter(messages=messages, **kwargs)

    return chat_formatter_to_chat_completion_handler(qwen38_formatter)


# ── 主节点 ───────────────────────────────────────────────────────────────────
class Dapao_LlamaChat:
    CATEGORY = "🍭大炮-llama-cpp"
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("💬回复文本", "📋完整对话历史", "🔢使用的种子")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        llm_files = folder_paths.get_filename_list("LLM")
        mmproj_files = ["None"] + [f for f in llm_files if "mmproj" in f.lower()]
        model_files = [f for f in llm_files if "mmproj" not in f.lower()]

        return {
            "required": {
                # ── 模型加载 ──
                "🤖模型文件": (model_files,),
                "🔌对话处理器": (CHAT_HANDLERS, {"default": "None"}),
                "🖼️mmproj文件": (mmproj_files, {"default": "None"}),
                "📐上下文长度": ("INT", {"default": 8192, "min": 512, "max": 131072, "step": 512}),
                "💾显存限制(GB)": (
                    "FLOAT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 999.0,
                        "step": 0.5,
                        "tooltip": _VRAM_TOOLTIP,
                    },
                ),
                "🔢图像最小token": ("INT", {"default": 256, "min": 1, "max": 4096, "step": 1}),
                "🔢图像最大token": ("INT", {"default": 1344, "min": 1, "max": 8192, "step": 1}),
                # ── 提示词 ──
                "📝系统提示词": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "💬用户提示词": ("STRING", {"default": "请描述这张图片。", "multiline": True}),
                # ── 推理参数 ──
                "🎞️最大帧数": ("INT", {"default": 10, "min": 1, "max": 200, "step": 1}),
                "📏图像最大边长": ("INT", {"default": 1120, "min": 64, "max": 4096, "step": 32}),
                # ── 生成参数 ──
                "🎲随机种子": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "📊最大输出token": ("INT", {"default": 1024, "min": 1, "max": 32768, "step": 1}),
                "🌡️温度": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "🎯top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "🔝top_k": ("INT", {"default": 40, "min": 0, "max": 200, "step": 1}),
                "🔁重复惩罚": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 2.0, "step": 0.01}),
                "🧠思考模式": ("BOOLEAN", {"default": False, "tooltip": "开启后模型会输出思考过程（仅 Thinking 系列模型有效）"}),
                "🧠Qwen3.8推理强度": (QWEN38_REASONING_OPTIONS, {"default": "关闭", "tooltip": "仅 Qwen3.8 生效；关闭=不思考，自动/高=模型最高档，低/中等=降低思考强度。"}),
                "💾保存对话历史": ("BOOLEAN", {"default": False}),
                "⚡推理后卸载模型": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "🖼️图像1": ("IMAGE",),
                "🖼️图像2": ("IMAGE",),
                "🖼️图像3": ("IMAGE",),
                "🖼️图像4": ("IMAGE",),
                "🖼️图像5": ("IMAGE",),
                "🖼️图像6": ("IMAGE",),
                "🖼️图像7": ("IMAGE",),
                "🖼️图像8": ("IMAGE",),
                "🎬视频1": ("IMAGE",),
                "🎬视频2": ("IMAGE",),
                "🔊音频1": ("AUDIO",),
                "🔊音频2": ("AUDIO",),
                "🔗队列处理器": (any_type,),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def _load_model(self, model_file, handler_name, mmproj_file, n_ctx, vram_limit_gb,
                    image_min_tokens, image_max_tokens, think_mode=False,
                    reasoning_effort="off"):
        handler_name = _normalize_handler_name(handler_name)
        reasoning_effort = normalize_qwen38_reasoning_effort(reasoning_effort)
        model_path = os.path.join(folder_paths.models_dir, "LLM", model_file)
        mmproj_path = None
        if mmproj_file and mmproj_file != "None":
            mmproj_path = os.path.join(folder_paths.models_dir, "LLM", mmproj_file)

        if handler_name == "Qwen3.8":
            _validate_qwen38_backend()

        _validate_multimodal_pair(
            model_path, model_file, handler_name, mmproj_path, mmproj_file
        )

        # ── n_gpu_layers 计算（与参考节点一致，含 1.55 系数）────────────────
        layer_count = get_layer_count(model_path) or 32
        n_gpu_layers = -1
        if vram_limit_gb != -1:
            model_size_gb = os.path.getsize(model_path) * 1.55 / (1024 ** 3)
            layer_size_gb = model_size_gb / layer_count

            if mmproj_path:
                mmproj_size_gb = os.path.getsize(mmproj_path) * 1.55 / (1024 ** 3)
                n_gpu_layers = max(1, int((vram_limit_gb - mmproj_size_gb) / layer_size_gb))
            else:
                n_gpu_layers = max(1, int(vram_limit_gb / layer_size_gb))

        print(f"[大炮-llama] 加载模型: {model_file}  n_gpu_layers={n_gpu_layers}")
        if n_gpu_layers != -1 and n_gpu_layers < layer_count:
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                free_gb = free_bytes / (1024 ** 3)
                total_gb = total_bytes / (1024 ** 3)
                suggested_gb = max(1, int(free_gb - 2.0))
                gpu_hint = (
                    f"当前CUDA可用显存约 {free_gb:.1f}/{total_gb:.1f} GB；"
                    f"若没有其他即将执行的GPU节点，可尝试把显存限制提高到约 {suggested_gb} GB。"
                )
            except Exception:
                gpu_hint = "若显卡仍有空闲显存，可逐步提高“显存限制(GB)”。"
            print(
                f"[大炮-llama] 性能提示：当前仅 {n_gpu_layers}/{layer_count} 层在GPU，"
                f"其余层由CPU计算，速度会明显下降。{gpu_hint}"
            )

        # ── 实例化 ChatHandler ────────────────────────────────────────────────
        chat_handler = None
        # think_mode 由外部参数控制，不再依赖 handler 名字里的 "Thinking" 后缀

        if mmproj_path and handler_name != "None":
            kwargs = {"clip_model_path": mmproj_path, "verbose": False}

            if handler_name in ("Qwen3-VL", "Qwen3-VL-Thinking"):
                if Qwen3VLChatHandler is None:
                    raise RuntimeError("Qwen3VLChatHandler 未找到，请升级 llama-cpp-python")
                kwargs["force_reasoning"] = think_mode
                kwargs["image_max_tokens"] = image_max_tokens
                kwargs["image_min_tokens"] = image_min_tokens
                chat_handler = Qwen3VLChatHandler(**kwargs)

            elif handler_name == "Qwen2.5-VL":
                if Qwen25VLChatHandler is None:
                    raise RuntimeError("Qwen25VLChatHandler 未找到，请升级 llama-cpp-python")
                if _MTMD:
                    kwargs["image_max_tokens"] = image_max_tokens
                    kwargs["image_min_tokens"] = image_min_tokens
                chat_handler = Qwen25VLChatHandler(**kwargs)

            elif handler_name in ("Qwen3.5", "Qwen3.5-Thinking"):
                if Qwen35ChatHandler is None:
                    raise RuntimeError("Qwen35ChatHandler 未找到，请升级 llama-cpp-python")
                kwargs["enable_thinking"] = think_mode
                if _MTMD:
                    kwargs["image_max_tokens"] = image_max_tokens
                    kwargs["image_min_tokens"] = image_min_tokens
                chat_handler = Qwen35ChatHandler(**kwargs)

            elif handler_name in ("MiniCPM-v4.5", "MiniCPM-v4.5-Thinking"):
                kwargs["enable_thinking"] = think_mode
                if _MTMD:
                    kwargs["image_max_tokens"] = image_max_tokens
                    kwargs["image_min_tokens"] = image_min_tokens
                chat_handler = MiniCPMv26ChatHandler(**kwargs)

            elif handler_name == "Gemma3":
                if Gemma3ChatHandler is None:
                    raise RuntimeError("Gemma3ChatHandler 未找到，请升级 llama-cpp-python")
                if _MTMD:
                    kwargs["image_max_tokens"] = image_max_tokens
                    kwargs["image_min_tokens"] = image_min_tokens
                chat_handler = Gemma3ChatHandler(**kwargs)

            elif handler_name == "Gemma4":
                if Gemma4ChatHandler is None:
                    raise RuntimeError("Gemma4ChatHandler 未找到，请升级 llama-cpp-python")
                kwargs["enable_thinking"] = think_mode
                kwargs["image_max_tokens"] = image_max_tokens
                kwargs["image_min_tokens"] = image_min_tokens
                chat_handler = Gemma4ChatHandler(**kwargs)

            elif handler_name in ("GLM-4.6V", "GLM-4.6V-Thinking"):
                if GLM46VChatHandler is None:
                    raise RuntimeError("GLM46VChatHandler 未找到，请升级 llama-cpp-python")
                kwargs["enable_thinking"] = think_mode
                if _MTMD:
                    kwargs["image_max_tokens"] = image_max_tokens
                    kwargs["image_min_tokens"] = image_min_tokens
                chat_handler = GLM46VChatHandler(**kwargs)

            elif handler_name == "GLM-4.1V-Thinking":
                if GLM41VChatHandler is None:
                    raise RuntimeError("GLM41VChatHandler 未找到，请升级 llama-cpp-python")
                kwargs["enable_thinking"] = think_mode
                if _MTMD:
                    kwargs["image_max_tokens"] = image_max_tokens
                    kwargs["image_min_tokens"] = image_min_tokens
                chat_handler = GLM41VChatHandler(**kwargs)

            elif handler_name == "LFM2-VL":
                if LFM2VLChatHandler is None:
                    raise RuntimeError("LFM2VLChatHandler 未找到，请升级 llama-cpp-python")
                if _MTMD:
                    kwargs["image_max_tokens"] = image_max_tokens
                    kwargs["image_min_tokens"] = image_min_tokens
                chat_handler = LFM2VLChatHandler(**kwargs)

            elif handler_name == "Granite-Docling":
                if GraniteDoclingChatHandler is None:
                    raise RuntimeError("GraniteDoclingChatHandler 未找到，请升级 llama-cpp-python")
                if _MTMD:
                    kwargs["image_max_tokens"] = image_max_tokens
                    kwargs["image_min_tokens"] = image_min_tokens
                chat_handler = GraniteDoclingChatHandler(**kwargs)

            elif handler_name == "LLaVA-1.5":
                chat_handler = Llava15ChatHandler(**kwargs)
            elif handler_name == "LLaVA-1.6":
                chat_handler = Llava16ChatHandler(**kwargs)
            elif handler_name == "Moondream2":
                chat_handler = MoondreamChatHandler(**kwargs)
            elif handler_name == "nanoLLaVA":
                chat_handler = NanoLlavaChatHandler(**kwargs)
            elif handler_name == "llama3-Vision-Alpha":
                chat_handler = Llama3VisionAlphaChatHandler(**kwargs)
            elif handler_name == "MiniCPM-v2.6":
                chat_handler = MiniCPMv26ChatHandler(**kwargs)

        elif handler_name not in ("None", "Qwen3.8", "Qwen3.5", "Qwen3.5-Thinking"):
            # 无 mmproj 但有 handler（纯文本模式下某些 handler 可无 mmproj）
            pass

        # ── 加载 Llama ────────────────────────────────────────────────────────
        llm = None
        try:
            llm = Llama(
                model_path=model_path,
                chat_handler=chat_handler,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=False,
            )

            if handler_name == "Qwen3.8":
                chat_template = (getattr(llm, "metadata", {}) or {}).get(
                    "tokenizer.chat_template"
                )
                if not chat_template:
                    raise RuntimeError("Qwen3.8 GGUF 缺少 tokenizer.chat_template。")
                if mmproj_path:
                    chat_handler = _create_qwen38_mm_handler(
                        mmproj_path,
                        enable_thinking=think_mode and reasoning_effort != "off",
                        preserve_thinking=False,
                        reasoning_effort=reasoning_effort,
                        chat_template_override=_adapt_qwen38_chat_template(chat_template),
                        image_min_tokens=image_min_tokens,
                        image_max_tokens=image_max_tokens,
                    )
                else:
                    chat_handler = _create_qwen38_text_handler(
                        llm,
                        enable_thinking=think_mode and reasoning_effort != "off",
                        preserve_thinking=False,
                        reasoning_effort=reasoning_effort,
                    )
                llm.chat_handler = chat_handler

            # MTMD 默认延迟到首次推理才加载。这里提前验证，避免缓存无效处理器。
            if chat_handler is not None and hasattr(chat_handler, "_init_mtmd_context"):
                try:
                    chat_handler._init_mtmd_context(llm)
                except ValueError as gpu_error:
                    if not getattr(chat_handler, "use_gpu", False):
                        raise
                    print("[大炮-llama] GPU 视觉编码器初始化失败，自动改用 CPU 重试")
                    chat_handler.use_gpu = False
                    try:
                        chat_handler._init_mtmd_context(llm)
                    except ValueError as cpu_error:
                        raise ValueError(
                            "多模态投影加载失败："
                            f"主模型“{model_file}”，mmproj“{mmproj_file}”。"
                            "请确认两者来自同一模型系列和参数规模，且 GGUF 文件完整。"
                        ) from cpu_error
        except Exception as load_error:
            if chat_handler is not None and hasattr(chat_handler, "close"):
                chat_handler.close()
            if llm is not None and hasattr(llm, "close"):
                llm.close()
            if isinstance(load_error, ValueError) and "Failed to load model from file" in str(load_error):
                try:
                    model_size_gb = os.path.getsize(model_path) / (1024 ** 3)
                except OSError:
                    model_size_gb = None
                try:
                    mmproj_size_gb = (
                        os.path.getsize(mmproj_path) / (1024 ** 3)
                        if mmproj_path else 0.0
                    )
                except OSError:
                    mmproj_size_gb = None
                size_parts = []
                if model_size_gb is not None:
                    size_parts.append(f"主模型文件约 {model_size_gb:.1f} GB")
                if mmproj_size_gb is not None and mmproj_size_gb > 0:
                    size_parts.append(f"mmproj约 {mmproj_size_gb:.1f} GB")
                size_hint = "，".join(size_parts) + "。" if size_parts else ""
                estimated = (
                    (model_size_gb or 0.0) * 1.55 + (mmproj_size_gb or 0.0) * 1.55
                    if model_size_gb is not None and mmproj_size_gb is not None
                    else None
                )
                estimate_hint = (
                    f"粗略加载预算约 {estimated:.1f} GB（还未计入上下文缓存）。"
                    if estimated is not None else ""
                )
                if n_gpu_layers == -1:
                    raise RuntimeError(
                        f"模型加载失败：{size_hint}{estimate_hint}"
                        "当前“显存限制(GB)”为 -1，表示尝试全部放入GPU；"
                        "这通常是显存不足，而不是提示词或图片输入错误。"
                        "请把该参数改成实际剩余显存的一部分（新手可先填8），"
                        "让模型自动部分卸载到系统内存后重试；显存有余量时再逐步提高。"
                        "同时确认主模型与mmproj来自同一模型系列。"
                    ) from load_error
                raise RuntimeError(
                    f"模型加载失败：{size_hint}{estimate_hint}"
                    f"当前显存预算为 {vram_limit_gb:.1f} GB（n_gpu_layers={n_gpu_layers}）。"
                    "如果显卡还有空余，请适当提高显存预算；如果仍然显存不足，请降低预算，"
                    "并关闭其他占用显存的节点或程序。若调整后仍失败，再检查GGUF下载完整性、"
                    "CUDA后端和主模型/mmproj配对。"
                ) from load_error
            raise

        DapaoLlamaStorage.llm = llm
        DapaoLlamaStorage.chat_handler = chat_handler
        DapaoLlamaStorage.current_config = {
            "model_file": model_file,
            "handler_name": handler_name,
            "mmproj_file": mmproj_file,
            "n_ctx": n_ctx,
            "vram_limit_gb": vram_limit_gb,
            "image_min_tokens": image_min_tokens,
            "image_max_tokens": image_max_tokens,
            "think_mode": think_mode,
            "reasoning_effort": reasoning_effort,
        }
        return llm

    def run(self, unique_id, **kwargs):
        model_file       = kwargs["🤖模型文件"]
        handler_name     = _normalize_handler_name(kwargs["🔌对话处理器"])
        mmproj_file      = kwargs["🖼️mmproj文件"]
        n_ctx            = kwargs["📐上下文长度"]
        vram_limit_gb    = kwargs["💾显存限制(GB)"]
        image_min_tokens = kwargs["🔢图像最小token"]
        image_max_tokens = kwargs["🔢图像最大token"]
        system_prompt    = kwargs["📝系统提示词"]
        user_prompt      = kwargs["💬用户提示词"]
        max_frames       = kwargs["🎞️最大帧数"]
        max_size         = kwargs["📏图像最大边长"]
        seed             = kwargs["🎲随机种子"]
        max_tokens       = kwargs["📊最大输出token"]
        temperature      = kwargs["🌡️温度"]
        top_p            = kwargs["🎯top_p"]
        top_k            = kwargs["🔝top_k"]
        repeat_penalty   = kwargs["🔁重复惩罚"]
        think_mode       = kwargs["🧠思考模式"]
        reasoning_effort = normalize_qwen38_reasoning_effort(
            kwargs.get("🧠Qwen3.8推理强度", "关闭")
        )
        save_states      = kwargs["💾保存对话历史"]
        force_offload    = kwargs["⚡推理后卸载模型"]
        uid              = str(unique_id)

        # ── 收集所有图像输入（8个图像口 + 2个视频口）────────────────────────
        image_slots = [
            kwargs.get("🖼️图像1"), kwargs.get("🖼️图像2"),
            kwargs.get("🖼️图像3"), kwargs.get("🖼️图像4"),
            kwargs.get("🖼️图像5"), kwargs.get("🖼️图像6"),
            kwargs.get("🖼️图像7"), kwargs.get("🖼️图像8"),
        ]
        video_slots = [kwargs.get("🎬视频1"), kwargs.get("🎬视频2")]
        audio_slots = [kwargs.get("🔊音频1"), kwargs.get("🔊音频2")]

        # 各 tensor 单独保留，不拼接（不同尺寸图片无法 cat）
        all_image_tensors = [t for t in image_slots + video_slots if t is not None]
        has_video_input = any(t is not None for t in video_slots)
        active_audio = [a for a in audio_slots if a is not None]
        input_frame_count = min(
            max_frames,
            sum(int(tensor.shape[0]) for tensor in all_image_tensors),
        )
        inference_strategy = _select_inference_strategy(
            input_frame_count, has_video_input, bool(active_audio), user_prompt,
        )

        # Qwen3.8 的视觉 token 和输出 token 都占用同一个上下文；旧工作流常保存
        # 很小的 n_ctx，必须在模型加载前按本次请求自动提高，避免 MTMD Context Shift。
        if input_frame_count:
            images_per_request = 1 if inference_strategy == "多图逐张分析" else input_frame_count
            minimum_n_ctx = image_max_tokens * images_per_request + max_tokens + 512
            if minimum_n_ctx > 131072:
                raise ValueError(
                    "当前图片/视频帧数量与图像最大 token 设置需要超过 131072 的上下文。"
                    "请降低“最大帧数”或“图像最大token”后重试。"
                )
            if n_ctx < minimum_n_ctx:
                print(
                    f"[大炮-llama] 自动提高上下文：{n_ctx} -> {minimum_n_ctx} "
                    f"（策略={inference_strategy}）"
                )
                n_ctx = minimum_n_ctx

        # ── 加载或复用模型 ────────────────────────────────────────────────────
        need_load = DapaoLlamaStorage.llm is None
        if not need_load:
            cfg = DapaoLlamaStorage.current_config or {}
            need_load = (
                cfg.get("model_file") != model_file
                or cfg.get("handler_name") != handler_name
                or cfg.get("mmproj_file") != mmproj_file
                or cfg.get("n_ctx") != n_ctx
                or cfg.get("vram_limit_gb") != vram_limit_gb
                or cfg.get("image_min_tokens") != image_min_tokens
                or cfg.get("image_max_tokens") != image_max_tokens
                or cfg.get("think_mode") != think_mode
                or cfg.get("reasoning_effort", "xhigh") != reasoning_effort
            )

        if need_load:
            DapaoLlamaStorage.clean()
            mm.soft_empty_cache()
            self._load_model(model_file, handler_name, mmproj_file, n_ctx,
                             vram_limit_gb, image_min_tokens, image_max_tokens,
                             think_mode, reasoning_effort)
            DapaoLlamaStorage.clean_state(uid)
        else:
            print("[大炮-llama] 复用已加载模型")

        llm = DapaoLlamaStorage.llm

        # ── 对话历史管理 ──────────────────────────────────────────────────────
        prev_sys = DapaoLlamaStorage.sys_prompts.get(uid, "")
        if not save_states or prev_sys != system_prompt:
            DapaoLlamaStorage.clean_state(uid)
            DapaoLlamaStorage.sys_prompts[uid] = system_prompt

        messages = DapaoLlamaStorage.messages.get(uid, [])
        if not messages:
            messages = [{"role": "system", "content": system_prompt}]

        # ── 构建图像帧列表 ────────────────────────────────────────────────────
        frame_list = []
        for tensor in all_image_tensors:
            for i in range(tensor.shape[0]):
                if len(frame_list) >= max_frames:
                    break
                pil_img = tensor2pil(tensor[i:i+1])
                pil_img = scale_image(pil_img, max_size)
                frame_list.append(pil_img)
            if len(frame_list) >= max_frames:
                break

        # 音频输入
        if active_audio:
            print(f"[大炮-llama] 检测到 {len(active_audio)} 个音频输入")
        if frame_list:
            print(f"[大炮-llama] 检测到 {len(frame_list)} 张图像/视频帧，最大处理数={max_frames}")

        _params = dict(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            seed=seed,
        )

        def _infer(msgs, pil_frames, audio_list=None, image_index=None, image_total=None):
            user_content = []
            for pil_img in pil_frames:
                b64 = image2base64(pil_img)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            if audio_list:
                for audio_dict in audio_list:
                    try:
                        b64 = audio2base64(audio_dict)
                        user_content.append({
                            "type": "input_audio",
                            "input_audio": {"data": b64, "format": "wav"},
                        })
                    except Exception as e:
                        print(f"[大炮-llama] 音频编码失败: {e}")
            prompt_text = user_prompt
            if image_index is not None and image_total is not None:
                prompt_text = (
                    f"这是一个多图逐张分析任务。当前正在分析第 {image_index}/{image_total} 张图片。"
                    "本次请求只附带当前这一张图片，这是预期行为，不是漏传图片。"
                    "请直接分析当前图片，不要回复‘只收到一张图片’、不要要求重新上传，"
                    f"并在回答开头标注‘图片 {image_index}/{image_total}’。\n\n"
                    f"用户要求：{user_prompt}"
                )
            user_content.append({"type": "text", "text": prompt_text})
            msgs.append({"role": "user", "content": user_content})
            mm.throw_exception_if_processing_interrupted()
            resp = llm.create_chat_completion(messages=msgs, **_params)
            text = (resp["choices"][0]["message"]["content"] or "").removeprefix(": ").lstrip()
            import re
            # 思考模式关闭时，剥离各模型的思考块，只保留最终回答
            if not think_mode:
                # Qwen/GLM 系列: <think>...</think>
                m = re.search(r"<think>.*?</think>(.*)", text, re.DOTALL)
                if m:
                    text = m.group(1).strip()
                # Gemma4 系列: <|channel>thought\n...<channel|>
                m = re.search(r"<\|channel>thought\n.*?<channel\|>(.*)", text, re.DOTALL)
                if m:
                    text = m.group(1).strip()
            msgs.append({"role": "assistant", "content": text})
            return text

        print(f"[大炮-llama] 推理开始  seed={seed}  自动策略={inference_strategy}")

        if inference_strategy == "多图逐张分析":
            replies = []
            total_frames = len(frame_list)
            for image_index, pil_img in enumerate(cqdm(frame_list, desc="逐帧推理"), start=1):
                frame_msgs = [{"role": "system", "content": system_prompt}]
                # 逐张模式中音频只在第一张附带，避免重复发送。
                audio_arg = active_audio if replies == [] else None
                result = _infer(
                    frame_msgs, [pil_img], audio_arg,
                    image_index=image_index, image_total=total_frames,
                )
                replies.append(f"【图片 {image_index}/{total_frames}】\n{result}")
            reply = "\n\n".join(replies)
            messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt}]})
            messages.append({"role": "assistant", "content": reply})
        else:
            reply = _infer(messages, frame_list, active_audio)

        # ── 更新历史 ──────────────────────────────────────────────────────────
        if save_states:
            DapaoLlamaStorage.messages[uid] = messages
        else:
            DapaoLlamaStorage.clean_state(uid)

        # ── 推理后处理 ────────────────────────────────────────────────────────
        if force_offload:
            print("[大炮-llama] 卸载模型")
            DapaoLlamaStorage.clean()
            mm.soft_empty_cache()
        elif handler_name in ("Qwen3.8", "Qwen3.5", "Qwen3.5-Thinking", "Qwen3-VL", "Qwen3-VL-Thinking"):
            # 这些模型需要手动清空 KV cache，否则下次推理会出错
            try:
                llm.n_tokens = 0
                llm._ctx.memory_clear(True)
                if llm.is_hybrid and llm._hybrid_cache_mgr is not None:
                    llm._hybrid_cache_mgr.clear()
            except Exception:
                pass

        # ── 构建历史文本 ──────────────────────────────────────────────────────
        history_lines = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, list):
                text_parts = [p["text"] for p in content if p.get("type") == "text"]
                content = " ".join(text_parts)
            history_lines.append(f"[{role}]: {content}")
        history_text = "\n".join(history_lines)

        return (reply, history_text, seed)


# ── 节点注册 ─────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "Dapao_LlamaChat": Dapao_LlamaChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Dapao_LlamaChat": "😶‍🌫️llama智能对话@炮老师的小课堂",
}
