"""Local GGUF adapters for the prompt tools copied from ComfyUI-dapaoAPI."""

import contextvars
import copy

import folder_paths

from .nodes import (
    CHAT_HANDLERS,
    DapaoLlamaStorage,
    Dapao_LlamaChat,
    QWEN38_REASONING_OPTIONS,
    normalize_qwen38_reasoning_effort,
)


LOCAL_TOOL_CATEGORY = "🍭大炮-llama-cpp"
_ACTIVE_LOCAL_SETTINGS = contextvars.ContextVar("dapao_local_tool_settings")
_API_MODEL_PLACEHOLDER = "gpt-5.5"
_VRAM_TOOLTIP = (
    "这是LLM可使用的显存预算，不是预留空间。-1=尝试全部放入GPU，最快但可能因显存不足失败；"
    "填写数值=只将部分模型层放入GPU，其余使用系统内存。参考起点：8GB显卡填6，12GB填10，"
    "16GB填13，24GB填20，32GB填24-28。请为ComfyUI、mmproj和上下文缓存保留约2GB。"
)


def _local_model_inputs():
    llm_files = folder_paths.get_filename_list("LLM")
    mmproj_files = ["None"] + [name for name in llm_files if "mmproj" in name.lower()]
    model_files = [name for name in llm_files if "mmproj" not in name.lower()]
    return {
        "🤖本地模型文件": (model_files,),
        "🔌本地对话处理器": (CHAT_HANDLERS, {"default": "Qwen3.8"}),
        "🖼️本地mmproj文件": (mmproj_files, {"default": "None"}),
        "📐本地上下文长度": ("INT", {"default": 8192, "min": 512, "max": 131072, "step": 512}),
        "💾本地显存限制(GB)": ("FLOAT", {"default": -1, "min": -1, "max": 999.0, "step": 0.5, "tooltip": _VRAM_TOOLTIP}),
        "🔢图像最小token": ("INT", {"default": 256, "min": 1, "max": 4096, "step": 1}),
        "🔢图像最大token": ("INT", {"default": 1344, "min": 1, "max": 8192, "step": 1}),
        "🧠本地思考模式": ("BOOLEAN", {"default": False}),
        "🧠Qwen3.8推理强度": (QWEN38_REASONING_OPTIONS, {"default": "关闭"}),
    }


def _settings_from_kwargs(kwargs):
    return {
        "model_file": kwargs["🤖本地模型文件"],
        "handler_name": kwargs["🔌本地对话处理器"],
        "mmproj_file": kwargs["🖼️本地mmproj文件"],
        "n_ctx": int(kwargs["📐本地上下文长度"]),
        "vram_limit_gb": float(kwargs["💾本地显存限制(GB)"]),
        "image_min_tokens": int(kwargs["🔢图像最小token"]),
        "image_max_tokens": int(kwargs["🔢图像最大token"]),
        "think_mode": bool(kwargs["🧠本地思考模式"]),
        "reasoning_effort": normalize_qwen38_reasoning_effort(
            kwargs.get("🧠Qwen3.8推理强度", "关闭")
        ),
    }


def _payload_image_count(messages):
    count = 0
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        count += sum(
            1 for item in content
            if isinstance(item, dict) and item.get("type") == "image_url"
        )
    return count


def _ensure_local_model(settings, payload):
    max_tokens = int(payload.get("max_tokens", 1024))
    image_count = _payload_image_count(payload.get("messages", []))
    n_ctx = settings["n_ctx"]
    if image_count:
        required = settings["image_max_tokens"] * image_count + max_tokens + 512
        if required > 131072:
            raise ValueError(
                "当前请求的图片数量、图像最大 token 与输出 token 需要超过 131072 上下文。"
                "请降低图片数量、图像最大token或最大输出令牌。"
            )
        n_ctx = max(n_ctx, required)

    expected = {
        **settings,
        "n_ctx": n_ctx,
    }
    if DapaoLlamaStorage.llm is None or DapaoLlamaStorage.current_config != expected:
        DapaoLlamaStorage.clean()
        Dapao_LlamaChat()._load_model(
            expected["model_file"],
            expected["handler_name"],
            expected["mmproj_file"],
            expected["n_ctx"],
            expected["vram_limit_gb"],
            expected["image_min_tokens"],
            expected["image_max_tokens"],
            expected["think_mode"],
            expected["reasoning_effort"],
        )
    return DapaoLlamaStorage.llm


class LocalPayloadClient:
    """Drop-in replacement for the reference nodes' OpenAI-compatible clients."""

    def __init__(self, _api_key, _timeout):
        self.settings = _ACTIVE_LOCAL_SETTINGS.get(None)
        if self.settings is None:
            raise RuntimeError("本地模型配置未初始化。")

    def chat(self, payload):
        llm = _ensure_local_model(self.settings, payload)
        params = {
            "max_tokens": int(payload.get("max_tokens", 1024)),
            "temperature": float(payload.get("temperature", 0.7)),
            "top_p": float(payload.get("top_p", 0.9)),
        }
        try:
            return llm.create_chat_completion(messages=payload.get("messages", []), **params)
        finally:
            try:
                llm.n_tokens = 0
                llm._ctx.memory_clear(True)
                if llm.is_hybrid and llm._hybrid_cache_mgr is not None:
                    llm._hybrid_cache_mgr.clear()
            except Exception:
                pass


class LocalToolNodeMixin:
    """Replace API credentials with local llama-cpp model controls."""

    CATEGORY = LOCAL_TOOL_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        schema = copy.deepcopy(super().INPUT_TYPES())
        required = schema.get("required", {})
        required.pop("🔑 API密钥", None)
        required.pop("🤖 LLM模型", None)
        schema["required"] = {**_local_model_inputs(), **required}
        return schema

    async def generate_prompt(self, **kwargs):
        settings = _settings_from_kwargs(kwargs)
        forwarded = dict(kwargs)
        # The copied business nodes retain their validation and parsing logic.
        # Their API client class is replaced with LocalPayloadClient below.
        forwarded["🔑 API密钥"] = "local-llama-cpp"
        forwarded["🤖 LLM模型"] = _API_MODEL_PLACEHOLDER
        token = _ACTIVE_LOCAL_SETTINGS.set(settings)
        try:
            result = await super().generate_prompt(**forwarded)
        finally:
            _ACTIVE_LOCAL_SETTINGS.reset(token)
        return _replace_api_info(result, settings["model_file"])


def _replace_api_info(result, model_file):
    if not isinstance(result, tuple):
        return result
    updated = []
    for value in result:
        if not isinstance(value, str):
            updated.append(value)
            continue
        value = value.replace("🌐 中转站：https://api.dapaoai.com", "🖥️ 推理后端：本地 llama.cpp")
        value = value.replace("🌐 中转站：本地 llama.cpp", "🖥️ 推理后端：本地 llama.cpp")
        value = value.replace(f"🤖 LLM模型：{_API_MODEL_PLACEHOLDER}", f"🤖 本地模型：{model_file}")
        updated.append(value)
    return tuple(updated)
