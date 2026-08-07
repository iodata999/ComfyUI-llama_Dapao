"""Local GGUF-powered MiniMax H3 video prompt compiler."""

import json
import re
import time
import traceback

import numpy as np
from PIL import Image

import folder_paths
import comfy.model_management as mm

from .h3_prompt_rules import (
    ASPECT_RATIO_OPTIONS,
    MAX_H3_AUDIOS,
    MAX_H3_IMAGES,
    MAX_H3_MIXED_FILES,
    MAX_H3_VIDEOS,
    MAX_MEDIA_DURATION,
    MAX_MEDIA_TOTAL_DURATION,
    MIN_MEDIA_DURATION,
    MODE_OPTIONS,
    STYLE_GUIDES,
    STYLE_OPTIONS,
    SYSTEM_PROMPT,
    build_repair_prompt,
)
from .nodes import (
    CHAT_HANDLERS,
    DapaoLlamaStorage,
    Dapao_LlamaChat,
    audio2base64,
    image2base64,
    scale_image,
    tensor2pil,
)


NODE_NAME = "Dapao_LlamaH3Prompt"
DISPLAY_NAME = "🦊H3本地视频提示词生成@炮老师的小课堂"


def _log_info(message):
    print(f"[大炮-llama-H3] 信息：{message}")


def _log_error(message):
    print(f"[大炮-llama-H3] 错误：{message}")


def _strip_thinking(text, think_mode=False):
    text = (text or "").removeprefix(": ").lstrip()
    if think_mode:
        return text.strip()
    match = re.search(r"<think>.*?</think>(.*)", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    match = re.search(r"<\|channel>thought\n.*?<channel\|>(.*)", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    return text.strip()


def _extract_response_text(result, think_mode=False):
    if not isinstance(result, dict):
        return ""
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = message.get("content", first.get("text", ""))
    if isinstance(content, list):
        content = "\n".join(
            str(item.get("text") or item.get("output_text") or "")
            for item in content
            if isinstance(item, dict)
        )
    return _strip_thinking(str(content or ""), think_mode)


def _parse_compiler_output(text, fallback_mode):
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    parsed = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, character in enumerate(cleaned):
            if character != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(cleaned[index:])
                break
            except json.JSONDecodeError:
                continue

    if not isinstance(parsed, dict):
        marker_positions = [
            position
            for position in (
                cleaned.find("subject_definitions:"),
                cleaned.find("integrated_multimodal_description:"),
                cleaned.find("For the target video,"),
                cleaned.find("How the reference pictures align"),
            )
            if position >= 0
        ]
        if marker_positions:
            cleaned = cleaned[min(marker_positions):].strip()
        return {
            "mode": fallback_mode,
            "h3_prompt": cleaned,
            "material_analysis": "本地模型未采用JSON封装，节点已直接提取H3正文。",
            "production_notes": "建议查看结构校验结果；必要时提高本地模型参数规模或开启自动修复。",
        }

    return {
        "mode": str(parsed.get("mode") or fallback_mode).strip(),
        "h3_prompt": str(parsed.get("h3_prompt") or parsed.get("prompt") or "").strip(),
        "material_analysis": str(parsed.get("material_analysis") or "").strip(),
        "production_notes": str(parsed.get("production_notes") or "").strip(),
    }


def _audit_h3_prompt(prompt, mode, duration, image_count, video_count, audio_count, native_audio):
    issues = []
    text = str(prompt or "").strip()
    fields = (
        [
            "subject_definitions:",
            "summary:",
            "retention_analysis:",
            "detailed_description:",
            "overall_soundscape:",
            "non_diegetic_music:",
        ]
        if mode == "Ref2VA"
        else ["integrated_multimodal_description:", "overall_soundscape:", "non_diegetic_music:"]
    )
    positions = [text.find(field) for field in fields]
    missing = [field for field, position in zip(fields, positions) if position < 0]
    if missing:
        issues.append("缺少H3固定字段：" + "、".join(missing))
    elif positions != sorted(positions):
        issues.append("H3固定字段顺序不符合官方规范。")

    if mode == "I2VA":
        required = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
        if not text.startswith(required):
            issues.append("I2VA首帧对齐句未采用官方固定格式。")
    elif mode == "FL2VA":
        prefix = "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark"
        if not text.startswith(prefix):
            issues.append("FL2VA首尾帧对齐句未采用官方固定格式。")
        if f"{float(duration):.2f}-second mark" not in text[:600]:
            issues.append(f"FL2VA对齐句未把尾帧锁定在{float(duration):.2f}秒。")
    elif mode == "L2VA":
        prefix = "How the reference pictures align with the target video — <Picture 1> (from [Shot"
        if not text.startswith(prefix):
            issues.append("L2VA尾帧对齐句未采用官方固定格式。")
        if f"{float(duration):.2f}-second mark" not in text[:500]:
            issues.append(f"L2VA对齐句未把尾帧锁定在{float(duration):.2f}秒。")

    previous_time = -1.0
    for minute, second, millisecond in re.findall(r"\bAt\s+(\d{2}):(\d{2})\.(\d{3})", text):
        timestamp = int(minute) * 60 + int(second) + int(millisecond) / 1000.0
        if timestamp <= previous_time:
            issues.append(f"镜头切点{minute}:{second}.{millisecond}没有严格递增。")
        if timestamp >= float(duration):
            issues.append(f"镜头切点{minute}:{second}.{millisecond}超出或等于目标时长{float(duration):.2f}秒。")
        previous_time = timestamp

    if mode == "Ref2VA":
        expected = {"Picture": image_count, "Video": video_count, "Audio": audio_count}
        for label, count in expected.items():
            used = {int(value) for value in re.findall(rf"<{label}\s+(\d+)>", text)}
            out_of_range = sorted(number for number in used if number < 1 or number > count)
            if out_of_range:
                issues.append(f"出现素材清单中不存在的<{label} N>编号：{out_of_range}。")
            missing_numbers = [number for number in range(1, count + 1) if number not in used]
            if missing_numbers:
                issues.append(f"未在Ref2VA提示词中引用{label}素材：{missing_numbers}。")
    elif re.search(r"<(Video|Audio)\s+\d+>", text):
        issues.append(f"{mode}基础模式不应出现<Video N>或<Audio N>引用。")

    if not native_audio and fields[-2] in text and fields[-1] in text:
        sound_block = text[text.find(fields[-2]):]
        if not re.search(r"overall_soundscape:\s*N/A\b", sound_block, re.IGNORECASE):
            issues.append("关闭原生音频后overall_soundscape必须为N/A。")
        if not re.search(r"non_diegetic_music:\s*N/A\b", sound_block, re.IGNORECASE):
            issues.append("关闭原生音频后non_diegetic_music必须为N/A。")
    return list(dict.fromkeys(issues))


def _validate_image_tensor(tensor, label):
    if tensor is None:
        return
    if not hasattr(tensor, "shape") or len(tensor.shape) != 4 or tensor.shape[-1] not in (1, 3, 4):
        raise ValueError(f"{label}必须是[B,H,W,C]格式的IMAGE张量。")


def _tensor_to_pil_images(tensor, label, max_side):
    _validate_image_tensor(tensor, label)
    images = []
    for index in range(tensor.shape[0]):
        image = tensor2pil(tensor[index:index + 1]).convert("RGB")
        images.append(scale_image(image, int(max_side)))
    return images


def _sample_video_tensor(tensor, source_index, slot, duration, sample_count, max_side):
    label = f"参考视频{slot}"
    _validate_image_tensor(tensor, label)
    frame_count = int(tensor.shape[0])
    if frame_count < 2:
        raise ValueError(f"{label}至少需要2帧IMAGE批次，当前只有{frame_count}帧。")
    duration = float(duration)
    if duration < MIN_MEDIA_DURATION or duration > MAX_MEDIA_DURATION:
        raise ValueError(
            f"{label}时长为{duration:.2f}秒；H3要求每个视频为"
            f"{MIN_MEDIA_DURATION:.0f}–{MAX_MEDIA_DURATION:.0f}秒。"
        )
    count = min(frame_count, max(2, int(sample_count)))
    indices = sorted(set(np.linspace(0, frame_count - 1, count).round().astype(int).tolist()))
    final_time = max(0.0, duration - duration / max(frame_count, 2))
    frames = []
    for frame_index in indices:
        image = tensor2pil(tensor[frame_index:frame_index + 1]).convert("RGB")
        frames.append({
            "frame_index": frame_index,
            "time": final_time * frame_index / max(frame_count - 1, 1),
            "image": scale_image(image, int(max_side)),
        })
    return {
        "index": source_index,
        "slot": slot,
        "duration": duration,
        "source_frame_count": frame_count,
        "width": int(tensor.shape[2]),
        "height": int(tensor.shape[1]),
        "frames": frames,
    }


def _normalize_audio(audio_input, source_index, slot, include_raw_audio):
    if not isinstance(audio_input, dict):
        raise ValueError(f"参考音频{slot}必须连接ComfyUI AUDIO输出。")
    waveform = audio_input.get("waveform")
    sample_rate = int(audio_input.get("sample_rate") or audio_input.get("sampler_rate") or 0)
    if waveform is None or sample_rate <= 0:
        raise ValueError(f"参考音频{slot}缺少有效waveform或sample_rate。")
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().numpy()
    channels = np.asarray(waveform)
    if channels.ndim == 3:
        if channels.shape[0] != 1:
            raise ValueError(f"参考音频{slot}包含多个批次，请先拆分为单条音频。")
        channels = channels[0]
    channels = np.squeeze(channels)
    if channels.ndim == 1:
        channels = channels.reshape(1, -1)
    elif channels.ndim == 2 and channels.shape[0] > 8 and channels.shape[1] <= 8:
        channels = channels.T
    if channels.ndim != 2 or channels.shape[0] > 8:
        raise ValueError(f"参考音频{slot}声道格式无法识别。")
    if np.issubdtype(channels.dtype, np.integer):
        limit = max(abs(np.iinfo(channels.dtype).min), np.iinfo(channels.dtype).max)
        channels = channels.astype(np.float32) / float(limit)
    else:
        channels = channels.astype(np.float32)
    channels = np.nan_to_num(np.clip(channels, -1.0, 1.0))
    duration = channels.shape[1] / float(sample_rate)
    if duration < MIN_MEDIA_DURATION - 0.05 or duration > MAX_MEDIA_DURATION + 0.05:
        raise ValueError(
            f"参考音频{slot}时长为{duration:.2f}秒；H3要求每个音频为"
            f"{MIN_MEDIA_DURATION:.0f}–{MAX_MEDIA_DURATION:.0f}秒。"
        )
    mono = channels.mean(axis=0)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    threshold = max(0.006, peak * 0.025)
    result = {
        "index": source_index,
        "slot": slot,
        "duration": duration,
        "sample_rate": sample_rate,
        "channels": int(channels.shape[0]),
        "rms": float(np.sqrt(np.mean(np.square(mono), dtype=np.float64))),
        "peak": peak,
        "silence_ratio": float(np.mean(np.abs(mono) < threshold)),
        "spectrogram": _audio_spectrogram(mono),
    }
    if include_raw_audio:
        result["raw_wav_base64"] = audio2base64(audio_input)
    return result


def _audio_spectrogram(mono):
    frame_size = 2048
    hop = 512
    if mono.size < frame_size:
        mono = np.pad(mono, (0, frame_size - mono.size))
    windows = []
    for start in range(0, mono.size - frame_size + 1, hop):
        window = mono[start:start + frame_size] * np.hanning(frame_size)
        windows.append(np.abs(np.fft.rfft(window))[:256])
    spectrum = np.stack(windows or [np.zeros(256)], axis=1)
    db = 20.0 * np.log10(np.maximum(spectrum, 1e-6))
    db -= np.max(db)
    normalized = np.flipud(np.clip((db + 80.0) / 80.0, 0.0, 1.0))
    red = np.clip(normalized * 1.8, 0.0, 1.0)
    green = np.clip((normalized - 0.15) * 1.35, 0.0, 1.0)
    blue = np.clip(0.18 + normalized * 0.82, 0.0, 1.0)
    rgb = (np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)
    return Image.fromarray(rgb).resize((1024, 512), Image.Resampling.BICUBIC)


def _image_part(image):
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image2base64(image)}"},
    }


def _model_config(kwargs):
    return {
        "model_file": kwargs["🤖模型文件"],
        "handler_name": kwargs["🔌对话处理器"],
        "mmproj_file": kwargs["🖼️mmproj文件"],
        "n_ctx": int(kwargs["📐上下文长度"]),
        "vram_limit_gb": float(kwargs["💾显存限制(GB)"]),
        "image_min_tokens": int(kwargs["🔢图像最小token"]),
        "image_max_tokens": int(kwargs["🔢图像最大token"]),
        "think_mode": bool(kwargs["🧠思考模式"]),
    }


def _ensure_model(kwargs):
    config = _model_config(kwargs)
    if DapaoLlamaStorage.llm is not None and DapaoLlamaStorage.current_config == config:
        _log_info("复用已加载的本地GGUF模型")
        return DapaoLlamaStorage.llm
    DapaoLlamaStorage.clean()
    mm.soft_empty_cache()
    loader = Dapao_LlamaChat()
    return loader._load_model(
        config["model_file"],
        config["handler_name"],
        config["mmproj_file"],
        config["n_ctx"],
        config["vram_limit_gb"],
        config["image_min_tokens"],
        config["image_max_tokens"],
        config["think_mode"],
    )


def _clear_runtime_cache(llm, handler_name):
    if handler_name not in ("Qwen3.5", "Qwen3.5-Thinking", "Qwen3-VL", "Qwen3-VL-Thinking"):
        return
    try:
        llm.n_tokens = 0
        llm._ctx.memory_clear(True)
        if llm.is_hybrid and llm._hybrid_cache_mgr is not None:
            llm._hybrid_cache_mgr.clear()
    except Exception:
        pass


class Dapao_LlamaH3Prompt:
    CATEGORY = "🍭大炮-llama-cpp"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("🎬H3最终提示词", "🎛️识别模式", "📑素材与制作分析", "📄LLM完整响应", "ℹ️处理信息")
    FUNCTION = "generate_prompt"
    DESCRIPTION = "本地GGUF模型执行MiniMax H3提示词规范，支持T2VA/I2VA/FL2VA/L2VA以及9图+3视频+3音频Ref2VA。"

    @classmethod
    def INPUT_TYPES(cls):
        llm_files = folder_paths.get_filename_list("LLM")
        mmproj_files = ["None"] + [name for name in llm_files if "mmproj" in name.lower()]
        model_files = [name for name in llm_files if "mmproj" not in name.lower()]
        optional = {
            "🎬首帧图": ("IMAGE", {"tooltip": "I2VA/FL2VA的精确首帧锚点。"}),
            "🏁尾帧图": ("IMAGE", {"tooltip": "L2VA/FL2VA的精确尾帧锚点。"}),
            "📚参考素材角色说明": ("STRING", {"multiline": True, "default": "", "placeholder": "例如：参考图1锁定人物外貌；视频1只参考运镜；音频1参考女主音色。"}),
            "🎞️参考视频说明": ("STRING", {"multiline": True, "default": "", "placeholder": "按视频1/2/3说明动作、运镜、剪辑、编辑或续写关系。"}),
            "🎵参考音频说明": ("STRING", {"multiline": True, "default": "", "placeholder": "按音频1/2/3说明复制或参考关系；精确歌词、对白和音色身份请明确写出。"}),
            "🗣️对白歌词与画面文字": ("STRING", {"multiline": True, "default": "", "placeholder": "此处原文将保持原语言和标点，不会擅自改写。"}),
            "🔊音效与配乐要求": ("STRING", {"multiline": True, "default": "", "placeholder": "环境音、动作音、配器、速度、节奏和动态变化。"}),
        }
        for index in range(1, MAX_H3_IMAGES + 1):
            optional[f"🖼️参考图{index}"] = ("IMAGE", {"tooltip": f"Ref2VA源图片接口{index}；图片总数最多9张。"})
        for index in range(1, MAX_H3_VIDEOS + 1):
            optional[f"🎞️参考视频{index}"] = ("IMAGE", {"tooltip": f"视频{index}请连接按时间排序的IMAGE帧批次；节点会均匀抽帧分析。"})
            optional[f"🎵参考音频{index}"] = ("AUDIO", {"tooltip": f"Ref2VA源音频接口{index}；每条2–15秒。"})
        return {
            "required": {
                "🤖模型文件": (model_files,),
                "🔌对话处理器": (CHAT_HANDLERS, {"default": "Qwen3.5"}),
                "🖼️mmproj文件": (mmproj_files, {"default": "None"}),
                "📐上下文长度": ("INT", {"default": 32768, "min": 4096, "max": 131072, "step": 1024}),
                "💾显存限制(GB)": ("FLOAT", {"default": -1, "min": -1, "max": 999.0, "step": 0.5}),
                "🔢图像最小token": ("INT", {"default": 256, "min": 1, "max": 4096, "step": 1}),
                "🔢图像最大token": ("INT", {"default": 768, "min": 1, "max": 8192, "step": 1}),
                "🎛️H3生成模式": (MODE_OPTIONS, {"default": "自动识别"}),
                "🎨创作类型": (STYLE_OPTIONS, {"default": "通用H3"}),
                "📝原始视频需求": ("STRING", {"multiline": True, "default": "电影感镜头，主体动作自然，音画同步，画面稳定且细节丰富。", "placeholder": "描述想生成的视频、剧情、动作、镜头和声音。"}),
                "⏱️目标时长(秒)": ("INT", {"default": 5, "min": 4, "max": 15, "step": 1}),
                "📐视频比例": (ASPECT_RATIO_OPTIONS, {"default": "16:9"}),
                "🔊原生音频": ("BOOLEAN", {"default": True}),
                "🎞️视频1时长(秒)": ("FLOAT", {"default": 5.0, "min": 2.0, "max": 15.0, "step": 0.1, "tooltip": "IMAGE批次不携带时长，请填写视频1的实际时长。"}),
                "🎞️视频2时长(秒)": ("FLOAT", {"default": 5.0, "min": 2.0, "max": 15.0, "step": 0.1, "tooltip": "仅连接视频2时生效。"}),
                "🎞️视频3时长(秒)": ("FLOAT", {"default": 5.0, "min": 2.0, "max": 15.0, "step": 0.1, "tooltip": "仅连接视频3时生效。"}),
                "🎞️每视频采样帧": ("INT", {"default": 4, "min": 2, "max": 8, "step": 1}),
                "📏图像最大边长": ("INT", {"default": 1024, "min": 128, "max": 2048, "step": 32}),
                "🎧原始音频直传模型": ("BOOLEAN", {"default": False, "tooltip": "默认使用频谱分析；只有确认本地模型支持input_audio时才开启。"}),
                "📝最大输出token": ("INT", {"default": 4096, "min": 512, "max": 32768, "step": 1}),
                "🌡️温度": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01}),
                "🎯top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "🔝top_k": ("INT", {"default": 40, "min": 0, "max": 200, "step": 1}),
                "🔁重复惩罚": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 2.0, "step": 0.01}),
                "🎲随机种子": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "🧠思考模式": ("BOOLEAN", {"default": False}),
                "🛠️自动修复结构": ("BOOLEAN", {"default": True, "tooltip": "结构审计不通过时执行一次纯文本修复，不重复分析素材。"}),
                "⚡推理后卸载模型": ("BOOLEAN", {"default": False}),
                "🚫出错时跳过": ("BOOLEAN", {"default": False}),
            },
            "optional": optional,
        }

    @staticmethod
    def _collect_images(kwargs, max_side):
        ordered = []
        for key, name, single in (("🎬首帧图", "首帧图", True), ("🏁尾帧图", "尾帧图", True)):
            tensor = kwargs.get(key)
            if tensor is None:
                continue
            if single and int(tensor.shape[0]) != 1:
                raise ValueError(f"{name}只能包含1张图片，当前批次为{int(tensor.shape[0])}张。")
            for image in _tensor_to_pil_images(tensor, name, max_side):
                ordered.append({"name": name, "image": image})
        for slot in range(1, MAX_H3_IMAGES + 1):
            tensor = kwargs.get(f"🖼️参考图{slot}")
            if tensor is None:
                continue
            images = _tensor_to_pil_images(tensor, f"参考图{slot}", max_side)
            for batch_index, image in enumerate(images, start=1):
                suffix = f"-{batch_index}" if len(images) > 1 else ""
                ordered.append({"name": f"参考图{slot}{suffix}", "image": image})
        if len(ordered) > MAX_H3_IMAGES:
            raise ValueError(f"H3最多支持{MAX_H3_IMAGES}张源图片，当前检测到{len(ordered)}张。")
        return ordered

    @staticmethod
    def _collect_video_audio(kwargs, max_side):
        sample_count = int(kwargs["🎞️每视频采样帧"])
        include_raw = bool(kwargs["🎧原始音频直传模型"])
        videos = []
        audios = []
        for slot in range(1, MAX_H3_VIDEOS + 1):
            tensor = kwargs.get(f"🎞️参考视频{slot}")
            if tensor is None:
                continue
            duration = kwargs[f"🎞️视频{slot}时长(秒)"]
            videos.append(_sample_video_tensor(tensor, len(videos) + 1, slot, duration, sample_count, max_side))
        for slot in range(1, MAX_H3_AUDIOS + 1):
            audio = kwargs.get(f"🎵参考音频{slot}")
            if audio is None:
                continue
            audios.append(_normalize_audio(audio, len(audios) + 1, slot, include_raw))
        return videos, audios

    @staticmethod
    def _validate_source_limits(images, videos, audios):
        file_count = len(images) + len(videos) + len(audios)
        if file_count > MAX_H3_MIXED_FILES:
            raise ValueError(
                f"Ref2VA图片、视频和音频合计最多{MAX_H3_MIXED_FILES}个文件，当前为{file_count}个"
                f"（图片{len(images)}＋视频{len(videos)}＋音频{len(audios)}）。"
            )
        video_duration = sum(item["duration"] for item in videos)
        audio_duration = sum(item["duration"] for item in audios)
        if video_duration > MAX_MEDIA_TOTAL_DURATION + 0.05:
            raise ValueError(f"参考视频总时长为{video_duration:.2f}秒，H3要求不超过{MAX_MEDIA_TOTAL_DURATION:.0f}秒。")
        if audio_duration > MAX_MEDIA_TOTAL_DURATION + 0.05:
            raise ValueError(f"参考音频总时长为{audio_duration:.2f}秒，H3要求不超过{MAX_MEDIA_TOTAL_DURATION:.0f}秒。")
        if audios and not (images or videos):
            raise ValueError("H3参考音频不能作为唯一素材，必须同时接入至少一张图片或一个视频。")

    @staticmethod
    def _resolve_mode(selected, has_first, has_last, has_references, has_video_or_audio):
        manual = {
            "T2VA-文生视频": "T2VA",
            "I2VA-首帧生视频": "I2VA",
            "FL2VA-首尾帧生视频": "FL2VA",
            "L2VA-尾帧生视频": "L2VA",
            "Ref2VA-全能参考": "Ref2VA",
        }
        if selected != "自动识别":
            return manual[selected]
        if has_references or has_video_or_audio:
            return "Ref2VA"
        if has_first and has_last:
            return "FL2VA"
        if has_first:
            return "I2VA"
        if has_last:
            return "L2VA"
        return "T2VA"

    @staticmethod
    def _validate_mode(mode, has_first, has_last, has_references, has_video_or_audio):
        any_media = has_first or has_last or has_references or has_video_or_audio
        if mode == "T2VA" and any_media:
            raise ValueError("T2VA不使用参考素材；请选择自动识别或对应素材模式。")
        if mode == "I2VA" and (not has_first or has_last or has_references or has_video_or_audio):
            raise ValueError("I2VA只允许接入一张首帧图。")
        if mode == "FL2VA" and (not has_first or not has_last or has_references or has_video_or_audio):
            raise ValueError("FL2VA必须且只能接入首帧图和尾帧图。")
        if mode == "L2VA" and (not has_last or has_first or has_references or has_video_or_audio):
            raise ValueError("L2VA只允许接入一张尾帧图。")
        if mode == "Ref2VA" and not any_media:
            raise ValueError("Ref2VA至少需要一张图片或一个参考视频；音频不能作为唯一素材。")

    @staticmethod
    def _build_user_content(kwargs, mode, style, images, videos, audios):
        image_labels = [f"<Picture {index}>={item['name']}" for index, item in enumerate(images, start=1)]
        video_labels = [
            f"<Video {item['index']}>=视频接口{item['slot']}，{item['duration']:.2f}秒，"
            f"{item['source_frame_count']}帧IMAGE批次，{item['width']}x{item['height']}"
            for item in videos
        ]
        audio_labels = [
            f"<Audio {item['index']}>=音频接口{item['slot']}，{item['duration']:.2f}秒，"
            f"{item['sample_rate']}Hz/{item['channels']}声道，RMS={item['rms']:.4f}，"
            f"静音比例={item['silence_ratio']:.1%}"
            for item in audios
        ]
        duration = int(kwargs["⏱️目标时长(秒)"])
        native_audio = bool(kwargs["🔊原生音频"])
        text = (
            f"Requested H3 mode: {mode}\n"
            f"Creative preset: {style}\n"
            f"Preset direction: {STYLE_GUIDES[style]}\n"
            f"Exact duration: {duration}.00 seconds\n"
            f"Aspect ratio: {kwargs['📐视频比例']}\n"
            f"Native audio enabled: {str(native_audio).lower()}\n"
            "SOURCE MEDIA MANIFEST (source files only; sampled frames and spectrograms are analysis artifacts):\n"
            f"Images ({len(image_labels)}/{MAX_H3_IMAGES}): {', '.join(image_labels) if image_labels else 'none'}\n"
            f"Videos ({len(video_labels)}/{MAX_H3_VIDEOS}): {'; '.join(video_labels) if video_labels else 'none'}\n"
            f"Audio ({len(audio_labels)}/{MAX_H3_AUDIOS}): {'; '.join(audio_labels) if audio_labels else 'none'}\n"
            f"Mixed source-file count: {len(images) + len(videos) + len(audios)}/{MAX_H3_MIXED_FILES}\n\n"
            f"原始视频需求：\n{kwargs['📝原始视频需求'].strip()}\n\n"
            f"参考素材角色说明：\n{(kwargs.get('📚参考素材角色说明') or '').strip() or '未提供，请根据需求和可见素材谨慎判断。'}\n\n"
            f"参考视频说明：\n{(kwargs.get('🎞️参考视频说明') or '').strip() or '无'}\n\n"
            f"参考音频说明：\n{(kwargs.get('🎵参考音频说明') or '').strip() or '无'}\n\n"
            f"必须保持原文的对白、歌词与画面文字：\n{(kwargs.get('🗣️对白歌词与画面文字') or '').strip() or '无，不得自行编造。'}\n\n"
            f"音效与配乐要求：\n{(kwargs.get('🔊音效与配乐要求') or '').strip() or ('按画面设计克制且同步的原生声音。' if native_audio else '完全无声。')}"
        )
        if not (images or videos or audios):
            return text
        content = [{"type": "text", "text": text}]
        for index, item in enumerate(images, start=1):
            content.append({"type": "text", "text": f"Source <Picture {index}> ({item['name']}) follows."})
            content.append(_image_part(item["image"]))
        for item in videos:
            for frame in item["frames"]:
                content.append({
                    "type": "text",
                    "text": (
                        f"Analysis artifact for <Video {item['index']}>: source frame {frame['frame_index'] + 1}/"
                        f"{item['source_frame_count']} at approximately {frame['time']:.3f}s. "
                        "This is not a <Picture N> source."
                    ),
                })
                content.append(_image_part(frame["image"]))
        for item in audios:
            content.append({
                "type": "text",
                "text": (
                    f"Analysis artifact for <Audio {item['index']}>: time-frequency spectrogram. "
                    "Use only for timing, rhythm, energy, silence, and dynamics; it is not a source picture."
                ),
            })
            content.append(_image_part(item["spectrogram"]))
            if item.get("raw_wav_base64"):
                content.append({
                    "type": "input_audio",
                    "input_audio": {"data": item["raw_wav_base64"], "format": "wav"},
                })
        return content

    def generate_prompt(self, **kwargs):
        skip_error = bool(kwargs["🚫出错时跳过"])
        resolved_mode = ""
        raw_responses = {}
        llm = None
        started = time.time()
        try:
            selected_mode = kwargs["🎛️H3生成模式"]
            style = kwargs["🎨创作类型"]
            if selected_mode not in MODE_OPTIONS or style not in STYLE_OPTIONS:
                raise ValueError("H3模式或创作类型无效。")
            if not kwargs["📝原始视频需求"].strip():
                raise ValueError("原始视频需求不能为空。")

            max_side = int(kwargs["📏图像最大边长"])
            images = self._collect_images(kwargs, max_side)
            videos, audios = self._collect_video_audio(kwargs, max_side)
            self._validate_source_limits(images, videos, audios)

            has_first = kwargs.get("🎬首帧图") is not None
            has_last = kwargs.get("🏁尾帧图") is not None
            has_references = any(kwargs.get(f"🖼️参考图{index}") is not None for index in range(1, MAX_H3_IMAGES + 1))
            has_video_or_audio = bool(videos or audios)
            resolved_mode = self._resolve_mode(selected_mode, has_first, has_last, has_references, has_video_or_audio)
            self._validate_mode(resolved_mode, has_first, has_last, has_references, has_video_or_audio)

            has_visual_input = bool(images or videos or audios)
            if has_visual_input and (kwargs["🔌对话处理器"] == "None" or kwargs["🖼️mmproj文件"] == "None"):
                raise ValueError("当前任务包含视觉素材或音频频谱，请选择支持多模态的对话处理器和匹配的mmproj文件。")

            llm = _ensure_model(kwargs)
            params = {
                "max_tokens": int(kwargs["📝最大输出token"]),
                "temperature": float(kwargs["🌡️温度"]),
                "top_p": float(kwargs["🎯top_p"]),
                "top_k": int(kwargs["🔝top_k"]),
                "repeat_penalty": float(kwargs["🔁重复惩罚"]),
                "seed": int(kwargs["🎲随机种子"]),
            }
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_content(kwargs, resolved_mode, style, images, videos, audios)},
            ]
            _log_info(
                f"开始本地编译：模式={resolved_mode}，模型={kwargs['🤖模型文件']}，"
                f"图片={len(images)}，视频={len(videos)}，音频={len(audios)}"
            )
            mm.throw_exception_if_processing_interrupted()
            initial_result = llm.create_chat_completion(messages=messages, **params)
            initial_text = _extract_response_text(initial_result, bool(kwargs["🧠思考模式"]))
            raw_responses["initial"] = initial_text
            if not initial_text:
                raise RuntimeError("本地模型返回内容为空。")
            compiled = _parse_compiler_output(initial_text, resolved_mode)
            h3_prompt = compiled["h3_prompt"]
            if not h3_prompt:
                raise RuntimeError("本地模型没有返回H3提示词正文。")

            native_audio = bool(kwargs["🔊原生音频"])
            audit_issues = _audit_h3_prompt(
                h3_prompt,
                resolved_mode,
                kwargs["⏱️目标时长(秒)"],
                len(images),
                len(videos),
                len(audios),
                native_audio,
            )
            repaired = False
            if audit_issues and bool(kwargs["🛠️自动修复结构"]):
                _log_info(f"结构审计发现{len(audit_issues)}项问题，执行一次纯文本修复")
                repair_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_repair_prompt(resolved_mode, kwargs["⏱️目标时长(秒)"], audit_issues, h3_prompt)},
                ]
                repair_params = dict(params)
                repair_params["temperature"] = min(float(params["temperature"]), 0.2)
                mm.throw_exception_if_processing_interrupted()
                repair_result = llm.create_chat_completion(messages=repair_messages, **repair_params)
                repair_text = _extract_response_text(repair_result, bool(kwargs["🧠思考模式"]))
                raw_responses["repair"] = repair_text
                repair_compiled = _parse_compiler_output(repair_text, resolved_mode)
                repair_prompt = repair_compiled["h3_prompt"]
                repair_issues = _audit_h3_prompt(
                    repair_prompt,
                    resolved_mode,
                    kwargs["⏱️目标时长(秒)"],
                    len(images),
                    len(videos),
                    len(audios),
                    native_audio,
                ) if repair_prompt else audit_issues
                if repair_prompt and len(repair_issues) < len(audit_issues):
                    h3_prompt = repair_prompt
                    audit_issues = repair_issues
                    compiled["material_analysis"] = repair_compiled["material_analysis"] or compiled["material_analysis"]
                    compiled["production_notes"] = repair_compiled["production_notes"] or compiled["production_notes"]
                    repaired = True

            analysis_parts = [part for part in (compiled["material_analysis"], compiled["production_notes"]) if part]
            if compiled["mode"] and compiled["mode"] != resolved_mode:
                analysis_parts.append(f"模式锁定：模型返回{compiled['mode']}，节点已按实际素材锁定为{resolved_mode}。")
            if audit_issues:
                analysis_parts.append("结构校验提醒：\n- " + "\n- ".join(audit_issues))
            else:
                analysis_parts.append("结构校验：全部通过。")
            analysis = "\n\n".join(analysis_parts)

            info = (
                "✅ H3本地视频提示词生成完成\n"
                f"🤖 本地模型：{kwargs['🤖模型文件']}\n"
                f"🔌 对话处理器：{kwargs['🔌对话处理器']}\n"
                f"🎛️ H3模式：{resolved_mode}\n"
                f"🎨 创作类型：{style}\n"
                f"⏱️ 目标时长：{int(kwargs['⏱️目标时长(秒)'])}秒\n"
                f"📐 比例：{kwargs['📐视频比例']}\n"
                f"🖼️ 图片：{len(images)}张\n"
                f"🎞️ 视频：{len(videos)}个 / 总计{sum(item['duration'] for item in videos):.2f}秒\n"
                f"🎵 音频：{len(audios)}个 / 总计{sum(item['duration'] for item in audios):.2f}秒\n"
                f"📦 混合素材：{len(images) + len(videos) + len(audios)}/{MAX_H3_MIXED_FILES}个\n"
                f"🛠️ 自动修复：{'已采用修复结果' if repaired else ('无需修复' if not audit_issues else '未改善，保留原结果')}\n"
                f"🔎 结构校验：{'通过' if not audit_issues else f'仍有{len(audit_issues)}项提醒'}\n"
                f"⏱️ 耗时：{time.time() - started:.2f}秒"
            )
            return h3_prompt, resolved_mode, analysis, json.dumps(raw_responses, ensure_ascii=False, indent=2), info
        except Exception as error:
            message = f"❌ H3本地视频提示词生成失败：{error}"
            _log_error(message)
            _log_error(traceback.format_exc())
            if skip_error:
                return message, resolved_mode or "未知", message, json.dumps(raw_responses, ensure_ascii=False, indent=2), message
            raise RuntimeError(message) from error
        finally:
            if llm is not None:
                if bool(kwargs.get("⚡推理后卸载模型", False)):
                    DapaoLlamaStorage.clean()
                    mm.soft_empty_cache()
                else:
                    _clear_runtime_cache(llm, kwargs.get("🔌对话处理器", "None"))


NODE_CLASS_MAPPINGS = {NODE_NAME: Dapao_LlamaH3Prompt}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_NAME: DISPLAY_NAME}


__all__ = [
    "Dapao_LlamaH3Prompt",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
