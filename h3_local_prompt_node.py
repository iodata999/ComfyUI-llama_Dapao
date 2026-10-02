"""本地 H3 提示词；提示与素材处理改编自 chflame163，许可见 h3_local/LICENSE。"""
from __future__ import annotations

import re

from comfy_api.latest import io

from .h3_local.media import image_content, image_grid_content, sample_indices_per_second, text_content
from .h3_local.skills import (
    SKILL_NAMES, detect_h3_mode, mode_router_prompt, output_issues,
    parse_mode_selection, parse_skill_selection, router_prompt, system_prompt,
)
from .multi_turn_chat_node import (
    _sync_model, _model_settings, _release_model, _should_unload,
    _create_completion, _clean_reply, _reset_llm,
)
from .nodes import _create_qwen38_text_handler

REFERENCE_VIDEO_FPS = 24.0

def _autogrow_values(inputs) -> list:
    if not inputs:
        return []

    def index(item):
        match = re.search(r"(\d+)$", item[0])
        return int(match.group(1)) if match else 0

    return [value for _, value in sorted(inputs.items(), key=index) if value is not None]


def _validate_reference_images(images: list) -> None:
    for image_index, image in enumerate(images, 1):
        image_count = int(image.shape[0])
        if image_count != 1:
            raise ValueError(
                f"Reference image {image_index} must contain exactly one image; got a batch of {image_count}."
            )


def _reference_video_details(
    videos: list,
    sample_frames_per_second: int,
) -> list[tuple]:
    details = []
    for video_index, frames in enumerate(videos, 1):
        frame_count = int(frames.shape[0])
        if frame_count < 1:
            raise ValueError(
                f"Reference video {video_index} must contain at least 1 frame; got {frame_count}."
            )
        source_duration = frame_count / REFERENCE_VIDEO_FPS
        details.append(
            (
                frames,
                frame_count,
                source_duration,
                sample_indices_per_second(
                    frame_count,
                    REFERENCE_VIDEO_FPS,
                    sample_frames_per_second,
                ),
            )
        )
    return details


def _picture_role(mode: str, picture_index: int) -> str:
    if mode == "i2va":
        return "the target video's first frame"
    if mode == "l2va":
        return "the target video's last frame"
    if mode == "fl2va":
        return "the target video's first frame" if picture_index == 1 else "the target video's last frame"
    return "a general visual reference whose role follows the user request"


def _sampling_settings(think_mode: bool) -> dict[str, float | int]:
    if think_mode:
        return {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repeat_penalty": 1.0,
        }
    return {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repeat_penalty": 1.0,
    }


def _user_content(
    prompt: str,
    duration: float,
    images: list,
    video_details: list[tuple],
    mode: str,
) -> tuple[list[dict[str, object]], str]:
    content: list[dict[str, object]] = [
        text_content(f"User request:\n{prompt}\n\nTarget duration: {duration:.2f} seconds."),
    ]
    assets = []
    for picture_index, image in enumerate(images, 1):
        role = _picture_role(mode, picture_index)
        content.append(text_content(f"<Picture {picture_index}>: connected as {role}."))
        content.append(image_content(image))
        assets.append(f"Picture {picture_index}={role}")

    for video_index, details in enumerate(video_details, 1):
        frames, frame_count, source_duration, sample_groups = details
        sampled_frame_count = sum(len(indices) for indices in sample_groups)
        content.append(
            text_content(
                f"<Video {video_index}>: {frame_count} ordered frames at {REFERENCE_VIDEO_FPS:.3f} fps "
                f"({source_duration:.2f} seconds), represented by {sampled_frame_count} sampled frames "
                f"grouped into {len(sample_groups)} chronological one-second contact sheets."
            )
        )
        for second_index, indices in enumerate(sample_groups):
            timestamps = ", ".join(
                f"{frame_index / REFERENCE_VIDEO_FPS:.3f}s" for frame_index in indices
            )
            content.append(
                text_content(
                    f"<Video {video_index}> second {second_index + 1}/{len(sample_groups)} contact sheet. "
                    f"Read cells in row-major chronological order at: {timestamps}."
                )
            )
            content.append(image_grid_content(frames, indices))
        assets.append(
            f"Video {video_index}={source_duration:.2f}s/{sampled_frame_count} sampled frames"
        )
    return content, ", ".join(assets) if assets else "none"



def _reply(response):
    content = response["choices"][0]["message"].get("content") or ""
    if isinstance(content, list):
        content = "\n".join(part.get("text", "") for part in content if part.get("type") == "text")
    if "<think>" in content and "</think>" not in content:
        return ""
    if "</think>" in content:
        content = content.split("</think>", 1)[1]
    return _clean_reply(str(content)).strip()


def _route(llm, messages, seed, max_tokens):
    # A separate text formatter disables thinking for short routing calls without reloading GGUF.
    handler = _create_qwen38_text_handler(
        llm, enable_thinking=False, preserve_thinking=False, reasoning_effort="off",
    )
    _reset_llm(llm)
    return _reply(handler(
        llama=llm, messages=messages, seed=seed, max_tokens=max_tokens,
        temperature=0.0, top_p=1.0, top_k=1, min_p=0.0,
        repeat_penalty=1.0, stream=False,
    ))


class DapaoLocalH3Prompt(io.ComfyNode):
    # V3 supplies INPUT_TYPES/RETURN_TYPES/RETURN_NAMES/FUNCTION/CATEGORY from this schema.
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DapaoLocalH3Prompt",
            display_name="本地H3提示词@炮老师的小课堂",
            category="🍭大炮-llama-cpp",
            description="复用本地模型加载器与便携Python；支持H3模式识别、Skill、多图与视频帧。视频帧按24fps解释，不分析音轨。",
            inputs=[
                io.Custom("DAPAO_LOCAL_MODEL").Input("model", display_name="🤖本地模型"),
                io.String.Input("prompt", display_name="✏️需求描述", multiline=True, default=""),
                io.Combo.Input("skill", display_name="🧩Skill", options=["auto", *SKILL_NAMES], default="auto"),
                io.Float.Input("duration", display_name="⏱️目标秒数", default=10.0, min=1, max=60, step=0.5),
                io.Int.Input("seed", display_name="🎲随机种子", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                io.Int.Input("max_tokens", display_name="📝最大输出tokens", default=8192, min=256, max=32768, step=128),
                io.Int.Input("video_sample_frames_per_sec", display_name="🎞️视频每秒采样帧数", default=2, min=1, max=8),
                io.Boolean.Input("force_unload_model", display_name="🧹完成后卸载模型", default=True),
                io.Autogrow.Input("reference_images", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Image.Input("reference_image", display_name="🖼️参考图", tooltip="每个接口一张图；首帧或尾帧用途写在需求里。"),
                    prefix="reference_image_", min=0, max=9,
                )),
                io.Autogrow.Input("reference_videos", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Image.Input("reference_video", display_name="🎞️参考视频帧", tooltip="连接IMAGE批次，视频读取节点请设24fps。"),
                    prefix="reference_video_", min=0, max=3,
                )),
            ],
            outputs=[io.String.Output("h3_prompt", display_name="🎬H3提示词"),
                     io.String.Output("selected_skill", display_name="🧩选中Skill"),
                     io.String.Output("detected_mode", display_name="🧭识别模式")],
        )

    @classmethod
    def execute(cls, model, prompt, skill, duration, seed, max_tokens,
                video_sample_frames_per_sec, force_unload_model,
                reference_images=None, reference_videos=None):
        failed = True
        active = model
        try:
            if not prompt.strip():
                raise ValueError("请先填写需求描述。")
            if skill != "auto" and skill not in SKILL_NAMES:
                raise ValueError("所选Skill不存在，请刷新节点后重新选择。")
            images = _autogrow_values(reference_images)
            videos = _autogrow_values(reference_videos)
            _validate_reference_images(images)
            details = _reference_video_details(videos, video_sample_frames_per_sec)
            settings = _model_settings(model)
            if (images or videos) and settings["mmproj_file"] in ("None", "无", ""):
                raise ValueError("有参考图或视频时，请在本地模型加载器选择与主模型匹配的mmproj。")
            active = _sync_model(model)
            llm = active.llm
            seed = int(seed) % 0xFFFFFFFF
            mode = detect_h3_mode(len(images), len(videos))
            if mode is None:
                mode = parse_mode_selection(_route(llm, mode_router_prompt(prompt, len(images)), seed, 32), len(images))
            content, summary = _user_content(prompt, duration, images, details, mode)
            selected = skill
            if selected == "auto":
                selected = parse_skill_selection(_route(llm, router_prompt(prompt, mode, summary), seed, 96))
            # Text-only handlers expect a string, not a multimodal content array.
            user_content = content if images or videos else content[0]["text"]
            messages = [{"role": "system", "content": system_prompt(selected, mode, duration)},
                        {"role": "user", "content": user_content}]
            params = {**_sampling_settings(settings["think"]), "seed": seed,
                      "max_tokens": int(max_tokens), "stream": False}
            response, _ = _create_completion(llm, messages, params)
            result = _reply(response)
            issues = output_issues(result, mode, duration)
            if issues:
                messages.extend([
                    {"role": "assistant", "content": result},
                    {"role": "user", "content": "Return the complete requested content only, following the selected Skill. Problems: " + "; ".join(issues)},
                ])
                response, _ = _create_completion(llm, messages, params)
                result = _reply(response)
                if output_issues(result, mode, duration):
                    raise RuntimeError("模型未返回正文。请关闭思考，或提高上下文长度和最大输出tokens后重试。")
            failed = False
            return io.NodeOutput(result, selected, mode)
        finally:
            if failed or force_unload_model or _should_unload(model):
                _release_model(active)


NODE_CLASS_MAPPINGS = {"DapaoLocalH3Prompt": DapaoLocalH3Prompt}
NODE_DISPLAY_NAME_MAPPINGS = {"DapaoLocalH3Prompt": "本地H3提示词@炮老师的小课堂"}
