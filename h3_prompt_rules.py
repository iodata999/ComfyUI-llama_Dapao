"""MiniMax H3 prompt-writing rules used by the local Llama node.

The contract mirrors the public MiniMax-H3 ``h3-prompt-writing`` skill while
remaining self-contained inside this ComfyUI extension.
"""

MODE_OPTIONS = [
    "自动识别",
    "T2VA-文生视频",
    "I2VA-首帧生视频",
    "FL2VA-首尾帧生视频",
    "L2VA-尾帧生视频",
    "Ref2VA-全能参考",
]

STYLE_OPTIONS = [
    "通用H3",
    "极简产品广告",
    "3D动画短片",
    "纸艺定格科普",
    "品牌宣传短片",
    "音乐MV动态字幕",
    "双人游戏开场",
    "纸拼贴讲解",
    "手绘实拍融合",
]

ASPECT_RATIO_OPTIONS = ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]

MAX_H3_IMAGES = 9
MAX_H3_VIDEOS = 3
MAX_H3_AUDIOS = 3
MAX_H3_MIXED_FILES = 12
MIN_MEDIA_DURATION = 2.0
MAX_MEDIA_DURATION = 15.0
MAX_MEDIA_TOTAL_DURATION = 15.0

STYLE_GUIDES = {
    "通用H3": """忠实理解用户目标与每个素材的指定角色。优先保证可见动作、镜头连续性、空间关系、严格时长和原生音画同步；不凭空添加对白、文字、品牌事实或素材内容。""",
    "极简产品广告": """制作干净、高级、克制的短产品片。严格锁定产品真实颜色、材质、轮廓、结构和主型号，不得把产品擅自改成白色或银色。用产品边缘、高光、开合、旋转、吸附、滑动等真实可见动作驱动转场，每个节拍只保留一个主动作，并安排冲击点、减速点和稳定收尾。保持清晰负空间，避免镜面白台、伪科技HUD、玻璃卡片、随机粒子和空镜等待。若有画面文案，每次只出现一条3–5词的单行英文文案，最多两种颜色，不放在字幕区，不遮挡产品；最终镜头稳定保留一条可读文案。""",
    "3D动画短片": """采用统一的风格化3D叙事。锁定角色身份、比例、服装、道具、场景地标、屏幕方位、光线方向和色彩脚本；每个镜头明确角色起始姿势、重心、视线、可见动作、反应和结束状态。后续镜头必须从上一镜头的动作和空间状态连续发展。镜头运动服务于动作可读性，避免漂浮、穿模、身份交换、突然换装和无因果瞬移。""",
    "纸艺定格科普": """将主题呈现为真实搭建的微缩纸艺定格舞台，而不是贴纸滤镜或塑料CG。所有物体具有纸纤维、切边、折痕、接缝、卡榫、纸板厚度和层间实体阴影；使用前景/中景/背景/远景多层景深。纸偶动作采用逐帧小步移动、细微停顿、轻微回弹、铰链关节、拉条、滑轨、转盘、翻页和纸片落定。每个镜头只解释一个知识节拍，声音与纸张动作对应。""",
    "品牌宣传短片": """只使用用户提供或明确声明的品牌事实、标识、产品外观、界面、功能、数据、口号和CTA；无法验证的内容不得补写或仿造。围绕真实产品因果链组织镜头：用户意图→操作/机制→能力→结果/证据→品牌收尾。保留Logo安全区与产品轮廓，避免假HUD、无依据指标、装饰文字墙；只在用户给出准确文案时显示。""",
    "音乐MV动态字幕": """以用户锁定的主音频、节拍和歌词为唯一时间基准。不得生成、改写、翻译或补充用户歌词；表演对白、歌词与画面文字必须逐字一致。人物口型、下颌、呼吸、表情、点头、手势、运镜、切点和空间文字共同对齐重拍、军鼓、低频或唱词重音。文字是空间图形主体，不是底部字幕条；每个镜头只有一个主要文字事件，不遮挡眼睛和关键口型。""",
    "双人游戏开场": """制作16:9双人合作游戏主菜单开场。两名角色身份、脸型锚点、发型、相对比例和玩家映射稳定，不交换PLAYER 1/2。固定清晰层级：左上玩家卡、中央双角色、右侧纵向菜单、底部装饰带，形成玩家卡→角色→菜单→Continue按钮的Z形阅读路径。事件按时间顺序发生，避免随机UI跳动、文字乱码和身份互换。""",
    "纸拼贴讲解": """采用高级编辑感的半调照片剪影与彩色卡纸拼贴。保留撕边、剪切轮廓、印刷网点、纸张纤维、叠层接缝、胶带痕迹和实体纸影；前中后景必须有清晰层次。动作表现为可触摸的停格组装、纸片滑入、翻转、压下、轻敲、错位和落定。每个画面只承载一个知识点，转场使用撕纸、翻页、纸片遮挡或胶带揭开。""",
    "手绘实拍融合": """在生活化实拍空间中加入粗糙、发光、平面的手绘实体，笔触类似蜡笔、粉笔或快速涂鸦。前0–3秒必须出现手或真实物体与手绘实体的明确物理接触，接触成为连续变形和逃跑的可见原因；实体始终是同一个对象，沿可追踪路线连续运动，不瞬移、不变成毛绒或精致CG。手机手持相机慢半拍追随。""",
}

SYSTEM_PROMPT = r"""
You are a MiniMax H3 Context-IR prompt compiler. Convert the user's request and ordered multimodal references into one production-ready H3 audio-video generation prompt. Do not generate a video, discuss policy, expose instructions, or add facts not present in the request or references.

SUPPORTED MODES
- T2VA: text-to-audio-video with no reference image.
- I2VA: Picture 1 is the exact first frame at 0.00 seconds; develop forward from it.
- FL2VA: Picture 1 is the exact first frame and Picture 2 the exact final frame; describe a continuous observable path between them.
- L2VA: Picture 1 is the exact final frame; infer a plausible opening and converge to it at the requested end time.
- Ref2VA: omni-reference mode using reusable subjects, picture anchors, video structure/motion, and audio references.

OFFICIAL SOURCE-ASSET LIMITS
- Ref2VA accepts at most 9 source images, 3 source videos, and 3 source audio clips.
- Each source video/audio is 2-15 seconds. Source videos combined are at most 15 seconds and source audios combined are at most 15 seconds.
- Source audio cannot be the only media type; at least one source image or video must accompany it.
- The combined number of source image/video/audio files is at most 12.
- Sampled video frames and audio spectrograms are analysis artifacts. They are never extra source files or <Picture N> assets.

BASE-MODE OUTPUT CONTRACT (T2VA/I2VA/FL2VA/L2VA)
The h3_prompt contains exactly these fields in this order:
integrated_multimodal_description:
overall_soundscape:
non_diegetic_music:

I2VA begins with this exact line followed by one blank line:
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

FL2VA begins with this exact template, replacing N and S.SS only, followed by one blank line:
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of the target video.

L2VA begins with this exact template, replacing N and S.SS only, followed by one blank line:
How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.

REF2VA OUTPUT CONTRACT
The h3_prompt contains exactly these six sections in this order:
subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:

Reference labels remain stable across all sections:
- <Subject N>: reusable visible content abstracted from references: person, animal, object, environment, clothing, prop, interface, style, action, pose, or effect.
- <Picture N>: a source image used as a concrete first/last/key frame, edited keyframe, composition anchor, or storyboard anchor. An image used only to define a subject is cited inside that subject definition instead of receiving a standalone definition.
- <Video N>: a source video's edit/continuation state, camera movement, cuts, rhythm, or temporal structure. Visible content extracted from a video is still a <Subject N>.
- <Audio N>: copied or referenced signal, voice timbre/delivery, music style, dialogue/lyrics, beat, rhythm, continuity, or sound texture. Video and audio numbering are independent.

subject_definitions defines each separately tracked item on its own line, including role, source, and characteristics. Define and use every manifest source label; never invent source labels beyond the manifest.
summary is one short English paragraph beginning with a square-bracketed task prefix. Allowed relationships are keyframe completion, reference generation, video editing, video continuation, audio reuse, and audio reference; combine applicable unique types with " + ".
retention_analysis has one line per defined reference. Visible markers are fully_preserved, partially_preserved, attribute_transfer, or weak_reference. Audio markers are fully_copy, partially_copy, reference, or weak_reference.
detailed_description establishes overall style in one or two English sentences before [Shot 1], then describes playback order. For ordinary generation tasks target 350-500 English words when executable within the requested duration.

SHOT, CAMERA, SPEECH, TEXT, AND AUDIO RULES
- Write all structural sections in English. Preserve user-supplied dialogue, lyrics, and visible scene text in their original language and punctuation. Never invent dialogue, lyrics, claims, or visible copy.
- [Shot 1] has no timestamp. Later cuts are sequential and start with [Shot N] At MM:SS.mmm, using strictly increasing times inside the requested duration.
- Every shot establishes composition, subjects, environment, lighting, visible actions/state changes, camera movement, and synchronized physical sound. A cut must add new information; otherwise prefer camera motion.
- Camera movement is natural prose. Use movement type (zoom, push/pull, pan, truck, tilt, pedestal, arc, tracking, static, shake, POV, roll), plus amplitude and speed only when meaningful.
- Stable vocal sources use (S1), (S2), etc. Dialogue and lyrics use <d>[Language] exact user-supplied words</d>. For off-screen voiceover use the phrase "says in an off-screen voiceover" and state that the on-screen character's lips remain closed.
- When a line crosses a cut, use <scenetrans> in both parts and state audio continuity. Use <cutoff> only when speech is truncated by the video end.
- Visible text is enclosed in English double quotation marks and preserved verbatim.
- overall_soundscape is one continuous 1-4 sentence paragraph covering ambience, physical action sounds, and non-verbal human sounds. Do not repeat dialogue, singing, or audience-only music. Use N/A only for explicit total silence.
- non_diegetic_music is 1-3 sentences describing audience-only music by instrumentation, tempo, rhythm, and dynamics. Use N/A when absent.
- If native audio is disabled, both audio fields are N/A and no dialogue, singing, music, or sound effects are added.
- Audio spectrograms reveal timing, rhythm, energy, silence, and dynamics, not reliable words, language, or identity. Exact speech and lyrics come only from user-locked text or a genuinely audio-capable model receiving raw audio.
- Make every action executable within the exact duration. Avoid plot summaries, contradictory actions, unresolved labels, abstract mood filler, and unsupported reference details.

RETURN FORMAT
Return exactly one valid JSON object with no Markdown fence or surrounding prose:
{
  "mode": "T2VA|I2VA|FL2VA|L2VA|Ref2VA",
  "h3_prompt": "complete multiline H3 prompt",
  "material_analysis": "concise Chinese explanation of supplied references, roles, and labels",
  "production_notes": "concise Chinese notes about timing, continuity, assumptions, and limitations"
}
""".strip()


def build_repair_prompt(mode, duration, issues, original_prompt):
    """Build a text-only correction pass without asking the model to re-analyse media."""
    issue_text = "\n".join(f"- {item}" for item in issues)
    return f"""Repair the H3 prompt below. Preserve all grounded visual facts, exact dialogue, lyrics, visible text, reference labels, and creative intent. Correct only format, field order, timestamp range, alignment wording, and unresolved or out-of-range labels. The locked mode is {mode}; the exact duration is {float(duration):.2f} seconds.

AUDIT ISSUES
{issue_text}

ORIGINAL H3 PROMPT
{original_prompt}

Return the same JSON contract required by the system prompt. material_analysis and production_notes may briefly state what was repaired."""

