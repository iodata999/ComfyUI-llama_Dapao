"""Local-GGUF versions of the dapaoAPI common prompt tools."""

from . import detail_flow_prompt_node as detail_flow
from . import h3_prompt_box_node as h3_box
from . import h3_video_prompt_node as h3_video
from . import image_prompt_director_node as image_director
from . import music3_caption_prompt_node as music3
from . import seedance20_director_node as seedance
from . import visual_style_prompt_node as visual_style
from .local_common_tools import LOCAL_TOOL_CATEGORY, LocalPayloadClient, LocalToolNodeMixin


# Preserve each reference node's prompt assembly, validation and output layout;
# only its final OpenAI-compatible request is redirected to the local GGUF model.
detail_flow.DetailFlowLLMClient = LocalPayloadClient
h3_video.H3PromptLLMClient = LocalPayloadClient
image_director.ImagePromptLLMClient = LocalPayloadClient
music3.Music3CaptionLLMClient = LocalPayloadClient
seedance.SeedanceDirectorLLMClient = LocalPayloadClient
visual_style.VisualStyleLLMClient = LocalPayloadClient

for module in (detail_flow, h3_video, image_director, music3, seedance, visual_style):
    module.API_BASE_URL = "本地 llama.cpp"


class DapaoLocalDetailFlowPromptNode(LocalToolNodeMixin, detail_flow.DapaoDetailFlowPromptNode):
    RETURN_TYPES = detail_flow.DapaoDetailFlowPromptNode.RETURN_TYPES
    RETURN_NAMES = detail_flow.DapaoDetailFlowPromptNode.RETURN_NAMES
    OUTPUT_IS_LIST = detail_flow.DapaoDetailFlowPromptNode.OUTPUT_IS_LIST
    FUNCTION = "generate_prompt"
    CATEGORY = LOCAL_TOOL_CATEGORY


class DapaoLocalH3VideoPromptNode(LocalToolNodeMixin, h3_video.DapaoH3VideoPromptNode):
    RETURN_TYPES = h3_video.DapaoH3VideoPromptNode.RETURN_TYPES
    RETURN_NAMES = h3_video.DapaoH3VideoPromptNode.RETURN_NAMES
    FUNCTION = "generate_prompt"
    CATEGORY = LOCAL_TOOL_CATEGORY


class DapaoLocalAllroundImagePromptNode(LocalToolNodeMixin, image_director.DapaoAllroundImagePromptNode):
    RETURN_TYPES = image_director.DapaoAllroundImagePromptNode.RETURN_TYPES
    RETURN_NAMES = image_director.DapaoAllroundImagePromptNode.RETURN_NAMES
    FUNCTION = "generate_prompt"
    CATEGORY = LOCAL_TOOL_CATEGORY


class DapaoLocalMusic3CaptionPromptNode(LocalToolNodeMixin, music3.DapaoMusic3CaptionPromptNode):
    RETURN_TYPES = music3.DapaoMusic3CaptionPromptNode.RETURN_TYPES
    RETURN_NAMES = music3.DapaoMusic3CaptionPromptNode.RETURN_NAMES
    FUNCTION = "generate_prompt"
    CATEGORY = LOCAL_TOOL_CATEGORY


class DapaoLocalSeedance20DirectorNode(LocalToolNodeMixin, seedance.DapaoSeedance20DirectorNode):
    RETURN_TYPES = seedance.DapaoSeedance20DirectorNode.RETURN_TYPES
    RETURN_NAMES = seedance.DapaoSeedance20DirectorNode.RETURN_NAMES
    FUNCTION = "generate_prompt"
    CATEGORY = LOCAL_TOOL_CATEGORY


class DapaoLocalVisualStylePromptNode(LocalToolNodeMixin, visual_style.DapaoVisualStylePromptNode):
    RETURN_TYPES = visual_style.DapaoVisualStylePromptNode.RETURN_TYPES
    RETURN_NAMES = visual_style.DapaoVisualStylePromptNode.RETURN_NAMES
    FUNCTION = "generate_prompt"
    CATEGORY = LOCAL_TOOL_CATEGORY


class DapaoLocalH3PromptBoxNode(h3_box.DapaoH3PromptBoxNode):
    RETURN_TYPES = h3_box.DapaoH3PromptBoxNode.RETURN_TYPES
    RETURN_NAMES = h3_box.DapaoH3PromptBoxNode.RETURN_NAMES
    FUNCTION = "build_prompt"
    CATEGORY = LOCAL_TOOL_CATEGORY


NODE_CLASS_MAPPINGS_LOCAL_API_COMMON = {
    "DapaoLocalDetailFlowPromptNode": DapaoLocalDetailFlowPromptNode,
    "DapaoLocalH3PromptBoxNode": DapaoLocalH3PromptBoxNode,
    # Keep the existing local H3 node ID so saved workflows receive the
    # complete reference-node UI and @material editor without rewiring.
    "DapaoH3LocalPromptNode": DapaoLocalH3VideoPromptNode,
    "DapaoLocalAllroundImagePromptNode": DapaoLocalAllroundImagePromptNode,
    "DapaoLocalMusic3CaptionPromptNode": DapaoLocalMusic3CaptionPromptNode,
    "DapaoLocalSeedance20DirectorNode": DapaoLocalSeedance20DirectorNode,
    "DapaoLocalVisualStylePromptNode": DapaoLocalVisualStylePromptNode,
}

NODE_DISPLAY_NAME_MAPPINGS_LOCAL_API_COMMON = {
    "DapaoLocalDetailFlowPromptNode": "🛍️电商详情页提示词@炮老师的小课堂",
    "DapaoLocalH3PromptBoxNode": "🧙‍♂️H3专用提示词框@炮老师的小课堂",
    "DapaoH3LocalPromptNode": "🦊H3本地视频提示词生成@炮老师的小课堂",
    "DapaoLocalAllroundImagePromptNode": "🪂全能image提示词生成@炮老师的小课堂",
    "DapaoLocalMusic3CaptionPromptNode": "🎵Music3音乐提示词生成@炮老师的小课堂",
    "DapaoLocalSeedance20DirectorNode": "😶‍🌫️Seedance2全能导演@炮老师的小课堂",
    "DapaoLocalVisualStylePromptNode": "🎨全能视觉风格提示词@炮老师的小课堂",
}
