"""无需加载真实GGUF的本地Skill/素材/多轮对话回归测试。"""
from __future__ import annotations

import asyncio
import inspect
import io
import json
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACKAGE = "dapao_local_testpkg"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules.setdefault(PACKAGE, package)

folder_paths = types.ModuleType("folder_paths")
folder_paths.models_dir = str(ROOT / "_test_models")
folder_paths.folder_names_and_paths = {}
folder_paths.get_filename_list = lambda _name: []
folder_paths.get_full_path = lambda _name, value: value
folder_paths.get_input_directory = lambda: tempfile.gettempdir()
sys.modules.setdefault("folder_paths", folder_paths)

comfy = types.ModuleType("comfy")
model_management = types.ModuleType("comfy.model_management")
model_management.soft_empty_cache = lambda: None
comfy.model_management = model_management
sys.modules.setdefault("comfy", comfy)
sys.modules.setdefault("comfy.model_management", model_management)

try:
    from PIL import Image as _PILImage  # noqa: F401
except ModuleNotFoundError:
    pil_module = types.ModuleType("PIL")
    pil_module.Image = types.SimpleNamespace(Image=object)
    sys.modules["PIL"] = pil_module
    image_utils = types.ModuleType(f"{PACKAGE}.image_input_utils")
    image_utils.MAX_INPUT_IMAGE_EDGE = 2048
    image_utils.resize_pil_for_input = lambda image, max_edge=2048: image
    image_utils.tensor_to_png_data_uris = lambda image, max_edge=2048: ["data:image/png;base64,ZmFrZQ=="]
    sys.modules[f"{PACKAGE}.image_input_utils"] = image_utils

fake_nodes = types.ModuleType(f"{PACKAGE}.nodes")


class FakeStorage:
    llm = None
    chat_handler = None
    current_config = {}

    @classmethod
    def clean(cls):
        cls.llm = None


class FakeLegacyChat:
    def _load_model(self, *args, **kwargs):
        return None


fake_nodes.DapaoLlamaStorage = FakeStorage
fake_nodes.Dapao_LlamaChat = FakeLegacyChat
fake_nodes.QWEN38_REASONING_OPTIONS = ["关闭", "自动", "低", "中等", "高"]
fake_nodes.KV_CACHE_DEFAULT = "默认(F16)"
fake_nodes.KV_CACHE_OPTIONS = ["默认(F16)", "Q8_0（更省显存）"]
fake_nodes.normalize_qwen38_reasoning_effort = lambda value: {
    "关闭": "off", "自动": "xhigh", "低": "low", "中等": "medium", "高": "xhigh",
}.get(str(value), str(value))
sys.modules[f"{PACKAGE}.nodes"] = fake_nodes

runtime = __import__(f"{PACKAGE}.skill_runtime", fromlist=["*"])
chat = __import__(f"{PACKAGE}.multi_turn_chat_node", fromlist=["*"])
loader = __import__(f"{PACKAGE}.skill_loader_node", fromlist=["*"])


class FakeLLM:
    def __init__(self, replies=None):
        self.replies = list(replies or ["本地回复"])
        self.messages = []
        self.calls = 0

    def tokenize(self, value, add_bos=False):
        return list(value)

    def create_chat_completion(self, **kwargs):
        self.calls += 1
        self.messages.append(kwargs["messages"])
        reply = self.replies[min(self.calls - 1, len(self.replies) - 1)]
        return {"choices": [{"message": {"content": reply}}]}

    def reset(self):
        return None


def source_model():
    return {
        "settings": {
            "family": "Qwen3.8-VL",
            "model_file": "fake.gguf",
            "mmproj_file": "None",
            "n_ctx": 8192,
            "n_gpu_layers": -1,
            "cache_type_k": "默认(F16)",
            "cache_type_v": "默认(F16)",
            "think": False,
            "preserve_thinking": False,
            "reasoning": "off",
            "cpu_moe": False,
            "n_cpu_moe": 0,
            "unload_after_run": False,
        },
    }


class LocalMultiTurnTests(unittest.TestCase):
    def setUp(self):
        self.original_sync = chat._sync_model
        self.fake_llm = FakeLLM()
        chat._sync_model = lambda _model: chat.DapaoLocalModel(
            llm=self.fake_llm,
            settings=source_model()["settings"],
            chat_handler=None,
        )

    def tearDown(self):
        chat._sync_model = self.original_sync

    @staticmethod
    def values(message="你好"):
        return {
            "🤖本地模型": source_model(),
            "💬本轮消息": message,
            "📚会话历史": "[]",
            "🖼️图片引用": "[]",
            "🧩流程状态": "{}",
            "🧩选项": "[]",
            "🆔请求标识": "1787390000000-test",
            "🧭执行动作": "chat",
        }

    def test_nodes_and_async_contract(self):
        self.assertTrue(inspect.iscoroutinefunction(chat.DapaoMultiTurnChatV2.run))
        self.assertEqual(chat.DapaoMultiTurnChatV2.RETURN_TYPES, ("STRING", "STRING", "STRING"))
        self.assertEqual(
            chat.DapaoMultiTurnChatV2.RETURN_NAMES,
            ("💬助手回复", "📚会话历史JSON", "🧩Skill最终结果"),
        )
        self.assertIn("DapaoLocalChatMaterialLibrary", chat.NODE_CLASS_MAPPINGS)
        self.assertEqual(chat.DapaoLocalChatMaterialLibrary.CATEGORY, "🍭大炮-llama-cpp")
        self.assertTrue(inspect.iscoroutinefunction(loader.DapaoSkillLoader.load_skill))

    def test_text_chat_returns_three_real_outputs(self):
        result = asyncio.run(chat.DapaoMultiTurnChatV2().run(**self.values()))
        self.assertEqual(result["result"][0], "本地回复")
        self.assertEqual(len(result["result"]), 3)
        history = json.loads(result["result"][1])
        self.assertEqual([item["role"] for item in history], ["user", "assistant"])
        self.assertEqual(result["result"][2], "本地回复")

    def test_empty_draft_replays_reply_without_model_sync(self):
        chat._sync_model = lambda _model: (_ for _ in ()).throw(AssertionError("不应同步模型"))
        values = self.values("")
        values["📚会话历史"] = json.dumps([
            {"role": "user", "content": "旧问题", "created_at": 10},
            {"role": "assistant", "content": "旧回复", "created_at": 20},
        ], ensure_ascii=False)
        result = asyncio.run(chat.DapaoMultiTurnChatV2().run(**values))
        self.assertEqual(result["result"][0], "旧回复")
        self.assertFalse(result["ui"]["✅已发送"][0])

    def test_publish_final_short_circuits_model_and_prefers_skill_result(self):
        chat._sync_model = lambda _model: (_ for _ in ()).throw(AssertionError("发布不得同步模型"))
        values = self.values("")
        values["🧭执行动作"] = "publish_final"
        values["📚会话历史"] = json.dumps([
            {"role": "assistant", "content": "最近回复", "created_at": 20},
        ], ensure_ascii=False)
        values["🧩流程状态"] = json.dumps({"final_result": "Skill最终成品", "context_cutoff": 10}, ensure_ascii=False)
        result = asyncio.run(chat.DapaoMultiTurnChatV2().run(**values))
        self.assertEqual(result["result"][0], "最近回复")
        self.assertEqual(result["result"][2], "Skill最终成品")
        self.assertEqual(self.fake_llm.calls, 0)

    def test_context_cutoff_keeps_display_but_filters_model_context(self):
        values = self.values("继续")
        values["📚会话历史"] = json.dumps([
            {"role": "user", "content": "旧问题", "created_at": 10},
            {"role": "assistant", "content": "旧回答", "created_at": 20},
            {"role": "user", "content": "新问题", "created_at": 200},
            {"role": "assistant", "content": "新回答", "created_at": 210},
        ], ensure_ascii=False)
        values["🧩流程状态"] = json.dumps({"context_cutoff": 100}, ensure_ascii=False)
        result = asyncio.run(chat.DapaoMultiTurnChatV2().run(**values))
        sent = json.dumps(self.fake_llm.messages[-1], ensure_ascii=False)
        self.assertNotIn("旧问题", sent)
        self.assertIn("新问题", sent)
        returned = json.loads(result["result"][1])
        self.assertEqual([item["content"] for item in returned[:4]], ["旧问题", "旧回答", "新问题", "新回答"])

    def test_material_mentions_are_current_turn_only(self):
        library = {"version": 1, "items": [
            {"kind": "image", "slot": 1, "token": "@图片1", "label": "正面", "value": object()},
            {"kind": "audio", "slot": 1, "token": "@音频1", "label": "旁白", "value": object()},
        ]}
        selected = runtime.select_material_mentions("只看 @图片1，不听别的", library)
        self.assertEqual([item["token"] for item in selected], ["@图片1"])
        with self.assertRaisesRegex(ValueError, "未连接"):
            runtime.select_material_mentions("请看 @图片2", library)
        history = [{"role": "user", "content": "上一轮 @图片1", "materials": [
            {"kind": "image", "slot": 1, "token": "@图片1", "label": "正面"},
        ]}]
        self.assertEqual(runtime.api_messages(history, 2048), [{"role": "user", "content": "上一轮 @图片1"}])

    def test_material_library_limits_single_image_batch(self):
        image = types.SimpleNamespace(shape=(2, 8, 8, 3))
        with self.assertRaisesRegex(ValueError, "请先拆分批次"):
            chat.DapaoLocalChatMaterialLibrary().build_library(**{"🏷️素材别名": "{}", "🖼️图片1": image})


class SkillRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.old_root = runtime.SKILLS_ROOT
        self.old_alias = runtime.SKILL_ALIAS_PATH
        runtime.SKILLS_ROOT = root / "skills"
        runtime.SKILL_ALIAS_PATH = root / "data" / "skill_display_names.json"
        runtime.SKILLS_ROOT.mkdir()
        for skill_id, description in (("poster-maker", "生成商业海报提示词"), ("audio-cleaner", "清理音频并生成处理建议")):
            folder = runtime.SKILLS_ROOT / skill_id
            folder.mkdir()
            (folder / "SKILL.md").write_text(
                f"---\nname: {skill_id}\ndescription: {description}\n---\n# {skill_id}\n",
                encoding="utf-8",
            )

    def tearDown(self):
        runtime.SKILLS_ROOT = self.old_root
        runtime.SKILL_ALIAS_PATH = self.old_alias
        self.temp.cleanup()

    def test_scan_alias_and_manual_precedence(self):
        self.assertEqual(len(runtime.list_skills()), 2)
        runtime.set_skill_display_name("poster-maker", "商业海报生成", "manual")
        item = runtime.get_skill("poster-maker")
        self.assertEqual(item["display_name"], "商业海报生成")
        self.assertEqual(item["display_source"], "manual")
        runtime.set_model_display_names({"poster-maker": "不应覆盖", "audio-cleaner": "音频清理助手"})
        self.assertEqual(runtime.get_skill("poster-maker")["display_name"], "商业海报生成")
        self.assertEqual(runtime.get_skill("audio-cleaner")["display_source"], "model")

    def test_local_ai_names_all_skills_with_one_call(self):
        fake = FakeLLM(["poster-maker\t商业海报设计\naudio-cleaner\t音频清理助手"])
        old_sync = chat._sync_model
        chat._sync_model = lambda _model: chat.DapaoLocalModel(fake, source_model()["settings"], None)
        try:
            result = loader._optimize_display_names(source_model(), "all", "")
        finally:
            chat._sync_model = old_sync
        self.assertEqual(fake.calls, 1)
        self.assertEqual(result["requested"], 2)
        self.assertEqual(result["updated"], 2)

    def test_zip_traversal_is_rejected(self):
        archive = Path(self.temp.name) / "bad.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("../escape/SKILL.md", "bad")
        with self.assertRaisesRegex(ValueError, "不安全路径"):
            runtime.install_uploaded_skills([("bad.zip", archive)], "zip")

    def test_state_v3_and_skill_contract(self):
        state = runtime.normalize_state({"context_cutoff": 123, "final_result": "成品"})
        self.assertEqual(state["version"], 3)
        self.assertEqual(state["context_cutoff"], 123)
        self.assertEqual(loader.DapaoSkillLoader.CATEGORY, "🍭大炮-llama-cpp")
        self.assertEqual(loader.DapaoSkillLoader.RETURN_NAMES, ("🧩Skill配置",))


class FrontendContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "web" / "js" / "dapao_multi_turn_chat_v2.js").read_text(encoding="utf-8")

    def test_material_and_conversation_controls_exist(self):
        for text in (
            "DapaoLocalChatMaterialLibrary",
            "清除上下文",
            "发送最终状态",
            "优化当前技能",
            "优化全部技能",
            "上传ZIP",
            "上传文件夹",
            "contentEditable",
        ):
            self.assertIn(text, self.source)

    def test_old_direct_image_upload_and_unsafe_html_are_absent(self):
        self.assertNotIn("/upload/image", self.source)
        self.assertNotIn("插入图片", self.source)
        self.assertNotIn(".innerHTML", self.source)
        self.assertNotIn("/dapao/local-skills/optimize-display-names", self.source)

    def test_final_publish_drops_local_model_dependency(self):
        self.assertIn('"🤖本地模型": undefined', self.source)
        self.assertIn('"🧩Skill配置": undefined', self.source)
        self.assertIn('"📦素材库": undefined', self.source)
        self.assertIn('"🧭执行动作": "publish_final"', self.source)


if __name__ == "__main__":
    unittest.main()
