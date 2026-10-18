"""CPU-only parity checks for the isolated V3 service and existing behavior."""
from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
TEMPLATE_ROOT = PROJECT_ROOT / "model_information_qwen"
PACKAGE_ALIAS = "model_information_qwen_v3"
BUSINESS_MODULES = (
    "chat_engine", "generation", "normalization", "role_names", "phonetics", "protocol",
    "sessions", "server", "request_cache", "console", "pi_client", "turns",
)


def _register_package():
    name = "_qwen_v3_service_parity_bootstrap"
    bootstrap = sys.modules.get(name)
    if bootstrap is None:
        spec = importlib.util.spec_from_file_location(name, PACKAGE_ROOT / "bootstrap.py")
        if spec is None or spec.loader is None:
            raise ImportError("V3 package bootstrap is unavailable")
        bootstrap = importlib.util.module_from_spec(spec)
        sys.modules[name] = bootstrap
        spec.loader.exec_module(bootstrap)
    return bootstrap.register_package()


def _module(name):
    _register_package()
    return importlib.import_module(f"{PACKAGE_ALIAS}.{name}")


def _canonical_source(path):
    source = path.read_text(encoding="utf-8")
    if path.name == "server.py":
        # A dotted filesystem directory needs registration when this file is
        # executed directly. Exempt exactly these two statements, only inside
        # the existing empty-__package__ entry guard; no server logic is skipped.
        guard = ast.parse('if __package__ in (None, ""):\n    pass').body[0].test
        allowed = ast.parse(
            "import runpy\n"
            "runpy.run_path(str(Path(__file__).with_name('bootstrap.py')))['register_package']()\n").body
        ignored_lines = set()
        for node in ast.parse(source).body:
            if not isinstance(node, ast.If) or ast.dump(node.test) != ast.dump(guard):
                continue
            for index in range(len(node.body) - 1):
                pair = node.body[index:index + 2]
                if (all(ast.dump(actual) == ast.dump(expected) for actual, expected in zip(pair, allowed))
                        and all(statement.lineno == statement.end_lineno for statement in pair)):
                    ignored_lines.update(statement.lineno for statement in pair)
        source = "".join(line for index, line in enumerate(source.splitlines(keepends=True), 1)
                         if index not in ignored_lines)
    return (source
            .replace("model_information_qwen_v3", "model_information_qwen")
            .replace("model_information_qwen_3.0", "model_information_qwen"))


class BusinessSourceParityTests(unittest.TestCase):
    def test_all_business_implementations_match_template_except_package_paths(self):
        for name in BUSINESS_MODULES:
            with self.subTest(module=name):
                self.assertEqual(_canonical_source(PACKAGE_ROOT / f"{name}.py"),
                                 _canonical_source(TEMPLATE_ROOT / f"{name}.py"))

    def test_logging_and_pi_start_helper_keep_existing_implementation(self):
        for name in ("dialogue_log", "pi_startup"):
            with self.subTest(module=name):
                self.assertEqual(_canonical_source(PACKAGE_ROOT / f"{name}.py"),
                                 _canonical_source(TEMPLATE_ROOT / f"{name}.py"))

    def test_chat_settings_and_prompt_files_are_identical(self):
        self.assertEqual((PACKAGE_ROOT / "chat_config.json").read_bytes(),
                         (TEMPLATE_ROOT / "chat_config.json").read_bytes())
        new_prompts = {p.name: p.read_bytes() for p in (PACKAGE_ROOT / "prompts").iterdir() if p.is_file()}
        old_prompts = {p.name: p.read_bytes() for p in (TEMPLATE_ROOT / "prompts").iterdir() if p.is_file()}
        self.assertEqual(new_prompts, old_prompts)

    def test_no_old_service_package_imports_or_dynamic_import_targets(self):
        files = [*PACKAGE_ROOT.glob("*.py"), *PACKAGE_ROOT.joinpath("tools").glob("*.py")]
        for path in files:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            targets = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    targets.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    targets.append(node.module)
                elif isinstance(node, ast.Call) and node.args:
                    function_name = (node.func.id if isinstance(node.func, ast.Name)
                                     else node.func.attr if isinstance(node.func, ast.Attribute) else "")
                    if function_name in {"import_module", "__import__"} and isinstance(node.args[0], ast.Constant):
                        if isinstance(node.args[0].value, str):
                            targets.append(node.args[0].value)
            for target in targets:
                with self.subTest(file=path.name, target=target):
                    self.assertFalse(target == "model_information_qwen" or target.startswith("model_information_qwen."))

    def test_launcher_locks_and_remote_helper_reference_the_version_directory(self):
        source = (PACKAGE_ROOT / "launcher.py").read_text(encoding="utf-8")
        for relative in ("runs/server.lock", "runs/pi-start.lock", "pi_startup.py"):
            with self.subTest(path=relative):
                self.assertIn(f'model_information_qwen_3.0/{relative}', source)
                self.assertNotIn(f'model_information_qwen/{relative}', source)
                self.assertNotIn(f'model_information_qwen_v3/{relative}', source)
        deploy = (PACKAGE_ROOT / "deploy_pi.py").read_text(encoding="utf-8")
        self.assertIn("model_information_qwen_3.0/pi_startup.py", deploy)
        self.assertNotIn("model_information_qwen/pi_startup.py", deploy)


class RuntimeNamespaceParityTests(unittest.TestCase):
    def test_alias_and_business_modules_resolve_to_the_new_directory(self):
        package = _register_package()
        self.assertEqual(Path(package.__file__).resolve(), PACKAGE_ROOT / "__init__.py")
        for name in BUSINESS_MODULES:
            with self.subTest(module=name):
                module = _module(name)
                self.assertEqual(module.__package__, PACKAGE_ALIAS)
                self.assertEqual(Path(module.__file__).resolve(), PACKAGE_ROOT / f"{name}.py")

    def test_conversation_archive_has_an_independent_directory(self):
        archive = _module("dialogue_log").DEFAULT_LOG_PATH.resolve()
        self.assertEqual(archive, PACKAGE_ROOT / "runs/conversation.log")
        self.assertNotEqual(archive, TEMPLATE_ROOT / "runs/conversation.log")

    def test_server_inherits_four_completed_turns(self):
        server = _module("server")
        self.assertEqual(server.DEFAULT_MAX_TURNS, 4)
        engine = type("Engine", (), {"reply": lambda self, messages: {"reply": "回答", "meme_id": "meme-0001"}})()
        instance = server.InferenceServer(engine, "cpu-parity-only-token")
        self.assertEqual(instance.manager.max_turns, 4)


class SessionAndCacheParityTests(unittest.IsolatedAsyncioTestCase):
    async def exercise_twelve_unique_turns(self, max_turns):
        sessions = _module("sessions")
        calls = []

        class Engine:
            def reply(self, messages):
                calls.append(copy.deepcopy(messages))
                return {"reply": "回答：" + messages[-1]["content"], "meme_id": "meme-0001"}

        manager = sessions.DialogueManager(Engine(), max_turns=max_turns,
                                           clock=lambda: 0.0, cleanup_interval=None)
        manager.start()
        conversation_ids = []
        revisions = []
        try:
            for number in range(1, 13):
                future = manager.accept("v3-parity-session", f"第{number}轮")
                await future
                conversation_ids.append(future.conversation_id)
                revisions.append(future.background_revision)
            return calls, conversation_ids, manager.history("v3-parity-session"), revisions
        finally:
            await manager.close()

    async def test_default_four_turn_window_slides_without_reset_and_changes_background(self):
        server = _module("server")
        calls, conversation_ids, history, revisions = await self.exercise_twelve_unique_turns(server.DEFAULT_MAX_TURNS)
        self.assertEqual([len(messages) for messages in calls], [1,3,5,7] + [9]*8)
        self.assertEqual(len(set(conversation_ids)), 1)
        self.assertEqual(revisions, [0]*8 + [1]*4)
        self.assertEqual(calls[4][0], {"role":"user", "content":"第1轮"})
        self.assertEqual(calls[5][0], {"role":"user", "content":"第2轮"})
        self.assertEqual(calls[10][0], {"role":"user", "content":"第7轮"})
        expected = [message for number in range(9,13) for message in (
            {"role":"user", "content":f"第{number}轮"},
            {"role":"assistant", "content":f"回答：第{number}轮"})]
        self.assertEqual(history, expected)

    async def test_explicit_none_keeps_twelve_unique_turns_in_the_same_context(self):
        calls, conversation_ids, history, revisions = await self.exercise_twelve_unique_turns(None)
        self.assertEqual([len(messages) for messages in calls], list(range(1,24,2)))
        self.assertEqual(len(set(conversation_ids)), 1)
        self.assertEqual(revisions, [0]*12)
        expected = [message for number in range(1,13) for message in (
            {"role":"user", "content":f"第{number}轮"},
            {"role":"assistant", "content":f"回答：第{number}轮"})]
        self.assertEqual(history, expected)

    async def test_v2_receipts_keep_result_and_server_instance_isolation(self):
        cache_type = _module("request_cache").RequestCache
        cache = cache_type(clock=lambda: 0.0)
        other = cache_type(clock=lambda: 0.0)
        try:
            receipt = cache.register("one-session", "same-request", "原问题")
            response = {"type": "result", "id": "same-request", "reply": "回答", "meme_id": "meme-0001"}
            cache.finish(receipt, response)
            self.assertIs(cache.get("one-session", "same-request"), receipt)
            self.assertEqual(await receipt.completed, response)
            self.assertIsNone(other.get("one-session", "same-request"))
            with self.assertRaises(ValueError):
                cache.register("one-session", "same-request", "不同文本")
        finally:
            cache.close()
            other.close()


class V3ModelIdentityTests(unittest.TestCase):
    def test_loader_defaults_and_selected_loss_metadata_are_pinned_to_v3(self):
        model = _module("model")
        self.assertEqual(model.DEFAULT_CONFIG_PATH.resolve(), PACKAGE_ROOT / "inference_config.json")
        run = PROJECT_ROOT / "model_qwen/runs/qwen3_8b_v3_loss_ce"
        self.assertEqual(model.DEFAULT_MODEL_DIR.resolve(), run)
        selection = json.loads((run / "generator/loss_selection.json").read_text(encoding="utf-8"))
        self.assertEqual(selection["model_version"], "3.0.0")
        self.assertEqual(selection["version"], "qwen3_8b_v3_loss_ce")
        self.assertEqual(selection["best_step"], 150000)
        self.assertEqual(selection["global_step"], 157137)
        self.assertEqual(selection["selection_criterion"], "dev_response_ce")
        self.assertEqual(selection["best_checkpoint_sha256"],
                         "87a8ed11e093ea073633720f420f517d4bc311fb4deb864292f1dee09919a601")
        self.assertFalse(selection["chat_quality_evaluated"])

    def test_inspection_reports_semantic_version_and_the_loss_selected_step_without_loading_weights(self):
        model = _module("model")
        info = model.inspect_model_version()
        self.assertEqual(info["version"], "3.0.0")
        self.assertEqual(info["model_version"], "3.0.0")
        self.assertEqual(info["base_model_version"], "qwen3_8b_v3_loss_ce")
        self.assertEqual(info["selected_step"], 150000)
        self.assertEqual(info["quality_status"], "not_evaluated")
        self.assertEqual(info["chat_quality_status"], "not_evaluated")
        self.assertTrue(info["file_hashes_verified"])


if __name__ == "__main__":
    unittest.main()
