from __future__ import annotations

import re
import tempfile
import textwrap
import unittest
from pathlib import Path

try:
    from . import generate_cmd_scripts_reference as reference
except ImportError:  # Direct execution from the workspace root or repository
    import generate_cmd_scripts_reference as reference


class CommandScriptReferenceTests(unittest.TestCase):
    def _write_repository(
        self,
        workspace: Path,
        name: str,
        scripts: dict[str, str],
        sources: dict[str, str],
        package: str = "demo_pkg",
    ) -> Path:
        repository = workspace / name
        repository.mkdir()
        script_lines = "\n".join(
            f'{command} = "{target}"' for command, target in scripts.items()
        )
        (repository / "pyproject.toml").write_text(
            textwrap.dedent(
                f"""
                [project]
                name = "{name}"

                [project.scripts]
                {script_lines}

                [tool.hatch.build.targets.wheel]
                packages = ["{package}"]
                """
            ),
            encoding="utf-8",
        )
        for relative_path, source in sources.items():
            path = repository / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(source), encoding="utf-8")
        return repository

    def _load_one(
        self, workspace: Path, name: str
    ) -> tuple[reference.RepositoryReference, tuple[str, ...]]:
        repositories, warnings = reference.load_repositories(
            workspace, (Path(name),)
        )
        self.assertEqual(len(repositories), 1)
        return repositories[0], warnings

    def test_extracts_direct_helper_and_module_level_parsers_without_importing(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            self._write_repository(
                workspace,
                "demo",
                {
                    "direct": "demo_pkg.direct:main",
                    "helper": "demo_pkg.helper:main",
                    "module-level": "demo_pkg.module_level:main",
                },
                {
                    "demo_pkg/direct.py": """
                        import dependency_that_is_not_installed
                        raise RuntimeError("this module must not execute")

                        def main():
                            parser = argparse.ArgumentParser(description="Direct command")
                            parser.add_argument("beam", type=int, choices=[1, 2])
                            parser.add_argument("--debug", action="store_true", help="Debug output")
                            parser.parse_args()
                    """,
                    "demo_pkg/helper.py": """
                        import argparse

                        def build_parser():
                            parser = argparse.ArgumentParser(description="Helper command")
                            mode = parser.add_mutually_exclusive_group(required=True)
                            mode.add_argument("--one", action="store_true")
                            mode.add_argument("--two", action="store_true")
                            return parser

                        def main():
                            parser = build_parser()
                            parser.parse_args()
                    """,
                    "demo_pkg/module_level.py": """
                        import argparse

                        parser = argparse.ArgumentParser(description="Module command")
                        parser.add_argument("--count", type=int, default=3)
                        args = parser.parse_args()

                        def main():
                            return args.count
                    """,
                },
            )

            repository, warnings = self._load_one(workspace, "demo")

            self.assertEqual(warnings, ())
            scripts = {script.command: script for script in repository.scripts}
            self.assertEqual(scripts["direct"].description, "Direct command")
            self.assertEqual(
                [argument.destination for argument in scripts["direct"].arguments],
                ["beam", "debug"],
            )
            self.assertEqual(scripts["direct"].arguments[0].choices, ("1", "2"))
            self.assertEqual(scripts["helper"].description, "Helper command")
            self.assertEqual(
                scripts["helper"].exclusive_groups,
                (reference.ExclusiveGroup(("one", "two"), True),),
            )
            self.assertEqual(scripts["module-level"].description, "Module command")
            self.assertEqual(scripts["module-level"].arguments[0].default, "3")

    def test_manual_sys_argv_and_dynamic_expressions_are_documented_statically(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            self._write_repository(
                workspace,
                "demo",
                {
                    "manual": "demo_pkg.manual:main",
                    "dynamic": "demo_pkg.dynamic:main",
                },
                {
                    "demo_pkg/manual.py": """
                        import sys

                        def main():
                            if len(sys.argv) == 1:
                                host = "mimir"
                            elif len(sys.argv) == 2:
                                host = sys.argv[1]
                            else:
                                print("Usage: python manual.py <host_name>")
                                sys.exit(1)
                            return host
                    """,
                    "demo_pkg/dynamic.py": """
                        import argparse

                        def explode():
                            raise RuntimeError("must not execute")

                        def main():
                            parser = argparse.ArgumentParser(description="Dynamic command")
                            parser.add_argument(
                                "--mode",
                                default=explode(),
                                choices=[f"H{number}" for number in [1, 2]],
                            )
                            parser.parse_args()
                    """,
                },
            )

            repository, warnings = self._load_one(workspace, "demo")

            self.assertEqual(warnings, ())
            scripts = {script.command: script for script in repository.scripts}
            manual = scripts["manual"].arguments[0]
            self.assertEqual(manual.destination, "host_name")
            self.assertFalse(manual.required)
            self.assertEqual(manual.default, "mimir")
            dynamic = scripts["dynamic"].arguments[0]
            self.assertEqual(dynamic.default, "explode()")
            self.assertEqual(
                dynamic.choices_expression,
                "[f'H{number}' for number in [1, 2]]",
            )

    def test_package_prefix_fallback_is_annotated(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            self._write_repository(
                workspace,
                "demo",
                {"fallback": "demo_pkg.calibration.tool:main"},
                {"calibration/tool.py": "def main():\n    pass\n"},
            )

            repository, warnings = self._load_one(workspace, "demo")

            script = repository.scripts[0]
            self.assertTrue(script.nonstandard_resolution)
            self.assertEqual(script.source, workspace / "demo/calibration/tool.py")
            self.assertEqual(len(warnings), 1)
            document = reference.render_document(
                (repository,), workspace / "reference.md"
            )
            self.assertIn("Resolution note", document)

    def test_missing_source_and_entry_function_are_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            self._write_repository(
                workspace,
                "missing-source",
                {"broken": "demo_pkg.missing:main"},
                {},
            )
            with self.assertRaisesRegex(reference.ExtractionError, "was not found"):
                self._load_one(workspace, "missing-source")

            self._write_repository(
                workspace,
                "missing-function",
                {"broken": "demo_pkg.command:main"},
                {"demo_pkg/command.py": "def another_function():\n    pass\n"},
            )
            with self.assertRaisesRegex(reference.ExtractionError, "top-level function"):
                self._load_one(workspace, "missing-function")

    def test_write_if_changed_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reference.md"
            self.assertTrue(reference._write_if_changed(output, "content\n"))
            self.assertFalse(reference._write_if_changed(output, "content\n"))
            self.assertEqual(output.read_text(encoding="utf-8"), "content\n")


class RealWorkspaceIntegrationTests(unittest.TestCase):
    def test_every_declared_workspace_script_is_included_once(self):
        workspace = Path(__file__).resolve().parent.parent

        repositories, warnings = reference.load_repositories(workspace)

        self.assertEqual([item.name for item in repositories], list(map(str, reference.REPOSITORIES)))
        self.assertEqual(sum(len(item.scripts) for item in repositories), 40)
        for repository in repositories:
            commands = [script.command for script in repository.scripts]
            self.assertEqual(len(commands), len(set(commands)))
            self.assertFalse(
                any(
                    argument.manual
                    for script in repository.scripts
                    for argument in script.arguments
                )
            )
        text_clients = [
            (repository.name, script.target)
            for repository in repositories
            for script in repository.scripts
            if script.command == "text-clients"
        ]
        self.assertEqual(len(text_clients), 2)
        self.assertTrue(
            any("find-focal-masks" in warning for warning in warnings), warnings
        )

    def test_command_scripts_do_not_access_sys_argv_directly(self):
        workspace = Path(__file__).resolve().parent.parent
        roots = (
            workspace / "asgard_guis/asgard_guis/cmd_scripts",
            workspace / "asgard-alignment/asgard_alignment/cmd_scripts",
            workspace / "dcs/dcs/cmd_scripts",
        )
        sources = [path for root in roots for path in root.glob("*.py")]
        sources.append(workspace / "dcs/utils/universal_client.py")

        offenders = []
        for source in sources:
            text = source.read_text(encoding="utf-8")
            if re.search(r"\bsys\s*\.\s*argv\b", text):
                offenders.append(source.relative_to(workspace).as_posix())
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
