import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asgard_guis.cmd_scripts import ob_creator


class ObCreatorTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.directory = Path(temporary_directory.name)
        original_directory = Path.cwd()
        os.chdir(self.directory)
        self.addCleanup(os.chdir, original_directory)

    def test_writes_to_run_directory(self):
        template = self.directory / "template.obd"
        template.write_text(
            "Header\n__TARGET_BLOCK_START__\n__NAME__ __DIT__ __GAIN__\n"
            "__TARGET_BLOCK_END__\nFooter\n",
            encoding="utf-8",
        )
        run_directory = self.directory / "run"
        run_directory.mkdir()
        os.chdir(run_directory)

        arguments = [
            "ob-creator",
            "HD123",
            "--mode",
            "ut",
            "--label",
            "../night",
            "--template",
            str(template),
        ]
        with patch.object(sys, "argv", arguments), patch.object(
            ob_creator,
            "extract_simbad_data",
            return_value={"NAME": "HD123", "K_FLOAT": 6.0},
        ), contextlib.redirect_stdout(io.StringIO()):
            ob_creator.main()

        output = run_directory / ".._night__HD123_UT.obd"
        self.assertEqual(
            output.read_text(encoding="utf-8"), "Header\nHD123 0.0004 30\nFooter\n"
        )

    def test_uses_included_template_from_another_directory(self):
        run_directory = self.directory / "run"
        run_directory.mkdir()
        os.chdir(run_directory)
        target = {
            "NAME": "HD123",
            "K_FLOAT": 6.0,
            "RA": "123456.789",
            "DEC": "+123456.789",
            "PMA": "0.",
            "PMD": "0.",
            "H": "6.0",
            "K": "6.0",
            "PARALLAX": "0.",
            "G": "6.0",
        }
        with patch.object(sys, "argv", ["ob-creator", "HD123", "--mode", "ut"]), patch.object(
            ob_creator, "extract_simbad_data", return_value=target
        ), contextlib.redirect_stdout(io.StringIO()):
            ob_creator.main()

        output = run_directory / "HD123_UT.obd"
        content = output.read_text(encoding="utf-8")
        self.assertIn('TEL.TARG.NAME           "HD123";', content)
        self.assertNotIn("__TARGET_BLOCK_START__", content)
        self.assertNotIn("__DIT__", content)

    def test_checks_template_before_query(self):
        arguments = ["ob-creator", "HD123", "--mode", "ut", "--template", "missing.obd"]
        with patch.object(sys, "argv", arguments), patch.object(
            ob_creator, "extract_simbad_data", side_effect=AssertionError("Query started")
        ), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                ob_creator.main()

        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
