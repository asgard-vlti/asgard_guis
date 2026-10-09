import os
import tempfile
import unittest
from pathlib import Path

from PyQt5 import QtWidgets

from asgard_guis.cmd_scripts import log_viewer


class LogRenderingTests(unittest.TestCase):
    def render(self, text):
        return log_viewer.ansi_to_html(text, log_viewer.severity_ranges(text))

    def test_severity_labels_in_common_log_formats(self):
        lines = (
            "2026-10-09 12:30:00.123 ERROR failed\n"
            "2026-10-09 12:30:01 WARNING delayed\n"
            "[WARN] retrying\n"
            "error: lower case\n"
            "INFO: error in message\n"
            "terror and warningly are ordinary words"
        )
        rendered = self.render(lines)

        self.assertEqual(rendered.count(f"color: {log_viewer.ERROR_COLOR}"), 2)
        self.assertEqual(rendered.count(f"color: {log_viewer.WARNING_COLOR}"), 2)
        self.assertIn("INFO: error in message", rendered)
        self.assertIn("terror and warningly are ordinary words", rendered)

    def test_ansi_formatting_and_html_escaping_survive_label_color(self):
        text = "\x1b[1;32mERROR: <failure>\x1b[0m\nWARN: & retry"
        rendered = self.render(text)

        self.assertIn(
            f'<span style="color: {log_viewer.ERROR_COLOR}; font-weight: bold">ERROR</span>',
            rendered,
        )
        self.assertIn('<span style="color: #00cd00; font-weight: bold">: &lt;failure&gt;</span>', rendered)
        self.assertIn(f'<span style="color: {log_viewer.WARNING_COLOR}">WARN</span>', rendered)
        self.assertIn(": &amp; retry", rendered)

    def test_ansi_sequence_inside_label_keeps_severity_color(self):
        text = "ER\x1b[34mROR: failed\x1b[0m"
        rendered = self.render(text)

        self.assertEqual(rendered.count(f"color: {log_viewer.ERROR_COLOR}"), 2)
        self.assertIn('<span style="color: #0000ee">: failed</span>', rendered)


class LogViewerSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_dark_viewer_renders_and_filters_log(self):
        with tempfile.TemporaryDirectory() as log_root:
            log_dir = Path(log_root) / "mds"
            log_dir.mkdir()
            (log_dir / "log.txt").write_text(
                "2026-10-09 12:30:00 ERROR bad\n"
                "2026-10-09 12:30:01 WARNING slow\n",
                encoding="utf-8",
            )
            viewer = log_viewer.UniversalLogClient(log_root)
            try:
                tab = viewer.tabs.widget(0)
                self.assertIn("#1e1f22", viewer.styleSheet())
                self.assertIn("#272a30", viewer.styleSheet())
                self.assertIn(log_viewer.ERROR_COLOR, tab.text_area.toHtml())
                self.assertIn(log_viewer.WARNING_COLOR, tab.text_area.toHtml())

                tab.filter_input.setText("warning")
                self.assertNotIn("bad", tab.text_area.toPlainText())
                self.assertIn("slow", tab.text_area.toPlainText())
            finally:
                viewer.close()
