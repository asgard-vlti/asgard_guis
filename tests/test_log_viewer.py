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

    def test_explicit_timezones_keep_ansi_severity_highlighting(self):
        for timestamp in (
            "2026-10-12T00:00:00.123Z",
            "2026-10-12T00:00:00Z",
            "2026-10-11T21:00:00.123-03:00",
            "2026-10-12 00:00:00,123+0000",
        ):
            with self.subTest(timestamp=timestamp):
                rendered = self.render(f"{timestamp} \x1b[31m[ERROR]\x1b[0m failed\n")
                self.assertIn(log_viewer.ERROR_COLOR, rendered)
                self.assertIn("failed", rendered)

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
                self.assertEqual(tab.log_path(), str(log_dir / "log.txt"))
                self.assertIn("#1e1f22", viewer.styleSheet())
                self.assertIn("#272a30", viewer.styleSheet())
                self.assertIn(log_viewer.ERROR_COLOR, tab.text_area.toHtml())
                self.assertIn(log_viewer.WARNING_COLOR, tab.text_area.toHtml())

                tab.filter_input.setText("warning")
                self.assertNotIn("bad", tab.text_area.toPlainText())
                self.assertIn("slow", tab.text_area.toPlainText())
            finally:
                viewer.close()

    def test_newest_start_timestamp_wins_and_refresh_switches_files(self):
        with tempfile.TemporaryDirectory() as log_root:
            log_dir = Path(log_root) / "heimdallr"
            log_dir.mkdir()
            newest = log_dir / "log_20261011T120000.000000002Z_21.txt"
            older = log_dir / "log_20261011T120000.000000001Z_22.txt"
            newest.write_text("new run\n", encoding="utf-8")
            older.write_text("old run\n", encoding="utf-8")
            (log_dir / "log.txt").write_text("legacy run\n", encoding="utf-8")
            (log_dir / "log_20261012T120000.000000000Z_23.txt.bak").write_text(
                "unrelated file\n", encoding="utf-8"
            )

            viewer = log_viewer.UniversalLogClient(log_root)
            try:
                tab = viewer.tabs.widget(log_viewer.SERVER_DIRS.index("heimdallr"))
                self.assertEqual(tab.log_path(), str(newest))
                self.assertIn("new run", tab.text_area.toPlainText())

                restarted = log_dir / "log_20261011T120001.000000000Z_24.txt"
                restarted.write_text("restarted run\n", encoding="utf-8")
                tab.refresh_log()
                self.assertEqual(tab.log_path(), str(restarted))
                self.assertIn("restarted run", tab.text_area.toPlainText())
                self.assertNotIn("new run", tab.text_area.toPlainText())
            finally:
                viewer.close()

    def test_baldr_instance_selector_uses_each_instances_latest_log(self):
        with tempfile.TemporaryDirectory() as log_root:
            for server in ("baldr", "baldr_tt"):
                for instance in ("1", "2"):
                    log_dir = Path(log_root) / server / instance
                    log_dir.mkdir(parents=True)
                    (log_dir / f"log_20261011T120000.000000000Z_{instance}.txt").write_text(
                        f"{server} instance {instance}\n", encoding="utf-8"
                    )

            viewer = log_viewer.UniversalLogClient(log_root)
            try:
                for tab_index, server in enumerate(("baldr", "baldr_tt"), len(log_viewer.SERVER_DIRS)):
                    tab = viewer.tabs.widget(tab_index)
                    self.assertIn(f"{server} instance 1", tab.text_area.toPlainText())
                    tab.instance_dropdown.setCurrentText("2")
                    self.assertIn(f"{server} instance 2", tab.text_area.toPlainText())
                    self.assertNotIn(f"{server} instance 1", tab.text_area.toPlainText())
            finally:
                viewer.close()

    def test_control_line_uses_selected_log_without_creating_legacy_file(self):
        with tempfile.TemporaryDirectory() as log_root:
            log_dir = Path(log_root) / "mds"
            log_dir.mkdir()
            older = log_dir / "log_20261011T120000.000000000Z_20.txt"
            newest = log_dir / "log_20261011T120001.000000000Z_21.txt"
            older.write_text("old run\n", encoding="utf-8")
            newest.write_text("new run\n", encoding="utf-8")

            viewer = log_viewer.UniversalLogClient(log_root)
            try:
                mds_tab = viewer.tabs.widget(log_viewer.SERVER_DIRS.index("mds"))
                mds_tab.append_action_line("Restarted mds.")
                self.assertIn("[CONTROL] Restarted mds.", newest.read_text(encoding="utf-8"))
                self.assertNotIn("[CONTROL]", older.read_text(encoding="utf-8"))

                eng_gui_tab = viewer.tabs.widget(log_viewer.SERVER_DIRS.index("eng_gui"))
                eng_gui_tab.append_action_line("No log yet.")
                self.assertFalse((Path(log_root) / "eng_gui" / "log.txt").exists())
            finally:
                viewer.close()
