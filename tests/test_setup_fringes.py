import importlib.machinery
import importlib.util
import io
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "setup-fringes"
SPEC = importlib.util.spec_from_loader(
    "setup_fringes", importlib.machinery.SourceFileLoader("setup_fringes", str(SCRIPT))
)
setup_fringes = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = setup_fringes
SPEC.loader.exec_module(setup_fringes)


class SetupFringesTests(unittest.TestCase):
    def setUp(self):
        self.left = setup_fringes.Monitor("DP-1", 0, 0, 1920, 1080, True)
        self.right = setup_fringes.Monitor("DP-2", 1920, 0, 1920, 1080, False)

    def window(self, key, desktop=2, pid=100):
        titles = {
            "h_guis": "Heimdallr Real-Time Plots",
            "rtd": "Asgard CRED1 viewer",
            "text": "Asgard servers text interface: mimir",
            "shortcuts": "ASGARD Shortcuts",
            "dmview": "DM SHM Viewer - beam 1, 2, 3, 4",
            "logs": "Asgard log viewer: /home/asg/logs",
            "status": "Watchdog Status",
            "engineering": "Heimdallr/Baldr/Solarstein",
            "terminal": "asg@mimir:~",
        }
        wm_class = "SetupFringesMimir.XTerm" if key == "terminal" else "other"
        return setup_fringes.Window(
            hex(100 + setup_fringes.WINDOW_KEYS.index(key)),
            desktop,
            pid,
            0,
            0,
            700,
            500,
            wm_class,
            titles[key],
        )

    def test_monitor_and_workspace_parsing(self):
        output = (
            "DP-2 connected 1920x1080+0+0 (normal)\n"
            "DP-1 connected primary 1920x1080-1920+0 (normal)\n"
        )
        monitors = setup_fringes.parse_monitors(output)
        self.assertEqual([monitor.x for monitor in monitors], [-1920, 0])
        self.assertTrue(monitors[0].primary)
        self.assertEqual(
            setup_fringes.parse_current_desktop("0  - DG: 1x1\n2  * DG: 1x1"),
            2,
        )

    def test_layout_matches_requested_regions(self):
        geometry = lambda key: setup_fringes.geometry_for(
            key, self.left, self.right, self.window(key)
        )
        rtd = geometry("rtd")
        text = geometry("text")
        shortcuts = geometry("shortcuts")
        dmview = geometry("dmview")
        logs = geometry("logs")
        status = geometry("status")
        engineering = geometry("engineering")
        terminal = geometry("terminal")

        self.assertEqual((rtd.x, rtd.y), (8, 8))
        self.assertGreater(text.y, rtd.y)
        self.assertEqual(shortcuts.x, text.x + text.width + 8)
        self.assertEqual((dmview.width, dmview.height), (613, 431))
        self.assertLess(dmview.x, status.x)
        self.assertGreater(logs.y, dmview.y)
        self.assertEqual(status.y, 8)
        self.assertGreater(engineering.y, status.y)
        self.assertAlmostEqual(terminal.x + terminal.width / 2, 2880, delta=1)
        self.assertAlmostEqual(terminal.y + terminal.height / 2, 540, delta=1)
        self.assertIsNone(geometry("h_guis"))

    def test_unequal_monitor_sizes_stay_within_right_monitor(self):
        right = setup_fringes.Monitor("DP-2", 1920, -120, 2560, 1440, False)
        for key in ("dmview", "logs", "status", "engineering", "terminal"):
            placement = setup_fringes.geometry_for(
                key, self.left, right, self.window(key)
            )
            self.assertGreaterEqual(placement.x, right.x)
            self.assertGreaterEqual(placement.y, right.y)
            self.assertLessEqual(placement.x + placement.width, right.x + right.width)
            self.assertLessEqual(placement.y + placement.height, right.y + right.height)

    def test_launcher_commands_preserve_existing_routes(self):
        args = ("operator", ":0", ["-fn", "fixed"])
        dmview = setup_fringes.launch_command("dmview", *args)
        chrome = setup_fringes.launch_command("engineering", *args)
        terminal = setup_fringes.launch_command("terminal", *args)
        self.assertEqual(dmview[:3], ["ssh", "-XC", "mimir"])
        self.assertEqual(dmview[-2:], ["/home/asg/.conda/envs/asgard/bin/dmview", "-1"])
        self.assertEqual(chrome[:5], ["ssh", "-XY", "-t", "operator@wgsgb1", "google-chrome"])
        self.assertEqual(terminal[:3], ["xterm", "-fn", "fixed"])
        self.assertEqual(terminal[-2:], ["-e", "mimir"])

    def test_reuse_and_timeout(self):
        existing = [self.window(key) for key in setup_fringes.WINDOW_KEYS]
        self.assertEqual(setup_fringes.missing_keys(existing, 2), [])
        self.assertEqual(
            setup_fringes.missing_keys(existing[:-1], 2), ["terminal"]
        )
        now = [0.0]

        def pause(seconds):
            now[0] += seconds

        result = setup_fringes.wait_for_windows(
            lambda: existing[:-1],
            2,
            timeout=1,
            clock=lambda: now[0],
            pause=pause,
        )
        self.assertEqual(result, existing[:-1])
        self.assertEqual(now[0], 1)

    def test_window_listing_parses_class_and_title(self):
        line = (
            "0x01000001  2  123  20  30  520  330  wvguws "
            "setupfringesmimir.XTerm asg@mimir:~"
        )
        windows = setup_fringes.parse_windows(line)
        self.assertEqual(len(windows), 1)
        self.assertTrue(setup_fringes.matches("terminal", windows[0]))
        self.assertEqual(windows[0].title, "asg@mimir:~")

    def test_run_launches_only_missing_window(self):
        initial = [self.window(key) for key in setup_fringes.WINDOW_KEYS if key != "logs"]
        complete = initial + [self.window("logs")]
        responses = iter([initial, complete])

        def output(command):
            if command[0] == "xrandr":
                return (
                    "DP-1 connected primary 1920x1080+0+0\n"
                    "DP-2 connected 1920x1080+1920+0\n"
                )
            return "2  * DG: 1x1\n"

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(setup_fringes.shutil, "which", return_value="/bin/tool")
            )
            stack.enter_context(
                mock.patch.object(setup_fringes, "command_output", side_effect=output)
            )
            stack.enter_context(
                mock.patch.object(
                    setup_fringes, "read_windows", side_effect=lambda: next(responses)
                )
            )
            popen = stack.enter_context(mock.patch.object(setup_fringes.subprocess, "Popen"))
            stack.enter_context(mock.patch.object(setup_fringes.subprocess, "run"))
            stack.enter_context(mock.patch.object(setup_fringes, "move_to_desktop"))
            stack.enter_context(mock.patch.object(setup_fringes, "position_window"))
            result = setup_fringes.run("operator", ":0", [])

        self.assertEqual(result, 0)
        self.assertEqual(popen.call_count, 1)
        self.assertEqual(
            popen.call_args.args[0],
            ["ssh", "-XC", "mimir", "/home/asg/.conda/envs/asgard/bin/logs-client"],
        )

    def test_missing_window_is_reported_after_timeout(self):
        windows = [
            self.window(key) for key in setup_fringes.WINDOW_KEYS if key != "logs"
        ]

        def output(command):
            if command[0] == "xrandr":
                return (
                    "DP-1 connected primary 1920x1080+0+0\n"
                    "DP-2 connected 1920x1080+1920+0\n"
                )
            return "2  * DG: 1x1\n"

        errors = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(setup_fringes.shutil, "which", return_value="/bin/tool")
            )
            stack.enter_context(
                mock.patch.object(setup_fringes, "command_output", side_effect=output)
            )
            stack.enter_context(
                mock.patch.object(setup_fringes, "read_windows", return_value=windows)
            )
            stack.enter_context(
                mock.patch.object(setup_fringes, "wait_for_windows", return_value=windows)
            )
            stack.enter_context(mock.patch.object(setup_fringes.subprocess, "Popen"))
            stack.enter_context(mock.patch.object(setup_fringes.subprocess, "run"))
            stack.enter_context(mock.patch.object(setup_fringes, "move_to_desktop"))
            position = stack.enter_context(
                mock.patch.object(setup_fringes, "position_window")
            )
            stack.enter_context(mock.patch("sys.stderr", errors))
            result = setup_fringes.run("operator", ":0", [])

        self.assertEqual(result, 1)
        self.assertIn("Windows did not appear: logs", errors.getvalue())
        self.assertEqual(position.call_count, 7)

    def test_menu_item_is_first_and_has_separator(self):
        menu = (SCRIPT.parent / "misc-all.wmrc").read_text()
        lines = menu.split("Menu HdlrBaldrPanels", 1)[1].splitlines()
        self.assertIn('"Setup Fringes"', lines[2])
        self.assertIn("f.separator", lines[3])


if __name__ == "__main__":
    unittest.main()
