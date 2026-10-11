import importlib.machinery
import importlib.util
import io
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from asgard_guis.cmd_scripts.window_geometry import apply_startup_geometry


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

    def test_layout_uses_left_monitor_even_when_right_is_primary(self):
        monitors = setup_fringes.parse_monitors(
            "DP-1 connected 1920x1080+0+0\n"
            "DP-2 connected primary 1920x1080+1920+0\n"
        )
        self.assertFalse(monitors[0].primary)
        self.assertEqual(
            setup_fringes.geometry_for("rtd", *monitors),
            setup_fringes.Geometry(8, 8, 850, 360),
        )

    def test_layout_matches_requested_regions(self):
        geometry = lambda key: setup_fringes.geometry_for(
            key, self.left, self.right
        )
        rtd = geometry("rtd")
        text = geometry("text")
        shortcuts = geometry("shortcuts")
        dmview = geometry("dmview")
        logs = geometry("logs")
        status = geometry("status")

        self.assertEqual((rtd.x, rtd.y), (8, 8))
        self.assertGreater(text.y, rtd.y)
        self.assertEqual(shortcuts.x, text.x + text.width + 8)
        self.assertGreaterEqual(shortcuts.y, rtd.y + rtd.height)
        self.assertEqual((dmview.width, dmview.height), (613, 431))
        self.assertLess(dmview.x, status.x)
        self.assertGreater(logs.y, dmview.y)
        self.assertEqual(status.y, 8)
        self.assertIsNone(geometry("h_guis"))
        self.assertIsNone(geometry("engineering"))
        self.assertIsNone(geometry("terminal"))

    def test_unequal_monitor_sizes_stay_within_right_monitor(self):
        right = setup_fringes.Monitor("DP-2", 1920, -120, 2560, 1440, False)
        for key in ("dmview", "logs", "status"):
            placement = setup_fringes.geometry_for(key, self.left, right)
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

    def test_only_python_gui_commands_receive_startup_geometry(self):
        args = ("operator", ":0", ["-fn", "fixed"])
        dmview = setup_fringes.command_with_geometry(
            setup_fringes.launch_command("dmview", *args), "dmview", self.left, self.right
        )
        text = setup_fringes.command_with_geometry(
            setup_fringes.launch_command("text", *args), "text", self.left, self.right
        )
        self.assertEqual(
            dmview[3:5], ["env", "ASGARD_SETUP_FRINGES_GEOMETRY=613x431+1928+324"]
        )
        self.assertEqual(text[:2], ["env", "ASGARD_SETUP_FRINGES_GEOMETRY=556x500+8+540"])
        for key in ("h_guis", "engineering", "terminal"):
            command = setup_fringes.launch_command(key, *args)
            self.assertEqual(
                setup_fringes.command_with_geometry(command, key, self.left, self.right),
                command,
            )

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
            "0x01000001  2  123  20  30  520  330  "
            "setupfringesmimir.XTerm wvguws asg@mimir:~"
        )
        windows = setup_fringes.parse_windows(line)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].wm_class, "setupfringesmimir.XTerm")
        self.assertTrue(setup_fringes.matches("terminal", windows[0]))
        self.assertEqual(windows[0].title, "asg@mimir:~")

    def test_x_window_tree_fallback_parses_real_window_lines(self):
        output = (
            '    0x5200006 "Asgard servers text interface: mimir": ()  '
            '517x500+0+0  +-2+617\n'
            '    0x280001e "asg@mimir:~": ("SetupFringesMimir" "XTerm")  '
            '665x344+0+0  +1934+637\n'
        )
        windows = setup_fringes.parse_tree_windows(output, 2)
        self.assertEqual(len(windows), 2)
        self.assertEqual((windows[0].x, windows[0].y), (-2, 617))
        self.assertTrue(setup_fringes.matches("text", windows[0]))
        self.assertTrue(setup_fringes.matches("terminal", windows[1]))

    def test_empty_wmctrl_list_uses_x_window_tree(self):
        with mock.patch.object(
            setup_fringes, "command_output", side_effect=["", '0x5200006 "Watchdog Status": ()  818x559+0+0  +3000+40']
        ) as output:
            windows = setup_fringes.read_windows(2)
        self.assertEqual(len(windows), 1)
        self.assertTrue(setup_fringes.matches("status", windows[0]))
        self.assertEqual(output.call_args.args[0], ["xwininfo", "-root", "-tree"])

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
                    setup_fringes, "read_windows", side_effect=lambda _: next(responses)
                )
            )
            popen = stack.enter_context(mock.patch.object(setup_fringes.subprocess, "Popen"))
            result = setup_fringes.run("operator", ":0", [])

        self.assertEqual(result, 0)
        self.assertEqual(popen.call_count, 1)
        self.assertEqual(
            popen.call_args.args[0],
            [
                "ssh", "-XC", "mimir", "env",
                "ASGARD_SETUP_FRINGES_GEOMETRY=1000x302+1928+738",
                "/home/asg/.conda/envs/asgard/bin/logs-client",
            ],
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
            stack.enter_context(mock.patch("sys.stderr", errors))
            result = setup_fringes.run("operator", ":0", [])

        self.assertEqual(result, 1)
        self.assertIn("Windows did not appear: logs", errors.getvalue())

    def test_menu_item_is_first_and_has_separator(self):
        menu = (SCRIPT.parent / "misc-all.wmrc").read_text()
        lines = menu.split("Menu HdlrBaldrPanels", 1)[1].splitlines()
        self.assertIn('"Setup Fringes"', lines[2])
        self.assertIn("f.separator", lines[3])

    def test_startup_geometry_is_optional(self):
        window = mock.Mock()
        with mock.patch.dict("os.environ", {}, clear=True):
            apply_startup_geometry(window)
        window.setGeometry.assert_not_called()
        with mock.patch.dict(
            "os.environ", {"ASGARD_SETUP_FRINGES_GEOMETRY": "613x431+1928+324"}
        ):
            apply_startup_geometry(window)
        window.setGeometry.assert_called_once_with(1928, 324, 613, 431)


if __name__ == "__main__":
    unittest.main()
