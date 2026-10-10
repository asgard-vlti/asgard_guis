import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from asgard_guis.cmd_scripts import camera_mode_settings, s_labmode, s_skymode


class FakeSocket:
    def __init__(self, name):
        self.name = name
        self.options = []

    def setsockopt(self, option, value):
        self.options.append((option, value))


class CameraModeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.snapshot_path = Path(self.directory.name) / "camera_settings.json"
        path_patch = mock.patch.object(
            camera_mode_settings, "SNAPSHOT_PATH", self.snapshot_path
        )
        path_patch.start()
        self.addCleanup(path_patch.stop)
        self.camera = FakeSocket("cam_server")
        self.mds = FakeSocket("MDS")
        self.commands = []

        def send(socket, command):
            self.commands.append((socket.name, command))
            if command == "get_gain":
                return "12"
            if command == "get_fps":
                return "500.0"
            if command == "status":
                return '{"nbreads": 64}'
            return "OK"

        self.send = send
        socket_patch = mock.patch.object(
            s_labmode.agu,
            "open_socket_connection",
            side_effect=lambda name: self.camera if name == "cam_server" else self.mds,
        )
        socket_patch.start()
        self.addCleanup(socket_patch.stop)
        send_patch = mock.patch.object(
            camera_mode_settings.agu, "send_and_get_response", side_effect=send
        )
        send_patch.start()
        self.addCleanup(send_patch.stop)
        sleep_patch = mock.patch.object(s_labmode.time, "sleep")
        sleep_patch.start()
        self.addCleanup(sleep_patch.stop)

    def write_snapshot(self, saved_at, gain=12, fps=500.0, nbreads=64):
        self.snapshot_path.write_text(
            json.dumps(
                {"saved_at": saved_at, "gain": gain, "fps": fps, "nbreads": nbreads}
            ),
            encoding="utf-8",
        )

    def camera_commands(self):
        return [command for name, command in self.commands if name == "cam_server"]

    def test_lab_saves_settings_before_setting_its_camera_configuration(self):
        with mock.patch.object(camera_mode_settings.time, "time", return_value=10000):
            s_labmode.main()

        self.assertEqual(
            self.camera_commands(),
            [
                "get_gain",
                "get_fps",
                "status",
                "ndmr_mode 1",
                "set_fps 1000.0",
                "set_gain 3",
                "make_dark",
            ],
        )
        self.assertEqual(
            json.loads(self.snapshot_path.read_text(encoding="utf-8")),
            {"gain": 12, "fps": 500.0, "saved_at": 10000, "nbreads": 64},
        )

    def test_repeated_lab_keeps_original_snapshot(self):
        self.write_snapshot(10000)
        s_labmode.main()

        self.assertEqual(
            self.camera_commands(),
            ["ndmr_mode 1", "set_fps 1000.0", "set_gain 3", "make_dark"],
        )
        self.assertEqual(camera_mode_settings.read_snapshot()["saved_at"], 10000)

    def test_concurrent_lab_save_preserves_first_snapshot(self):
        def another_process_saved_snapshot(_temporary, _destination):
            self.write_snapshot(9000, gain=8, fps=250)
            raise FileExistsError

        with mock.patch.object(
            camera_mode_settings.os, "link", side_effect=another_process_saved_snapshot
        ):
            saved = camera_mode_settings.save_snapshot_if_absent(self.camera)

        self.assertFalse(saved)
        self.assertEqual(
            camera_mode_settings.read_snapshot(),
            {"gain": 8, "fps": 250.0, "saved_at": 9000.0, "nbreads": 64},
        )

    def test_fresh_sky_restores_both_settings_and_consumes_snapshot(self):
        self.write_snapshot(10000)
        with mock.patch.object(camera_mode_settings.time, "time", return_value=10001):
            s_skymode.main()

        self.assertEqual(
            self.camera_commands(),
            ["ndmr_mode 64", "set_fps 500.0", "set_gain 12"],
        )
        self.assertFalse(self.snapshot_path.exists())
        self.assertEqual(self.commands[-4][1], "off SBB")

    def test_exactly_two_hours_old_is_fresh(self):
        self.write_snapshot(10000)
        self.assertEqual(
            camera_mode_settings.settings_for_sky(now=10000 + 7200),
            (500.0, 12, 64, "saved settings"),
        )

    def test_older_snapshot_restores_gain_and_fps_with_gcds_default(self):
        self.snapshot_path.write_text(
            json.dumps({"saved_at": 10000, "gain": 12, "fps": 500.0}),
            encoding="utf-8",
        )

        camera_mode_settings.restore_sky_settings(self.camera, now=10001)

        self.assertEqual(
            self.camera_commands(),
            ["ndmr_mode 1", "set_fps 500.0", "set_gain 12"],
        )

    def test_stale_sky_uses_server_defaults(self):
        self.write_snapshot(10000)
        camera_mode_settings.restore_sky_settings(self.camera, now=10000 + 7201)

        self.assertEqual(
            self.camera_commands(),
            ["ndmr_mode 1", "set_fps 1000.0", "set_gain 5"],
        )
        self.assertFalse(self.snapshot_path.exists())

    def test_missing_and_invalid_snapshots_use_server_defaults(self):
        for content in (None, "not JSON", '{"gain": 12}'):
            with self.subTest(content=content):
                self.commands.clear()
                if content is None:
                    self.snapshot_path.unlink(missing_ok=True)
                else:
                    self.snapshot_path.write_text(content, encoding="utf-8")
                camera_mode_settings.restore_sky_settings(self.camera, now=10000)
                self.assertEqual(
                    self.camera_commands(),
                    ["ndmr_mode 1", "set_fps 1000.0", "set_gain 5"],
                )
                self.assertFalse(self.snapshot_path.exists())

    def test_lab_query_failure_stops_before_camera_and_mechanism_changes(self):
        def fail_query(socket, command):
            self.commands.append((socket.name, command))
            return None

        with mock.patch.object(
            camera_mode_settings.agu, "send_and_get_response", side_effect=fail_query
        ):
            with self.assertRaisesRegex(SystemExit, "Lab Mode aborted"):
                s_labmode.main()

        self.assertEqual(self.commands, [("cam_server", "get_gain")])
        self.assertFalse(self.snapshot_path.exists())

    def test_lab_missing_ndmr_status_stops_before_camera_changes(self):
        def missing_nbreads(socket, command):
            self.commands.append((socket.name, command))
            if command == "status":
                return "{}"
            return "12" if command == "get_gain" else "500.0"

        with mock.patch.object(
            camera_mode_settings.agu,
            "send_and_get_response",
            side_effect=missing_nbreads,
        ):
            with self.assertRaisesRegex(SystemExit, "missing NDMR read count"):
                s_labmode.main()

        self.assertEqual(self.camera_commands(), ["get_gain", "get_fps", "status"])
        self.assertFalse(any(name == "MDS" for name, _ in self.commands))
        self.assertFalse(self.snapshot_path.exists())

    def test_lab_write_failure_stops_before_camera_and_mechanism_changes(self):
        with mock.patch.object(camera_mode_settings.os, "link", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(SystemExit, "disk full"):
                s_labmode.main()

        self.assertEqual(self.camera_commands(), ["get_gain", "get_fps", "status"])
        self.assertFalse(any(name == "MDS" for name, _ in self.commands))
        self.assertFalse(self.snapshot_path.exists())

    def test_sky_restore_failure_keeps_snapshot(self):
        self.write_snapshot(10000)

        def fail_gain(socket, command):
            self.commands.append((socket.name, command))
            return "ERROR: failed to set gain" if command == "set_gain 12" else "OK"

        with mock.patch.object(
            camera_mode_settings.agu, "send_and_get_response", side_effect=fail_gain
        ):
            with mock.patch.object(camera_mode_settings.time, "time", return_value=10001):
                with self.assertRaisesRegex(SystemExit, "Sky Mode camera restore failed"):
                    s_skymode.main()

        self.assertEqual(
            self.camera_commands(),
            ["ndmr_mode 64", "set_fps 500.0", "set_gain 12"],
        )
        self.assertTrue(self.snapshot_path.exists())

    def test_ndmr_restore_failure_keeps_snapshot(self):
        self.write_snapshot(10000)

        def fail_mode(socket, command):
            self.commands.append((socket.name, command))
            return "Error: failed to set mode" if command == "ndmr_mode 64" else "OK"

        with mock.patch.object(
            camera_mode_settings.agu, "send_and_get_response", side_effect=fail_mode
        ):
            with self.assertRaisesRegex(RuntimeError, "ndmr_mode 64"):
                camera_mode_settings.restore_sky_settings(self.camera, now=10001)

        self.assertEqual(self.camera_commands(), ["ndmr_mode 64"])
        self.assertTrue(self.snapshot_path.exists())


if __name__ == "__main__":
    unittest.main()
