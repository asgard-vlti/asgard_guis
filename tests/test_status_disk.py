import io
import os
import time
import unittest
import uuid
from contextlib import redirect_stdout
from unittest.mock import patch

import zmq

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from asgard_guis.cmd_scripts import status as status_module

DiskStatusClient = status_module.DiskStatusClient
StatusFormatter = status_module.StatusFormatter
TextStatusInfo = status_module.TextStatusInfo


def disk_reply():
    def check(state, age, limit=2.0):
        return {
            "state": state,
            "age_s": age,
            "limit_s": limit,
            "detail": "" if state == "fresh" else "no recent write",
        }

    return {
        "cred1": {
            "state": "green",
            "checks": {name: check("fresh", 1.0) for name in StatusFormatter.CRED1_STREAMS},
        },
        "ft_performance": {"state": "red", "checks": {"FT": check("stale", 3.0)}},
        "ft_settings": {
            "state": "green",
            "checks": {"ft_settings": check("fresh", 0.5, 3.0)},
        },
        "tt_performance": {
            "state": "yellow",
            "checks": {
                "beam1": check("fresh", 0.5),
                "beam2": check("stale", 4.0),
                "beam3": check("fresh", 0.6),
                "beam4": check("fresh", 0.7),
            },
        },
        "tt_settings": {
            "state": "green",
            "checks": {
                name: check("fresh", 0.5, 3.0)
                for name in StatusFormatter.TT_BEAMS
            },
        },
    }


def watchdog_reply():
    names = [
        *(f"BTT{beam}" for beam in range(1, 5)),
        *(f"BAO{beam}" for beam in range(1, 5)),
        "MDS",
        "CRED1",
        "DM",
        "HDLR",
        "back_end",
        "Eng gui",
        "Heim Telem",
        "Baldr TT Telem",
    ]
    status = {
        name: {"process": "running", "zmq": "open", "status": None}
        for name in names
    }
    status["CRED1"]["status"] = (
        '{"cam_status":"running","shm_error":false,"fps":2000,"gain":1}'
    )
    status["HDLR"]["status"] = '{"cnt":123,"locked":true}'
    status["MDS"]["status"] = '{"SDLA":2.4}'
    return status


class DiskStatusRenderTests(unittest.TestCase):
    def test_summary_colors_and_failure_detail(self):
        tasks = StatusFormatter().build_render_state(
            {}, disk_status=disk_reply(), disk_error=None
        )["tasks"]
        self.assertEqual([task["task_name"] for task in tasks], list(StatusFormatter.DISK_LABELS.values()))
        self.assertEqual([task["entries"][0]["color"] for task in tasks], ["green", "red", "yellow"])
        self.assertTrue(tasks[2]["has_yellow"])
        self.assertEqual(
            [entry["label"] for entry in tasks[2]["entries"][1:]],
            ["beam1", "beam2", "beam3", "beam4"],
        )
        self.assertEqual(tasks[2]["entries"][1]["color"], "green")
        self.assertIn("last write 4.0s ago", tasks[2]["entries"][2]["value"])

    def test_terminal_uses_explicit_colors(self):
        output = io.StringIO()
        with redirect_stdout(output):
            TextStatusInfo()._print_watchdog_status({}, disk_status=disk_reply())
        self.assertIn(StatusFormatter.YELLOW, output.getvalue())
        self.assertIn("beam2", output.getvalue())

    def test_cred1_summary_and_six_streams_stay_visible(self):
        payload = disk_reply()
        formatter = StatusFormatter()
        for state, stale_names, count in (
            ("green", (), 6),
            ("yellow", ("baldr2",), 5),
            ("red", StatusFormatter.CRED1_STREAMS, 0),
        ):
            payload["cred1"]["state"] = state
            for name, check in payload["cred1"]["checks"].items():
                check["state"] = "stale" if name in stale_names else "fresh"
            block = formatter.build_render_state({}, disk_status=payload)["tasks"][0]
            self.assertEqual(block["entries"][0]["value"], f"{count}/6 streams saving")
            self.assertEqual(
                [entry["label"] for entry in block["entries"][1:]],
                list(StatusFormatter.CRED1_STREAMS),
            )
            self.assertEqual(
                [
                    entry.get("short_value", entry["value"])
                    for entry in block["entries"][1:]
                ],
                [
                    "stale" if name in stale_names else "saving"
                    for name in StatusFormatter.CRED1_STREAMS
                ],
            )
            self.assertNotIn("limit", str(block["entries"]))
            self.assertNotIn("no recent write", str(block["entries"]))

        unavailable = formatter.build_render_state({}, disk_status={})["tasks"][0]
        self.assertEqual(unavailable["entries"][0]["value"], "?/6 streams saving")
        self.assertEqual(
            [entry["label"] for entry in unavailable["entries"][1:]],
            list(StatusFormatter.CRED1_STREAMS),
        )

    def test_overdue_or_invalid_reply_turns_all_summaries_red(self):
        formatter = StatusFormatter()
        for payload, error in ((disk_reply(), "disk status reply overdue"), ({}, None)):
            tasks = formatter.build_render_state(
                {}, disk_status=payload, disk_error=error
            )["tasks"]
            self.assertTrue(all(task["has_red"] for task in tasks))
            self.assertEqual(tasks[0]["entries"][0]["value"], "?/6 streams saving")
            self.assertTrue(
                all("cannot verify" in task["entries"][0]["value"] for task in tasks[1:])
            )
            self.assertEqual(
                [entry["label"] for entry in tasks[2]["entries"][1:]],
                ["beam1", "beam2", "beam3", "beam4"],
            )

    def test_client_marks_missing_reply_overdue(self):
        client = DiskStatusClient("inproc://missing-disk-status")
        self.addCleanup(client.close)
        with patch("asgard_guis.cmd_scripts.status.time.monotonic", return_value=1.0):
            client.tick()
        with patch("asgard_guis.cmd_scripts.status.time.monotonic", return_value=6.0):
            client.tick()
            self.assertEqual(client.result()[1], "waiting for disk status")
        with patch("asgard_guis.cmd_scripts.status.time.monotonic", return_value=11.1):
            client.tick()
            payload, error = client.result()
        self.assertIsNone(payload)
        self.assertIn("overdue", error)

    def test_client_keeps_recent_status_during_delayed_reply(self):
        client = DiskStatusClient("inproc://delayed-disk-status")
        self.addCleanup(client.close)
        client.payload = disk_reply()
        client.error = None
        client.last_reply_at = 1.0
        with patch("asgard_guis.cmd_scripts.status.time.monotonic", return_value=6.0):
            payload, error = client.result()
        self.assertEqual(payload["tt_performance"]["state"], "yellow")
        self.assertIsNone(error)

    def test_client_receives_disk_status_over_zmq(self):
        endpoint = f"inproc://disk-status-{uuid.uuid4()}"
        server = zmq.Context.instance().socket(zmq.REP)
        server.bind(endpoint)
        self.addCleanup(server.close)
        client = DiskStatusClient(endpoint)
        self.addCleanup(client.close)
        client.tick()
        self.assertTrue(server.poll(1000, zmq.POLLIN))
        self.assertEqual(server.recv_string(), "disk_status")
        server.send_json(disk_reply())
        for _ in range(10):
            client.tick()
            payload, error = client.result()
            if error is None:
                break
            time.sleep(0.01)
        self.assertIsNone(error)
        self.assertEqual(payload["tt_performance"]["state"], "yellow")

    def _window(self):
        if status_module.QtWidgets is None:
            self.skipTest("PyQt5 is unavailable")
        QtWidgets = status_module.QtWidgets
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.__class__._qt_app = app
        window = status_module.WatchdogStatusWindow(
            "inproc://absent-wd", "inproc://absent-mds"
        )
        window.timer.stop()
        self.addCleanup(window.close)
        self.assertIsNotNone(app)
        return window, app

    def test_gui_merges_saving_cards_and_fits_default_window(self):
        window, app = self._window()
        window.disk_client.payload = disk_reply()
        window.disk_client.error = None
        window.disk_client.last_reply_at = time.monotonic()
        window._render(watchdog_reply(), update_last_time=False)
        window.show()
        app.processEvents()

        self.assertEqual(len(window._boxes), 16)
        self.assertTrue(
            all(label not in window._boxes for label in StatusFormatter.DISK_LABELS.values())
        )
        cred1 = window._boxes["CRED1"]
        heim = window._boxes["Heim Telem"]
        tt = window._boxes["Baldr TT Telem"]
        self.assertEqual(cred1.saving_label.text(), "disk: 6/6 streams saved")
        self.assertEqual(heim.saving_label.text(), "disk: 1/2 streams saved")
        self.assertEqual(tt.saving_label.text(), "disk: 7/8 streams saved")
        self.assertIn("baldr1: saved", cred1.saving_label.toolTip())
        self.assertIn("hei_k2: saved", cred1.saving_label.toolTip())
        self.assertIn("Telemetry:", heim.saving_label.toolTip())
        self.assertIn("FT: not saving", heim.saving_label.toolTip())
        self.assertIn("Settings:", heim.saving_label.toolTip())
        self.assertIn("ft_settings: saved", heim.saving_label.toolTip())
        self.assertIn("beam2: not saving", tt.saving_label.toolTip())
        self.assertIn("beam4: saved", tt.saving_label.toolTip())
        self.assertIn("#4f5b73", cred1.styleSheet())
        self.assertIn("#ffd700", heim.styleSheet())
        self.assertIn("#ffd700", tt.styleSheet())

        positions = {
            name: window.grid.getItemPosition(window.grid.indexOf(box))
            for name, box in window._boxes.items()
        }
        self.assertEqual(
            [positions[f"BTT{beam}"] for beam in range(1, 5)],
            [(0, 0, 1, 3), (0, 3, 1, 3), (0, 6, 1, 3), (0, 9, 1, 3)],
        )
        self.assertEqual(positions["BAO1"], (1, 0, 1, 3))
        self.assertEqual(positions["CRED1"], (2, 0, 1, 4))
        self.assertEqual(positions["Eng gui"], (2, 10, 1, 2))
        self.assertEqual(positions["Heim Telem"], (3, 3, 1, 3))
        self.assertEqual(positions["Baldr TT Telem"], (3, 6, 1, 3))
        self.assertEqual(max(row for row, _, _, _ in positions.values()), 3)
        self.assertLessEqual(window.minimumSizeHint().width(), window.width())
        self.assertLessEqual(window.minimumSizeHint().height(), window.height())
        for name, box in window._boxes.items():
            self.assertGreater(box.width(), 0, name)
            self.assertGreater(box.height(), 0, name)
            self.assertTrue(window.rect().contains(box.geometry()), name)
        boxes = list(window._boxes.values())
        for index, box in enumerate(boxes):
            for other in boxes[index + 1 :]:
                self.assertFalse(box.geometry().intersects(other.geometry()))

        varying_positions = window._layout_positions(
            [name for name in watchdog_reply() if name != "DM"] + ["extra task"]
        )
        self.assertNotIn("DM", varying_positions)
        self.assertEqual(varying_positions["extra task"], (4, 0, 1, 3))

    def test_gui_updates_saving_on_cached_watchdog_and_keeps_process_red(self):
        window, app = self._window()
        payload = disk_reply()
        window.disk_client.payload = payload
        window.disk_client.error = None
        window.disk_client.last_reply_at = time.monotonic()
        watchdog = watchdog_reply()
        window._render(watchdog, update_last_time=True)
        cred1 = window._boxes["CRED1"]
        heim = window._boxes["Heim Telem"]
        tt = window._boxes["Baldr TT Telem"]

        payload["cred1"]["checks"]["baldr2"]["state"] = "stale"
        payload["cred1"]["state"] = "yellow"
        window._render(watchdog, update_last_time=False, evaluate_progress=False)
        self.assertEqual(cred1.saving_label.text(), "disk: 5/6 streams saved")
        self.assertIn("baldr2: not saving", cred1.saving_label.toolTip())
        self.assertIn("#ffd700", cred1.styleSheet())

        for check in payload["cred1"]["checks"].values():
            check["state"] = "stale"
        payload["cred1"]["state"] = "red"
        window.disk_client.payload["tt_performance"] = {
            "state": "green",
            "checks": {
                "beam1": {"state": "fresh", "age_s": 0.5, "limit_s": 2.0},
                "beam2": {"state": "fresh", "age_s": 0.5, "limit_s": 2.0},
                "beam3": {"state": "fresh", "age_s": 0.5, "limit_s": 2.0},
                "beam4": {"state": "fresh", "age_s": 0.5, "limit_s": 2.0},
            },
        }
        payload["ft_performance"]["state"] = "green"
        payload["ft_performance"]["checks"]["FT"]["state"] = "fresh"
        payload["ft_performance"]["checks"]["FT"]["age_s"] = 0.5
        window._render(watchdog, update_last_time=False, evaluate_progress=False)
        self.assertEqual(cred1.saving_label.text(), "disk: 0/6 streams saved")
        self.assertIn("#ffd700", cred1.styleSheet())
        self.assertEqual(heim.saving_label.text(), "disk: 2/2 streams saved")
        self.assertEqual(tt.saving_label.text(), "disk: 8/8 streams saved")
        self.assertIn("#4f5b73", tt.styleSheet())

        payload["ft_settings"]["state"] = "red"
        payload["ft_settings"]["checks"]["ft_settings"].update(
            state="stale", age_s=3.2, detail="no recent write"
        )
        payload["tt_settings"]["state"] = "yellow"
        payload["tt_settings"]["checks"]["beam3"].update(
            state="stale", age_s=3.2, detail="no recent write"
        )
        window._render(watchdog, update_last_time=False, evaluate_progress=False)
        self.assertEqual(heim.saving_label.text(), "disk: 1/2 streams saved")
        self.assertEqual(tt.saving_label.text(), "disk: 7/8 streams saved")
        self.assertIn("ft_settings: not saving", heim.saving_label.toolTip())
        self.assertIn("beam3: not saving", tt.saving_label.toolTip())
        self.assertIn("#ffd700", heim.styleSheet())
        self.assertIn("#ffd700", tt.styleSheet())

        del payload["ft_settings"]
        window._render(watchdog, update_last_time=False, evaluate_progress=False)
        self.assertEqual(heim.saving_label.text(), "disk: ?/2 streams saved")
        self.assertIn("FT: saved", heim.saving_label.toolTip())
        self.assertIn("ft_settings: unknown", heim.saving_label.toolTip())
        payload["tt_settings"]["state"] = "green"
        window._render(watchdog, update_last_time=False, evaluate_progress=False)
        self.assertEqual(tt.saving_label.text(), "disk: ?/8 streams saved")
        self.assertIn("beam3: unknown", tt.saving_label.toolTip())

        window.disk_client.error = "disk status reply overdue"
        window._render(watchdog, update_last_time=False, evaluate_progress=False)
        self.assertEqual(cred1.saving_label.text(), "disk: ?/6 streams saved")
        self.assertEqual(heim.saving_label.text(), "disk: ?/2 streams saved")
        self.assertEqual(tt.saving_label.text(), "disk: ?/8 streams saved")
        self.assertIn("disk status reply overdue", tt.saving_label.toolTip())
        self.assertIn("#ffd700", cred1.styleSheet())

        window.disk_client.error = None
        watchdog["CRED1"]["process"] = "closed"
        watchdog["Heim Telem"]["process"] = "closed"
        window._render(watchdog, update_last_time=False, evaluate_progress=True)
        self.assertIn("#ff7b7b", cred1.styleSheet())
        self.assertIn("#ff7b7b", heim.styleSheet())

        window.formatter.last_wd_time -= status_module.datetime.timedelta(seconds=11)
        window._render(watchdog, update_last_time=False, evaluate_progress=False)
        self.assertEqual(cred1.details.graphicsEffect().opacity(), 0.4)
        self.assertIsNone(cred1.saving_label.graphicsEffect())
        self.assertIsNotNone(app)

    def test_gui_keeps_saving_visible_without_watchdog_cards(self):
        window, _ = self._window()
        window.disk_client.payload = disk_reply()
        window.disk_client.error = None
        window.disk_client.last_reply_at = time.monotonic()
        window._render({}, update_last_time=False, evaluate_progress=False)
        self.assertEqual(set(window._boxes), {"CRED1", "Heim Telem", "Baldr TT Telem"})
        self.assertIn("watchdog status unavailable", window._boxes["CRED1"].details.text())
        self.assertEqual(window._boxes["CRED1"].saving_label.text(), "disk: 6/6 streams saved")


if __name__ == "__main__":
    unittest.main()
