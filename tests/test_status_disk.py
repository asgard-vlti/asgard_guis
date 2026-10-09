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
    def check(state, age):
        return {
            "state": state,
            "age_s": age,
            "limit_s": 2.0,
            "detail": "" if state == "fresh" else "no recent write",
        }

    return {
        "cred1": {"state": "green", "checks": {"baldr1": check("fresh", 1.0)}},
        "ft_performance": {"state": "red", "checks": {"FT": check("stale", 3.0)}},
        "tt_performance": {
            "state": "yellow",
            "checks": {
                "beam1": check("fresh", 0.5),
                "beam2": check("stale", 4.0),
                "beam3": check("fresh", 0.6),
                "beam4": check("fresh", 0.7),
            },
        },
    }


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

    def test_overdue_or_invalid_reply_turns_all_summaries_red(self):
        formatter = StatusFormatter()
        for payload, error in ((disk_reply(), "disk status reply overdue"), ({}, None)):
            tasks = formatter.build_render_state(
                {}, disk_status=payload, disk_error=error
            )["tasks"]
            self.assertTrue(all(task["has_red"] for task in tasks))
            self.assertTrue(all("cannot verify" in task["entries"][0]["value"] for task in tasks))
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

    def test_gui_shows_partial_failure_details(self):
        if status_module.QtWidgets is None:
            self.skipTest("PyQt5 is unavailable")
        QtWidgets = status_module.QtWidgets
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = status_module.WatchdogStatusWindow(
            "inproc://absent-wd", "inproc://absent-mds"
        )
        window.timer.stop()
        window.disk_client.payload = disk_reply()
        window.disk_client.error = None
        window.disk_client.last_reply_at = time.monotonic()
        window._render(
            {"MDS": {"process": "running", "zmq": "open", "status": '{"SDLA": 2.4}'}},
            update_last_time=False,
            evaluate_progress=True,
        )
        box = window._boxes["TT performance saving"]
        self.assertIn("beam2", box.beam_labels[1].text())
        self.assertIn("beam4", box.beam_labels[3].text())
        self.assertIn("#ffd700", box.styleSheet())
        self.assertEqual(box.minimumHeight(), box.maximumHeight())
        self.assertLessEqual(
            abs(box.maximumHeight() - window._boxes["MDS"].sizeHint().height()), 5
        )
        self.assertEqual(window.grid.getItemPosition(window.grid.indexOf(box))[3], 2)
        window.disk_client.payload["tt_performance"] = {
            "state": "green",
            "checks": {
                "beam1": {"state": "fresh", "age_s": 0.5, "limit_s": 2.0},
                "beam2": {"state": "fresh", "age_s": 0.5, "limit_s": 2.0},
                "beam3": {"state": "fresh", "age_s": 0.5, "limit_s": 2.0},
                "beam4": {"state": "fresh", "age_s": 0.5, "limit_s": 2.0},
            },
        }
        window._render({}, update_last_time=False, evaluate_progress=False)
        beam_text = "".join(beam.text() for beam in box.beam_labels)
        self.assertTrue(all(f"beam{beam}" in beam_text for beam in range(1, 5)))
        self.assertIn("#4f5b73", box.styleSheet())
        window.close()
        self.assertIsNotNone(app)


if __name__ == "__main__":
    unittest.main()
