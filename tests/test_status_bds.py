import json
import os
import time
import unittest
import uuid
from unittest.mock import patch

import zmq

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from asgard_guis.cmd_scripts import status as status_module


def watchdog_status(process="running"):
    return {
        "MDS": {
            "process": process,
            "zmq": "open",
            "status": json.dumps({"service": "ready"}),
        }
    }


def readings(position, sdla="2.4"):
    return {
        "SDLA": sdla,
        **{axis: str(position) for axis in status_module.StatusFormatter.BDS_BEAMS},
    }


def mds_task(raw_readings, process="running"):
    status = status_module.add_mds_to_status(watchdog_status(process), raw_readings)
    return status_module.StatusFormatter().build_render_state(status)["tasks"][0]


class BdsStatusTests(unittest.TestCase):
    def test_uniform_bif_positions_are_normal(self):
        for position, label in ((133.07, "BIF H"), (63.07, "BIF Y/J")):
            with self.subTest(label=label):
                task = mds_task(readings(position + 0.09))
                bds = next(entry for entry in task["entries"] if entry["label"] == "BDS")
                self.assertEqual(bds["value"], label)
                self.assertEqual(len(bds["tooltip"].splitlines()), 4)
                self.assertFalse(task["has_red"])

    def test_mixed_beams_show_readings_in_tooltip_and_red_border(self):
        raw = readings(133.07)
        raw["BDS3"] = "63.07"
        task = mds_task(raw)
        self.assertTrue(task["has_red"])
        self.assertEqual([entry["label"] for entry in task["entries"]], [
            "process", "zmq", "SDLA", "BDS"
        ])
        self.assertIn("mixed", task["entries"][-1]["value"])
        self.assertIn("BDS3: BIF_YJ", task["entries"][-1]["tooltip"])

    def test_align_and_empty_are_named_but_red(self):
        align = readings(133.07)
        for axis, position in zip(
            status_module.StatusFormatter.BDS_BEAMS,
            status_module.StatusFormatter.BDS_ALIGN_POSITIONS,
        ):
            align[axis] = str(position)
        for raw, expected in ((align, "align"), (readings(0.0), "empty")):
            with self.subTest(expected=expected):
                task = mds_task(raw)
                self.assertTrue(task["has_red"])
                self.assertEqual(task["entries"][-1]["value"], expected)

    def test_unnamed_invalid_and_missing_readings_are_red(self):
        unnamed = mds_task(readings(20.0))
        self.assertEqual(unnamed["entries"][-1]["value"], "unnamed")
        for value, expected in (
            ("20.0", "unnamed"),
            ("nan", "unknown"),
            ("NACK: unavailable", "unknown"),
            (None, "unknown"),
        ):
            with self.subTest(value=value):
                raw = readings(133.07)
                raw["BDS4"] = value
                task = mds_task(raw)
                self.assertTrue(task["has_red"])
                self.assertIn(expected, task["entries"][-1]["tooltip"])

    def test_sdla_yellow_and_process_red_are_preserved(self):
        task = mds_task(readings(133.07, sdla="0.0"))
        self.assertFalse(task["has_red"])
        self.assertTrue(task["has_yellow"])
        task = mds_task(readings(133.07, sdla=None))
        self.assertTrue(task["has_yellow"])
        task = mds_task(readings(133.07), process="closed")
        self.assertTrue(task["has_red"])

    def test_mds_client_reads_all_axes_without_blocking(self):
        endpoint = f"inproc://bds-status-{uuid.uuid4()}"
        server = zmq.Context.instance().socket(zmq.REP)
        server.bind(endpoint)
        self.addCleanup(server.close)
        client = status_module.MdsStatusClient(endpoint)
        self.addCleanup(client.close)

        client.tick()
        replies = ("2.4", "133.07", "133.07", "NACK", "133.07")
        for axis, reply in zip(client.AXES, replies):
            self.assertTrue(server.poll(1000, zmq.POLLIN))
            self.assertEqual(server.recv_string(), f"read {axis}")
            server.send_string(reply)
            for _ in range(20):
                client.tick()
                if client._cycle is None or client._axis_index > client.AXES.index(
                    axis
                ):
                    break
                time.sleep(0.01)
        self.assertEqual(client.result()["BDS3"], "NACK")

    def test_mds_client_timeout_returns_unknown(self):
        client = status_module.MdsStatusClient(f"inproc://absent-bds-{uuid.uuid4()}")
        self.addCleanup(client.close)
        with patch("asgard_guis.cmd_scripts.status.time.monotonic", return_value=0.0):
            start = time.perf_counter()
            client.tick()
            self.assertLess(time.perf_counter() - start, 0.2)
        for index in range(len(client.AXES)):
            with patch(
                "asgard_guis.cmd_scripts.status.time.monotonic",
                return_value=2.0 * (index + 1),
            ):
                client.tick()
        self.assertIsNone(client._cycle)
        with patch("asgard_guis.cmd_scripts.status.time.monotonic", return_value=10.0):
            self.assertTrue(all(value is None for value in client.result().values()))

    def test_gui_border_and_details(self):
        if status_module.QtWidgets is None:
            self.skipTest("PyQt5 is unavailable")
        QtWidgets = status_module.QtWidgets
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.__class__._qt_app = app
        window = status_module.WatchdogStatusWindow(
            "inproc://absent-wd-bds", "inproc://absent-mds-bds"
        )
        window.timer.stop()
        self.addCleanup(window.close)
        window.mds_client.readings = readings(133.07)
        window.mds_client._completed_at = time.monotonic()
        window._render(watchdog_status(), update_last_time=True)
        box = window._boxes["MDS"]
        self.assertIn("BIF H", box.details.text())
        self.assertIn("BDS1: BIF_H", box.toolTip())
        self.assertIn("#4f5b73", box.styleSheet())
        line_count = box.details.text().count("<br/>")

        window.mds_client.readings["BDS2"] = "63.07"
        window._render(watchdog_status(), update_last_time=False)
        self.assertNotIn("BDS2", box.details.text())
        self.assertIn("BDS2: BIF_YJ", box.toolTip())
        self.assertEqual(box.details.text().count("<br/>"), line_count)
        self.assertIn("#ff7b7b", box.styleSheet())

        window.mds_client.readings = readings(63.07, sdla="0.0")
        window._render(watchdog_status(), update_last_time=False)
        self.assertIn("BIF Y/J", box.details.text())
        self.assertIn("#ffd700", box.styleSheet())


if __name__ == "__main__":
    unittest.main()
