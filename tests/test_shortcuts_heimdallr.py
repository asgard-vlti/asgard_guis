import os
import json
import threading
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtWidgets

from asgard_guis.cmd_scripts import shortcuts


def _settings(servo=4, offload=2, kp=0.3, delay_line="rmn"):
    return {
        "servo_mode": servo,
        "offload_mode": offload,
        "kp": kp,
        "delay_line_type": delay_line,
    }


def _ddspc(configured_cutoff=100.0, active_cutoff=None, history=(30, 3), active_history=None):
    configured = {
        "reg_start": 1000.0,
        "reg_cutoff": configured_cutoff,
        "history": history[0],
        "future": history[1],
    }
    active = None
    if active_cutoff is not None:
        active_history = active_history or history
        active = {
            "reg_start": 1000.0,
            "reg_cutoff": active_cutoff,
            "history": active_history[0],
            "future": active_history[1],
        }
    return {"configured": configured, "active": active, "runtime": None}


class _FakeSocket:
    def __init__(self, replies, requests):
        self.replies = replies
        self.requests = requests
        self.command = None
        self.closed = False

    def setsockopt(self, *_args):
        pass

    def connect(self, _endpoint):
        pass

    def send_string(self, command):
        self.command = command
        self.requests.append(command)

    def recv_string(self):
        reply = self.replies[self.command]
        if isinstance(reply, Exception):
            raise reply
        return reply

    def close(self):
        self.closed = True


class _FakeContext:
    def __init__(self, replies):
        self.replies = replies
        self.requests = []
        self.sockets = []

    def socket(self, _kind):
        socket = _FakeSocket(self.replies, self.requests)
        self.sockets.append(socket)
        return socket


class HeimdallrWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_one_second_poll_reads_both_settings_and_ddspc(self):
        settings = _settings(servo=5, offload=0, kp=0.42)
        ddspc = _ddspc(active_cutoff=100.0)
        context = _FakeContext({
            "settings": json.dumps(settings),
            'ddspc "get"': json.dumps(ddspc),
        })
        worker = shortcuts.HeimdallrLoopWorker("mimir")
        updated = []
        worker.state_updated.connect(lambda s, d: updated.append((s, d)))

        self.assertEqual(worker._interval_seconds, 1.0)
        self.assertEqual(worker._poll_once(context), (settings, ddspc))
        self.assertEqual(context.requests, ["settings", 'ddspc "get"'])
        self.assertEqual(updated, [(settings, ddspc)])
        self.assertTrue(all(socket.closed for socket in context.sockets))

    def test_commands_read_back_once_and_timeout_is_not_retried(self):
        settings = _settings()
        ddspc = _ddspc()
        context = _FakeContext({
            'servo "off"': "OK",
            'offload "off"': "OK",
            "settings": json.dumps(settings),
            'ddspc "get"': json.dumps(ddspc),
        })
        worker = shortcuts.HeimdallrLoopWorker("mimir")
        finished = []
        worker.command_finished.connect(lambda label, ok, detail: finished.append((label, ok, detail)))
        worker._execute(context, ['servo "off"', 'offload "off"'], "All Off")
        self.assertEqual(
            context.requests,
            ['servo "off"', 'offload "off"', "settings", 'ddspc "get"'],
        )
        self.assertEqual(finished, [("All Off", True, "")])

        context = _FakeContext({
            'servo "ddspc"': TimeoutError("timed out"),
            "settings": json.dumps(settings),
            'ddspc "get"': json.dumps(ddspc),
        })
        worker._execute(context, ['servo "ddspc"'], "DDSPC")
        self.assertEqual(context.requests, ['servo "ddspc"', "settings", 'ddspc "get"'])
        self.assertEqual(finished[-1][0:2], ("DDSPC", False))
        self.assertIn("timed out", finished[-1][2])

    def test_ack_without_matching_readback_is_not_success(self):
        cases = [
            ('servo "ddspc"', _settings(servo=4), _ddspc(), "servo_mode"),
            ("gain 0.42", _settings(kp=0.25), _ddspc(), "Kp"),
            ('ddspc "set-reg-cutoff" 120', _settings(), _ddspc(configured_cutoff=100), "cutoff"),
            ('ddspc "set-history" [40,3]', _settings(), _ddspc(history=(30, 3)), "history"),
        ]
        for command, settings, ddspc, detail in cases:
            with self.subTest(command=command):
                context = _FakeContext({
                    command: "OK",
                    "settings": json.dumps(settings),
                    'ddspc "get"': json.dumps(ddspc),
                })
                worker = shortcuts.HeimdallrLoopWorker("mimir")
                finished = []
                worker.command_finished.connect(
                    lambda label, ok, message: finished.append((label, ok, message))
                )
                worker._execute(context, [command], "change")
                self.assertEqual(context.requests, [command, "settings", 'ddspc "get"'])
                self.assertEqual(finished[0][0:2], ("change", False))
                self.assertIn(detail, finished[0][2])

    def test_ack_with_failed_readback_is_not_success(self):
        context = _FakeContext({
            'servo "off"': "OK",
            "settings": TimeoutError("settings timed out"),
        })
        worker = shortcuts.HeimdallrLoopWorker("mimir")
        finished = []
        failed = []
        worker.command_finished.connect(lambda label, ok, detail: finished.append((label, ok, detail)))
        worker.state_failed.connect(failed.append)
        worker._execute(context, ['servo "off"'], "Servo Off")
        self.assertEqual(context.requests, ['servo "off"', "settings"])
        self.assertEqual(finished[0][0:2], ("Servo Off", False))
        self.assertIn("could not be verified", finished[0][2])
        self.assertEqual(failed, ["settings timed out"])

    def test_failed_readback_reports_stale_state(self):
        worker = shortcuts.HeimdallrLoopWorker("mimir")
        failures = []
        worker.state_failed.connect(failures.append)
        with mock.patch.object(worker, "_poll_once", side_effect=TimeoutError("no reply")):
            worker._refresh_state(object())
        self.assertEqual(failures, ["no reply"])

    def test_server_error_reply_is_a_failure(self):
        worker = shortcuts.HeimdallrLoopWorker("mimir")
        context = _FakeContext({'ddspc "set-history" [40,3]': "ERROR: servo must be off"})
        with self.assertRaisesRegex(RuntimeError, "servo must be off"):
            worker._request(context, 'ddspc "set-history" [40,3]', 500)
        self.assertEqual(context.requests, ['ddspc "set-history" [40,3]'])
        self.assertTrue(context.sockets[0].closed)

    def test_slow_polls_do_not_overlap_or_build_up(self):
        worker = shortcuts.HeimdallrLoopWorker("mimir", interval_seconds=0.05)
        starts = []
        lock = threading.Lock()
        in_poll = False
        overlaps = []

        def slow_refresh(_context):
            nonlocal in_poll
            with lock:
                if in_poll:
                    overlaps.append(True)
                in_poll = True
                starts.append(time.monotonic())
            time.sleep(0.08)
            with lock:
                in_poll = False

        with mock.patch.object(shortcuts.zmq, "Context", return_value=mock.Mock()), mock.patch.object(
            worker, "_refresh_state", side_effect=slow_refresh
        ):
            worker.start()
            try:
                deadline = time.monotonic() + 1.0
                while len(starts) < 3 and time.monotonic() < deadline:
                    time.sleep(0.01)
            finally:
                worker.stop()
                self.assertTrue(worker.wait(1000))

        self.assertGreaterEqual(len(starts), 3)
        self.assertEqual(overlaps, [])
        self.assertTrue(all(b - a >= 0.10 for a, b in zip(starts, starts[1:])))


class HeimdallrGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.blf_start = mock.patch.object(shortcuts.BLFPoller, "start")
        self.heim_start = mock.patch.object(shortcuts.HeimdallrLoopWorker, "start")
        self.blf_start.start()
        self.heim_start.start()
        self.window = shortcuts.ShortcutsGUI(debug=True)
        self.addCleanup(self.window.close)
        self.addCleanup(self.heim_start.stop)
        self.addCleanup(self.blf_start.stop)

    def test_mode_buttons_and_all_off_queue_exact_commands(self):
        self.window._on_heim_state(_settings(), _ddspc())
        with mock.patch.object(self.window, "_queue_heim_action") as queue:
            cases = [
                (self.window.servo_buttons["off"], ['servo "off"']),
                (self.window.servo_buttons["on"], ['servo "on"']),
                (self.window.servo_buttons["ddspc"], ['servo "ddspc"']),
                (self.window.offload_buttons["off"], ['offload "off"']),
                (self.window.offload_buttons["gd"], ['offload "gd"']),
                (self.window.offload_buttons["mod"], ['offload "mod"']),
                (self.window.all_off_btn, ['servo "off"', 'offload "off"']),
            ]
            for button, expected in cases:
                with self.subTest(button=button.text()):
                    queue.reset_mock()
                    button.click()
                    queue.assert_called_once()
                    self.assertEqual(queue.call_args.args[0], expected)

    def test_four_column_heimdallr_grid_has_compact_dl_row(self):
        for column, script_button in enumerate(
            (self.window.align_btn, self.window.tilts_btn, self.window.sky_mode_btn, self.window.lab_mode_btn)
        ):
            self.assertIs(self.window.heim_layout.itemAtPosition(0, column).widget(), script_button)
            left, right = self.window.jump_buttons[column + 1]
            self.assertLessEqual(left.width(), 36)
            self.assertLessEqual(right.width(), 36)

        self.window.show()
        self.app.processEvents()
        left_positions = [
            self.window.jump_buttons[dl][0].mapTo(self.window, QtCore.QPoint()).x()
            for dl in range(1, 5)
        ]
        gaps = [right - left for left, right in zip(left_positions, left_positions[1:])]
        self.assertLessEqual(max(gaps) - min(gaps), 4)
        self.assertEqual(self.window.jump_spin.geometry().center().y(), self.window.gd_spin.geometry().center().y())

    def test_reported_modes_control_highlight_and_history_gate(self):
        self.window._on_heim_state(_settings(servo=5, offload=0), _ddspc(active_cutoff=100.0))
        self.assertTrue(self.window.servo_buttons["ddspc"].isChecked())
        self.assertFalse(self.window.servo_buttons["off"].isChecked())
        self.assertFalse(self.window.set_history_btn.isEnabled())
        self.assertIn("DDSPC", self.window.servo_status_label.text())
        self.assertIn("Nested", self.window.offload_status_label.text())

        self.window._on_heim_state(_settings(servo=4, offload=4), _ddspc())
        self.assertTrue(self.window.servo_buttons["off"].isChecked())
        self.assertTrue(self.window.offload_buttons["mod"].isChecked())
        self.assertTrue(self.window.set_history_btn.isEnabled())

        self.window._on_heim_state(_settings(servo=1, offload=1, delay_line="off"), _ddspc())
        self.assertTrue(self.window.servo_buttons["on"].isChecked())
        self.assertTrue(self.window.offload_buttons["gd"].isChecked())
        self.assertFalse(self.window.offload_buttons["mod"].isEnabled())
        self.assertFalse(self.window.offload_buttons["gd"].isEnabled())

        self.window._on_heim_state(_settings(servo=0, offload=3), _ddspc())
        self.assertIn("Fight", self.window.servo_status_label.text())
        self.assertIn("Manual", self.window.offload_status_label.text())
        self.assertFalse(any(button.isChecked() for button in self.window.servo_buttons.values()))
        self.assertFalse(any(button.isChecked() for button in self.window.offload_buttons.values()))

    def test_gain_is_validated_and_read_back_without_losing_edit(self):
        self.window._on_heim_state(_settings(kp=0.25), _ddspc())
        self.assertIn("0.25", self.window.kp_status_label.text())
        with mock.patch.object(self.window, "_queue_heim_action") as queue:
            self.window.kp_edit.setText("nan")
            self.window.apply_kp_btn.click()
            queue.assert_not_called()

            self.window.kp_edit.setText("0.42")
            self.window.apply_kp_btn.click()
            queue.assert_called_once()
            self.assertEqual(queue.call_args.args[0], ["gain 0.42"])

        self.window.kp_edit.setText("0.56")
        self.window._on_heim_state(_settings(kp=0.42), _ddspc())
        self.assertEqual(self.window.kp_edit.text(), "0.56")
        self.assertIn("0.42", self.window.kp_status_label.text())

    def test_poll_preserves_unfinished_kp_and_cutoff_edits(self):
        self.window._on_heim_state(_settings(kp=0.25), _ddspc(configured_cutoff=50))
        self.window.kp_edit.setText("0.56")
        self.window.kp_edit.textEdited.emit("0.56")
        self.window.cutoff_edit.setText("75")
        self.window.cutoff_edit.textEdited.emit("75")

        self.window._on_heim_state(_settings(kp=0.4), _ddspc(configured_cutoff=60))
        self.assertEqual(self.window.kp_edit.text(), "0.56")
        self.assertEqual(self.window.cutoff_edit.text(), "75")
        self.assertIn("0.4", self.window.kp_status_label.text())
        self.assertIn("60", self.window.cutoff_status_label.text())

    def test_history_combo_syncs_configured_preset_and_preserves_user_choice(self):
        self.window._on_heim_state(_settings(servo=4), _ddspc(history=(40, 3)))
        self.assertEqual(self.window.history_combo.currentText(), "40/3")

        self.window.history_combo.setCurrentIndex(0)
        self.window._on_heim_state(_settings(servo=4), _ddspc(history=(30, 3)))
        self.assertEqual(self.window.history_combo.currentText(), "20/2")
        self.assertIn("30/3", self.window.history_status_label.text())

    def test_ddspc_tuning_stages_only_valid_cutoff_and_presets(self):
        self.window._on_heim_state(
            _settings(servo=4),
            _ddspc(configured_cutoff=50.0, active_cutoff=100.0, active_history=(30, 3)),
        )
        with mock.patch.object(self.window, "_queue_heim_action") as queue:
            self.window.cutoff_edit.setText("120")
            self.window.stage_cutoff_btn.click()
            queue.assert_called_once()
            command = queue.call_args.args[0][0]
            self.assertTrue(command.startswith('ddspc "set-reg-cutoff" '))
            self.assertEqual(float(command.split()[-1]), 120.0)

            queue.reset_mock()
            for invalid in ("0", "nan", "1001"):
                with self.subTest(invalid=invalid):
                    self.window.cutoff_edit.setText(invalid)
                    self.window.stage_cutoff_btn.click()
                    queue.assert_not_called()

            self.assertEqual(
                [self.window.history_combo.itemText(i) for i in range(self.window.history_combo.count())],
                ["20/2", "30/3", "40/3", "50/3"],
            )
            self.window.history_combo.setCurrentIndex(2)
            self.window.set_history_btn.click()
            queue.assert_called_once()
            self.assertEqual(queue.call_args.args[0], ['ddspc "set-history" [40,3]'])

        self.window._on_heim_state(
            _settings(servo=5),
            _ddspc(configured_cutoff=120.0, active_cutoff=100.0, history=(40, 3), active_history=(30, 3)),
        )
        self.assertIn("120", self.window.cutoff_status_label.text())
        self.assertIn("100", self.window.cutoff_status_label.text())
        self.assertIn("40/3", self.window.history_status_label.text())
        self.assertIn("30/3", self.window.history_status_label.text())
        self.assertFalse(self.window.set_history_btn.isEnabled())

    def test_failed_refresh_marks_feedback_stale_and_removes_history_permission(self):
        self.window._on_heim_state(_settings(servo=4), _ddspc())
        self.assertTrue(self.window.set_history_btn.isEnabled())
        self.window._on_heim_state_failed("timeout")
        self.assertFalse(self.window.set_history_btn.isEnabled())
        self.assertIn("Stale", self.window.heim_status_label.text())

    def test_return_during_an_action_does_not_leave_unsent_tuning_pending(self):
        self.window._on_heim_state(_settings(servo=4), _ddspc())
        self.window.kp_edit.setText("0.5")
        self.window.cutoff_edit.setText("75")
        self.window._heim_action_pending = True
        with mock.patch.object(self.window, "_queue_heim_action") as queue:
            self.window._apply_kp()
            self.window._stage_cutoff()
            self.window._set_history()

        queue.assert_not_called()
        self.assertIsNone(self.window._pending_kp)
        self.assertIsNone(self.window._pending_cutoff)
        self.assertIsNone(self.window._pending_history)


if __name__ == "__main__":
    unittest.main()
