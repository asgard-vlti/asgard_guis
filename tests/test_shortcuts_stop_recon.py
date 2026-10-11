import os
import signal
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets

from asgard_guis.cmd_scripts import shortcuts


class ReconProcessTests(unittest.TestCase):
    def test_matches_only_recon_entry_points(self):
        cases = [
            (["/home/asg/.conda/envs/asgard/bin/python", shortcuts.RECON_SCRIPT], True),
            (["/usr/bin/python3", "-m", "dcs.baldr_tt.save_recon"], True),
            (["/usr/bin/python3", "/opt/dcs/baldr_tt/save_recon.py"], True),
            (["/usr/bin/python3", "/opt/other/save_recon.py"], False),
            (["/bin/sh", "-c", "baldrtt-recon"], False),
            (["/usr/bin/python3", "-m", "something_else"], False),
        ]
        for argv, expected in cases:
            with self.subTest(argv=argv):
                self.assertEqual(shortcuts._is_recon_command(argv), expected)

    def test_discovers_all_recon_jobs_in_proc(self):
        with tempfile.TemporaryDirectory() as proc_root:
            for pid, argv in {
                810001: ["/usr/bin/python3", shortcuts.RECON_SCRIPT],
                810002: ["/usr/bin/python3", "-m", "dcs.baldr_tt.save_recon"],
                810003: ["/usr/bin/python3", "/opt/other.py"],
            }.items():
                path = Path(proc_root) / str(pid)
                path.mkdir()
                (path / "cmdline").write_bytes(b"\0".join(part.encode() for part in argv) + b"\0")

            self.assertEqual(shortcuts._recon_pids(proc_root), [810001, 810002])

    def test_force_stops_job_after_grace_period(self):
        events = []
        with mock.patch.object(shortcuts, "_recon_pids", return_value=[810001]), mock.patch.object(
            shortcuts, "_is_recon_pid", side_effect=[True, True, True, False]
        ), mock.patch.object(shortcuts.os, "kill", side_effect=lambda pid, sig: events.append((pid, sig))):
            self.assertTrue(shortcuts._stop_recon_processes(lambda _: None, grace_seconds=0))

        self.assertEqual(events, [(810001, signal.SIGTERM), (810001, signal.SIGKILL)])


class StopWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_partial_beam_failure_is_reported_after_all_attempts(self):
        beams = []

        def clear(_host, beam):
            beams.append(beam)
            if beam == 3:
                raise RuntimeError("offline")

        worker = shortcuts.StopReconWorker("mimir")
        finished = []
        worker.stop_finished.connect(finished.append)
        with mock.patch.object(shortcuts, "_stop_recon_processes", return_value=True), mock.patch.object(
            shortcuts, "_clear_baldr_beam", side_effect=clear
        ):
            worker.run()

        self.assertEqual(set(beams), {1, 2, 3, 4})
        self.assertEqual(finished, [False])

    def test_debug_has_no_side_effects(self):
        worker = shortcuts.StopReconWorker("mimir", debug=True)
        with mock.patch.object(shortcuts, "_stop_recon_processes") as stop, mock.patch.object(
            shortcuts, "_clear_baldr_beam"
        ) as clear:
            worker.run()
        stop.assert_not_called()
        clear.assert_not_called()

    def test_stale_completion_cannot_reload_reconstructor(self):
        with mock.patch.object(shortcuts.BLFPoller, "start"):
            window = shortcuts.ShortcutsGUI(debug=True)
        try:
            window._recon_generation = 2
            with mock.patch.object(window, "_send_baldr_all") as send:
                window._send_recon_command(1)
                window._on_recon_finished(True, 1)
                send.assert_not_called()
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
