import contextlib
import csv
import io
import os
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import matplotlib
from PyQt5 import QtWidgets

from asgard_guis.cmd_scripts import plot_temps, temperature_watchdog


class TemperatureTimestampTests(unittest.TestCase):
    def test_legacy_utc_and_iso_offsets_represent_the_same_instant(self):
        expected = datetime(2026, 10, 12, 0, 0, 0, tzinfo=timezone.utc)
        for value in (
            "2026-10-12 00:00:00",
            "2026-10-12T00:00:00Z",
            "2026-10-12T00:00:00+00:00",
            "2026-10-11T21:00:00-03:00",
        ):
            with self.subTest(value=value):
                self.assertEqual(plot_temps.parse_timestamp(value), expected)

    def test_fractional_seconds_survive_parsing(self):
        timestamp = plot_temps.parse_timestamp("2026-10-12T00:00:00.123Z")
        self.assertEqual(timestamp.microsecond, 123000)
        self.assertEqual(timestamp.tzinfo, timezone.utc)

    @unittest.skipUnless(hasattr(time, "tzset"), "Requires POSIX timezone support")
    def test_watchdog_writes_utc_on_both_hosts_without_changing_csv(self):
        instant = datetime(2026, 10, 12, 0, 0, 0, 123000, tzinfo=timezone.utc)

        class FixedDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime.fromtimestamp(instant.timestamp(), tz=tz)

        try:
            for zone in ("UTC", "America/Santiago"):
                with self.subTest(zone=zone), tempfile.TemporaryDirectory() as directory:
                    with mock.patch.dict(os.environ, {"TZ": zone}):
                        time.tzset()
                        log_path = Path(directory) / "tempWD_20261012T000000Z.log"
                        with contextlib.ExitStack() as stack:
                            stack.enter_context(mock.patch.object(temperature_watchdog, "datetime", FixedDateTime))
                            expanduser = stack.enter_context(mock.patch.object(
                                temperature_watchdog.os.path, "expanduser", return_value=str(log_path)
                            ))
                            stack.enter_context(mock.patch("sys.argv", ["temperature-watchdog", "--duration", "0.001"]))
                            stack.enter_context(mock.patch.object(temperature_watchdog.time, "time", side_effect=[0, 0, 4]))
                            stack.enter_context(mock.patch.object(temperature_watchdog.time, "sleep"))
                            pdu = stack.enter_context(mock.patch.object(temperature_watchdog.asgard_guis.PDU, "AtenEcoPDU"))
                            pdu.return_value.read_power_value.return_value = 1.25
                            stack.enter_context(mock.patch.object(temperature_watchdog.asgard_guis.utils, "open_socket_connection"))
                            stack.enter_context(mock.patch.object(
                                temperature_watchdog.asgard_guis.utils, "send_and_get_response",
                                side_effect=["[Lower T]", "[512]"],
                            ))
                            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                            temperature_watchdog.main()
                        self.assertTrue(expanduser.call_args.args[0].endswith(log_path.name))
                        with log_path.open() as file:
                            rows = list(csv.reader(file))
                        self.assertEqual(rows, [
                            ["Time", "Lower T", "outlet5 (MDS)", "outlet6 (C RED)"],
                            ["2026-10-12T00:00:00.123Z", "512.00", "1.25", "1.25"],
                        ])
        finally:
            time.tzset()


class TemperaturePlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_initial_read_and_append_normalize_across_midnight(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "temperatures.log"
            path.write_text(
                "Time,Lower T,Lower m_pin_val,Lower integral,Lower k_prop,outlet5 (MDS)\n"
                "2026-10-11 23:59:55,512,100,1,1,1\n"
                "2026-10-12T00:00:00.123Z,513,100,1,1,1\n",
                encoding="utf-8",
            )
            with matplotlib.rc_context({"timezone": "America/Santiago"}):
                widget = plot_temps.TempPlotWidget(60, None, str(path), 5)
                try:
                    widget.timer.stop()
                    self.assertEqual(len(widget.times), 2)
                    with path.open("a", encoding="utf-8") as file:
                        file.write("2026-10-11T21:00:05.123-03:00,514,100,1,1,1\n")
                    widget.update_plot()
                    self.assertEqual(len(widget.times), 3)
                    self.assertTrue(all(value.tzinfo == timezone.utc for value in widget.times))
                    self.assertEqual(widget.times[-1], datetime(2026, 10, 12, 0, 0, 5, 123000, tzinfo=timezone.utc))
                    self.assertEqual((widget.times[-1] - widget.times[-2]).total_seconds(), 5)
                    self.assertEqual(widget.axes[-1].get_xlabel(), "Time (UTC)")
                    self.assertEqual(widget.axes[-1].xaxis.get_major_formatter()._tz, timezone.utc)
                    self.assertEqual(len(widget.probe_data), 3)
                finally:
                    widget.close()


if __name__ == "__main__":
    unittest.main()
