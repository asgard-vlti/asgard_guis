import os
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import call, mock_open, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from asgard_guis.cmd_scripts import plot_temps, shutdown_mimir, startup_mimir
from asgard_guis.cmd_scripts import temperature_watchdog
from asgard_guis.pdu_config import (
    PDUConfigError,
    load_pdu_config,
    parse_pdu_config,
)


CONFIG_TEXT = """
[pdu]
ip = "192.168.100.11"

[outlets]
"7" = ["Mimir", "Controller"]
"2" = ["Mimir"]
"4" = ["MDS"]
"9" = ["C-RED"]
"""


class PDUConfigTests(unittest.TestCase):
    def test_packaged_configuration_and_multi_device_lookup(self):
        packaged = load_pdu_config()
        self.assertEqual(packaged.ip, "192.168.100.11")
        self.assertEqual(packaged.outlets_for("Mimir"), (1, 8))
        self.assertEqual(packaged.devices_for(6), ("C-RED",))

        config = parse_pdu_config(CONFIG_TEXT)
        self.assertEqual(config.outlets_for("Mimir"), (2, 7))
        self.assertEqual(config.outlets_for("Controller"), (7,))
        self.assertEqual(config.devices_for(7), ("Mimir", "Controller"))
        with self.assertRaisesRegex(PDUConfigError, "No PDU outlets"):
            config.outlets_for("unknown")

    def test_invalid_configuration_is_rejected(self):
        cases = (
            "[pdu]\nip = 'not-an-ip'\n[outlets]\n'1' = ['Mimir']",
            "[pdu]\nip = '192.168.100.11'",
            "[pdu]\nip = '192.168.100.11'\n[outlets]\n'0' = ['Mimir']",
            "[pdu]\nip = '192.168.100.11'\n[outlets]\n'1' = []",
            "[pdu]\nip = '192.168.100.11'\n[outlets]\n'1' = ['Mimir', 'Mimir']",
            "[pdu]\nip = '192.168.100.11'\n[outlets]\n'1' = ['Mimir'",
        )
        for contents in cases:
            with self.subTest(contents=contents), self.assertRaises(PDUConfigError):
                parse_pdu_config(contents)

    def test_missing_packaged_file_has_clear_error(self):
        with patch(
            "asgard_guis.pdu_config.resources.read_text",
            side_effect=FileNotFoundError("pdu.toml"),
        ):
            with self.assertRaisesRegex(PDUConfigError, "Could not read packaged"):
                load_pdu_config()


class PDUCommandTests(unittest.TestCase):
    def setUp(self):
        self.config = parse_pdu_config(CONFIG_TEXT)

    def test_startup_uses_configured_mimir_outlets(self):
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(startup_mimir, "load_pdu_config", return_value=self.config)
            )
            pdu_class = stack.enter_context(patch("asgard_guis.PDU.AtenEcoPDU"))
            stack.enter_context(patch.object(startup_mimir.time, "sleep"))
            stack.enter_context(redirect_stdout(StringIO()))
            pdu_class.return_value.read_outlet_status.return_value = "on"
            startup_mimir.main()

        pdu_class.assert_called_once_with("192.168.100.11")
        self.assertEqual(
            pdu_class.return_value.switch_outlet_status.call_args_list,
            [call(2, "on"), call(7, "on")],
        )

    def test_shutdown_uses_configured_mimir_outlets(self):
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(shutdown_mimir, "load_pdu_config", return_value=self.config)
            )
            pdu_class = stack.enter_context(patch("asgard_guis.PDU.AtenEcoPDU"))
            ssh = stack.enter_context(
                patch.object(shutdown_mimir.subprocess, "call", return_value=0)
            )
            stack.enter_context(patch.object(shutdown_mimir.time, "sleep"))
            stack.enter_context(redirect_stdout(StringIO()))
            pdu_class.return_value.read_outlet_status.return_value = "off"
            shutdown_mimir.main()

        pdu_class.assert_called_once_with("192.168.100.11")
        ssh.assert_called_once()
        self.assertEqual(
            pdu_class.return_value.switch_outlet_status.call_args_list,
            [call(2, "off"), call(7, "off")],
        )

    def test_invalid_config_stops_before_power_or_remote_shutdown(self):
        for module in (startup_mimir, shutdown_mimir):
            with self.subTest(module=module.__name__):
                with ExitStack() as stack:
                    stack.enter_context(
                        patch.object(
                            module,
                            "load_pdu_config",
                            side_effect=PDUConfigError("bad config"),
                        )
                    )
                    pdu_class = stack.enter_context(
                        patch("asgard_guis.PDU.AtenEcoPDU")
                    )
                    ssh = stack.enter_context(
                        patch.object(shutdown_mimir.subprocess, "call")
                    )
                    stack.enter_context(
                        self.assertRaisesRegex(PDUConfigError, "bad config")
                    )
                    module.main()
                pdu_class.assert_not_called()
                ssh.assert_not_called()

    def test_watchdog_uses_configured_outlets_and_labels(self):
        self.assertEqual(temperature_watchdog.monitored_outlets(self.config), (4, 9))
        shared = parse_pdu_config(
            '[pdu]\nip = "192.168.100.11"\n[outlets]\n"4" = ["MDS", "C-RED"]'
        )
        self.assertEqual(temperature_watchdog.monitored_outlets(shared), (4,))
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    temperature_watchdog.asgard_guis.utils,
                    "open_socket_connection",
                )
            )
            stack.enter_context(
                patch.object(
                    temperature_watchdog,
                    "load_pdu_config",
                    return_value=self.config,
                )
            )
            pdu_class = stack.enter_context(patch("asgard_guis.PDU.AtenEcoPDU"))
            pdu_class.return_value.read_power_value.return_value = "1.23"
            stack.enter_context(
                patch.object(
                    temperature_watchdog.asgard_guis.utils,
                    "send_and_get_response",
                    side_effect=["[Lower T]", "[20.0]"],
                )
            )
            opened = stack.enter_context(patch("builtins.open", mock_open()))
            stack.enter_context(
                patch.object(temperature_watchdog.time, "time", side_effect=[0, 0, 3601])
            )
            stack.enter_context(patch.object(temperature_watchdog.time, "sleep"))
            stack.enter_context(redirect_stdout(StringIO()))
            stack.enter_context(
                patch.object(sys, "argv", ["temperature-watchdog", "--duration", "1"])
            )
            temperature_watchdog.main()

        pdu_class.assert_called_once_with("192.168.100.11")
        self.assertEqual(
            pdu_class.return_value.read_power_value.call_args_list,
            [call("olt", 4, "curr", "simple"), call("olt", 9, "curr", "simple")],
        )
        header = opened().write.call_args_list[0].args[0]
        self.assertIn("outlet4 (MDS)", header)
        self.assertIn("outlet9 (C-RED)", header)

    def test_invalid_config_stops_watchdog_before_connecting(self):
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    temperature_watchdog,
                    "load_pdu_config",
                    side_effect=PDUConfigError("bad config"),
                )
            )
            pdu_class = stack.enter_context(patch("asgard_guis.PDU.AtenEcoPDU"))
            mds = stack.enter_context(
                patch.object(
                    temperature_watchdog.asgard_guis.utils, "open_socket_connection"
                )
            )
            stack.enter_context(patch.object(sys, "argv", ["temperature-watchdog"]))
            stack.enter_context(self.assertRaisesRegex(PDUConfigError, "bad config"))
            temperature_watchdog.main()
        pdu_class.assert_not_called()
        mds.assert_not_called()


class TemperaturePlotTests(unittest.TestCase):
    def test_existing_and_new_outlet_headers_are_plotted(self):
        app = plot_temps.QtWidgets.QApplication.instance() or plot_temps.QtWidgets.QApplication(
            []
        )
        self.__class__._qt_app = app
        for names in (
            ["Lower T", "outlet5 (MDS)", "outlet6 (C RED)"],
            ["Lower T", "outlet5 (MDS / Controller)", "outlet6 (C-RED)"],
        ):
            with self.subTest(names=names):
                with tempfile.TemporaryDirectory() as folder:
                    log_path = Path(folder) / "temperature.log"
                    log_path.write_text(
                        "Time," + ",".join(names) + "\n"
                        "2026-10-14 12:00:00,20,1,2\n"
                    )
                    widget = plot_temps.TempPlotWidget(60, None, str(log_path), 20)
                    widget.timer.stop()
                    self.assertEqual(
                        [line.get_label() for line in widget.axes[-1].lines], names[1:]
                    )
                    with log_path.open("a") as log:
                        log.write("2026-10-14 12:01:00,21,1.1,2.1\n")
                    widget.update_plot()
                    self.assertEqual(len(widget.times), 2)
                    widget.close()


if __name__ == "__main__":
    unittest.main()
