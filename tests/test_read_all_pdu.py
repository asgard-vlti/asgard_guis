import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import call, patch

from asgard_guis.PDU import AtenEcoPDU
from asgard_guis.cmd_scripts import read_all_pdu
from asgard_guis.pdu_config import parse_pdu_config


CONFIG = """
[pdu]
ip = "192.168.100.11"

[outlets]
"7" = ["Mimir"]
"2" = ["Mimir"]
"9" = ["C-RED"]
"4" = ["MDS"]
"""


class ReadAllPDUTests(unittest.TestCase):
    def test_prints_only_configured_outlet_statuses_in_numeric_order(self):
        config = parse_pdu_config(CONFIG)
        output = StringIO()
        with patch.object(read_all_pdu, "load_pdu_config", return_value=config):
            with patch.object(read_all_pdu, "AtenEcoPDU") as pdu_class:
                pdu = pdu_class.return_value
                pdu.connect.return_value = True
                pdu.read_outlet_status.side_effect = ["on", "off", "on", "off"]
                with redirect_stdout(output):
                    result = read_all_pdu.main()

        self.assertEqual(result, 0)
        self.assertEqual(output.getvalue(), "2: on\n4: off\n7: on\n9: off\n")
        pdu_class.assert_called_once_with("192.168.100.11")
        pdu.connect.assert_called_once_with(verbose=False)
        self.assertEqual(
            pdu.read_outlet_status.call_args_list,
            [call(2), call(4), call(7), call(9)],
        )
        pdu.close.assert_called_once_with(verbose=False)

    def test_unreadable_status_is_unknown_and_still_closes(self):
        config = parse_pdu_config(CONFIG)
        output = StringIO()
        with patch.object(read_all_pdu, "load_pdu_config", return_value=config):
            with patch.object(read_all_pdu, "AtenEcoPDU") as pdu_class:
                pdu = pdu_class.return_value
                pdu.connect.return_value = True
                pdu.read_outlet_status.side_effect = ["on", "", "on", "off"]
                with redirect_stdout(output):
                    result = read_all_pdu.main()

        self.assertEqual(result, 1)
        self.assertEqual(output.getvalue(), "2: on\n4: unknown\n7: on\n9: off\n")
        pdu.close.assert_called_once_with(verbose=False)

    def test_failed_connection_does_not_read_outlets(self):
        config = parse_pdu_config(CONFIG)
        with patch.object(read_all_pdu, "load_pdu_config", return_value=config):
            with patch.object(read_all_pdu, "AtenEcoPDU") as pdu_class:
                pdu_class.return_value.connect.return_value = False
                self.assertEqual(read_all_pdu.main(), 1)

        pdu_class.return_value.read_outlet_status.assert_not_called()

    def test_quiet_client_has_no_connection_or_close_messages(self):
        output = StringIO()
        with patch("asgard_guis.PDU.telnetlib.Telnet") as telnet_class:
            telnet = telnet_class.return_value
            telnet.read_until.side_effect = [
                b"Login: ",
                b"Password: ",
                b"Logged in successfully",
                b">",
                b"",
            ]
            pdu = AtenEcoPDU("192.168.100.11")
            with redirect_stdout(output):
                self.assertTrue(pdu.connect(verbose=False))
                pdu.close(verbose=False)

        self.assertEqual(output.getvalue(), "")
        telnet.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
