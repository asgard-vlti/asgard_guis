import contextlib
import importlib.util
import io
import os
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from asgard_guis.cmd_scripts import ob_creator


class ObCreatorTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.directory = Path(temporary_directory.name)
        original_directory = Path.cwd()
        os.chdir(self.directory)
        self.addCleanup(os.chdir, original_directory)

    def test_writes_to_run_directory(self):
        template = self.directory / "template.obd"
        template.write_text(
            "Header\n__TARGET_BLOCK_START__\n__NAME__ __DIT__ __GAIN__\n"
            "__TARGET_BLOCK_END__\nFooter\n",
            encoding="utf-8",
        )
        run_directory = self.directory / "run"
        run_directory.mkdir()
        os.chdir(run_directory)

        arguments = [
            "ob-creator",
            "HD123",
            "--mode",
            "ut",
            "--label",
            "../night",
            "--template",
            str(template),
        ]
        with patch.object(sys, "argv", arguments), patch.object(
            ob_creator,
            "extract_simbad_data",
            return_value={"NAME": "HD123", "K_FLOAT": 6.0},
        ), contextlib.redirect_stdout(io.StringIO()):
            ob_creator.main()

        output = run_directory / ".._night__HD123_UT.obd"
        self.assertEqual(
            output.read_text(encoding="utf-8"), "Header\nHD123 0.0004 30\nFooter\n"
        )

    def test_uses_included_template_from_another_directory(self):
        run_directory = self.directory / "run"
        run_directory.mkdir()
        os.chdir(run_directory)
        target = {
            "NAME": "HD123",
            "K_FLOAT": 6.0,
            "RA": "123456.789",
            "DEC": "+123456.789",
            "PMA": "0.",
            "PMD": "0.",
            "H": "6.0",
            "K": "6.0",
            "PARALLAX": "0.",
            "G": "6.0",
        }
        with patch.object(sys, "argv", ["ob-creator", "HD123", "--mode", "ut"]), patch.object(
            ob_creator, "extract_simbad_data", return_value=target
        ), contextlib.redirect_stdout(io.StringIO()):
            ob_creator.main()

        output = run_directory / "HD123_UT.obd"
        content = output.read_text(encoding="utf-8")
        self.assertIn('TEL.TARG.NAME           "HD123";', content)
        self.assertNotIn("__TARGET_BLOCK_START__", content)
        self.assertNotIn("__DIT__", content)

    def test_checks_template_before_query(self):
        arguments = ["ob-creator", "HD123", "--mode", "ut", "--template", "missing.obd"]
        with patch.object(sys, "argv", arguments), patch.object(
            ob_creator, "extract_simbad_data", side_effect=AssertionError("Query started")
        ), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                ob_creator.main()

        self.assertEqual(error.exception.code, 2)

    def test_lookup_failure_does_not_write_partial_ob(self):
        stderr = io.StringIO()
        with patch.object(
            sys, "argv", ["ob-creator", "HD123", "missing", "--mode", "ut"]
        ), patch.object(
            ob_creator,
            "extract_simbad_data",
            side_effect=[
                {"NAME": "HD123", "K_FLOAT": 6.0},
                ob_creator.SimbadLookupError("Target 'missing' was not found in SIMBAD."),
            ],
        ), patch.object(ob_creator, "create_ob") as create_ob, contextlib.redirect_stderr(
            stderr
        ):
            with self.assertRaises(SystemExit) as error:
                ob_creator.main()

        self.assertEqual(error.exception.code, 1)
        self.assertIn("Target 'missing' was not found", stderr.getvalue())
        create_ob.assert_not_called()
        self.assertEqual(list(self.directory.iterdir()), [])


@unittest.skipUnless(importlib.util.find_spec("astroquery"), "astroquery is unavailable")
@unittest.skipUnless(hasattr(signal, "setitimer"), "Unix interval timers are required")
class SimbadLookupTests(unittest.TestCase):
    def setUp(self):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        stack.enter_context(contextlib.redirect_stdout(self.stdout))
        stack.enter_context(contextlib.redirect_stderr(self.stderr))
        self.result = [{"ra": 15.0, "dec": -30.0}]
        self.primary = MagicMock()
        self.mirror = MagicMock()
        self.primary.query_object.return_value = self.result
        self.mirror.query_object.return_value = self.result
        self.clients = stack.enter_context(
            patch("astroquery.simbad.Simbad", side_effect=[self.primary, self.mirror])
        )
        self.previous_handler = signal.getsignal(signal.SIGALRM)
        self.previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def assert_deadline_cleared(self):
        self.assertEqual(signal.getsignal(signal.SIGALRM), self.previous_handler)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), self.previous_timer)

    def test_primary_success_does_not_query_mirror(self):
        self.assertIs(ob_creator._query_simbad("Vega"), self.result)

        self.clients.assert_called_once_with()
        self.assertEqual(self.primary.server, "simbad.cds.unistra.fr")
        self.primary.add_votable_fields.assert_called_once_with(
            "pmra", "pmdec", "V", "G", "J", "H", "K", "plx_value", "rvz_radvel"
        )
        self.primary.query_object.assert_called_once_with("Vega")
        self.mirror.query_object.assert_not_called()
        self.assert_deadline_cleared()

    def test_setup_network_failure_tries_fresh_harvard_client(self):
        from requests.exceptions import ConnectionError

        self.primary.add_votable_fields.side_effect = ConnectionError("primary offline")

        self.assertIs(ob_creator._query_simbad("Vega"), self.result)

        self.assertEqual(self.clients.call_count, 2)
        self.assertEqual(self.mirror.server, "simbad.harvard.edu")
        self.primary.query_object.assert_not_called()
        self.mirror.add_votable_fields.assert_called_once()
        self.mirror.query_object.assert_called_once_with("Vega")
        self.assertIn("simbad.cds.unistra.fr: primary offline", self.stderr.getvalue())
        self.assert_deadline_cleared()

    def test_query_service_failure_tries_harvard(self):
        from pyvo.dal.exceptions import DALServiceError

        self.primary.query_object.side_effect = DALServiceError("service unavailable")

        self.assertIs(ob_creator._query_simbad("Vega"), self.result)

        self.mirror.query_object.assert_called_once_with("Vega")
        self.assert_deadline_cleared()

    def test_deadline_covers_client_setup_and_query(self):
        for phase in ("constructor", "add_votable_fields", "query_object"):
            with self.subTest(phase=phase):
                primary = MagicMock()
                if phase != "constructor":
                    getattr(primary, phase).side_effect = lambda *args: time.sleep(1)
                clients = iter([primary, self.mirror])

                def create_client():
                    client = next(clients)
                    if phase == "constructor" and client is primary:
                        time.sleep(1)
                    return client

                with patch("astroquery.simbad.Simbad", side_effect=create_client), patch.object(
                    ob_creator, "SIMBAD_TIMEOUT", 0.05
                ):
                    started = time.monotonic()
                    self.assertIs(ob_creator._query_simbad("Vega"), self.result)
                    elapsed = time.monotonic() - started

                self.assertLess(elapsed, 0.8)
                self.assertIn("Timed out after 0.05 seconds", self.stderr.getvalue())
                self.assert_deadline_cleared()

    def test_both_failures_report_target_hosts_and_reasons(self):
        from requests.exceptions import ConnectionError

        self.primary.query_object.side_effect = ConnectionError("primary offline")
        self.mirror.query_object.side_effect = TimeoutError("mirror too slow")

        with self.assertRaises(ob_creator.SimbadLookupError) as error:
            ob_creator._query_simbad("Vega")

        message = str(error.exception)
        self.assertIn("Vega", message)
        self.assertIn("simbad.cds.unistra.fr: primary offline", message)
        self.assertIn("simbad.harvard.edu: mirror too slow", message)
        self.assert_deadline_cleared()

    def test_library_exception_handler_cannot_absorb_deadline(self):
        def retrying_library_call(*args):
            try:
                time.sleep(1)
            except Exception:
                time.sleep(1)

        self.primary.add_votable_fields.side_effect = retrying_library_call

        with patch.object(ob_creator, "SIMBAD_TIMEOUT", 0.05):
            started = time.monotonic()
            self.assertIs(ob_creator._query_simbad("Vega"), self.result)
            elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.8)
        self.assertEqual(self.clients.call_count, 2)
        self.assertIn("Timed out after 0.05 seconds", self.stderr.getvalue())
        self.assert_deadline_cleared()

    def test_both_stalled_servers_have_bounded_total_wait(self):
        self.primary.query_object.side_effect = lambda *args: time.sleep(1)
        self.mirror.query_object.side_effect = lambda *args: time.sleep(1)

        with patch.object(ob_creator, "SIMBAD_TIMEOUT", 0.05):
            started = time.monotonic()
            with self.assertRaises(ob_creator.SimbadLookupError):
                ob_creator._query_simbad("Vega")
            elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.8)
        self.assertEqual(self.clients.call_count, 2)
        self.assert_deadline_cleared()

    def test_empty_success_is_not_retried(self):
        for result in (None, []):
            with self.subTest(result=result):
                self.primary.query_object.return_value = result
                with self.assertRaisesRegex(ob_creator.SimbadLookupError, "not found"):
                    ob_creator._query_simbad("unknown target")
                self.clients.assert_called_once_with()
                self.clients.reset_mock(side_effect=True)
                self.clients.side_effect = [self.primary, self.mirror]
                self.mirror.query_object.assert_not_called()
                self.assert_deadline_cleared()

    def test_programming_error_is_not_retried(self):
        self.primary.query_object.side_effect = ValueError("invalid local configuration")

        with self.assertRaisesRegex(ValueError, "invalid local configuration"):
            ob_creator._query_simbad("Vega")

        self.clients.assert_called_once_with()
        self.assert_deadline_cleared()

    def test_existing_alarm_is_restored_after_success_and_timeout(self):
        def previous_handler(signum, frame):
            raise AssertionError("Previous alarm should not expire during the test")

        for expire in (False, True):
            with self.subTest(expire=expire):
                signal.signal(signal.SIGALRM, previous_handler)
                signal.setitimer(signal.ITIMER_REAL, 5, 5)
                try:
                    if expire:
                        with self.assertRaises(TimeoutError):
                            with ob_creator._simbad_deadline(0.05):
                                time.sleep(1)
                    else:
                        with ob_creator._simbad_deadline(0.05):
                            pass

                    self.assertIs(signal.getsignal(signal.SIGALRM), previous_handler)
                    remaining, interval = signal.getitimer(signal.ITIMER_REAL)
                    self.assertGreater(remaining, 4)
                    self.assertLessEqual(remaining, 5)
                    self.assertEqual(interval, 5)
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, self.previous_handler)
                    signal.setitimer(signal.ITIMER_REAL, *self.previous_timer)

    def test_extraction_keeps_units_formatting_and_missing_value_defaults(self):
        import numpy as onp

        self.primary.query_object.return_value = [{
            "ra": 15.0,
            "dec": -30.0,
            "pmra": 100.0,
            "pmdec": onp.ma.masked,
            "H": onp.nan,
            "K": 6.25,
            "plx_value": 50.0,
            "G": 5.55,
        }]

        self.assertEqual(ob_creator.extract_simbad_data("Vega"), {
            "NAME": "Vega",
            "RA": "010000.000",
            "DEC": "-300000.000",
            "PMA": "0.100",
            "PMD": "0.",
            "H": "99.",
            "K": "6.25",
            "K_FLOAT": 6.25,
            "PARALLAX": "0.050",
            "G": "5.55",
        })


if __name__ == "__main__":
    unittest.main()
