import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets

from asgard_guis.cmd_scripts import baldr_faint_alignment


class BaldrFaintAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_spherical_disables_alignment_for_every_beam(self):
        labels = {
            "1": "Align CRed box",
            "2": "Align BOTX",
            "3": "Align BOTX",
            "4": "Align BOTX",
        }
        with mock.patch.object(baldr_faint_alignment.subprocess, "Popen") as launch:
            window = baldr_faint_alignment.BaldrFaintAlignmentGUI()
            try:
                for beam, label in labels.items():
                    window.beam_combo.setCurrentText(beam)
                    self.assertEqual(window.align_btn.text(), label)
                    self.assertTrue(window.align_btn.isEnabled())

                window.beam_combo.setCurrentText("1")
                window.spherical_checkbox.setChecked(True)
                for beam, label in labels.items():
                    window.beam_combo.setCurrentText(beam)
                    self.assertEqual(window.align_btn.text(), label)
                    self.assertFalse(window.align_btn.isEnabled())
                    window.align_btn.click()
                launch.assert_called_once_with(
                    ["/home/asg/.conda/envs/asgard/bin/dm-zernike", "1", "11", "0.1"]
                )

                window.spherical_checkbox.setChecked(False)
                for beam, label in labels.items():
                    window.beam_combo.setCurrentText(beam)
                    self.assertEqual(window.align_btn.text(), label)
                    self.assertTrue(window.align_btn.isEnabled())
                self.assertEqual(
                    launch.call_args_list[-1],
                    mock.call(
                        ["/home/asg/.conda/envs/asgard/bin/dm-zernike", "4", "11", "0.0"]
                    ),
                )
            finally:
                window.close()


if __name__ == "__main__":
    unittest.main()
