import unittest

import numpy as onp

from asgard_guis.cmd_scripts.Heimdallr_RTC_performance import (
    telescope_gd_snr_ranks,
)


class TelescopeGdSnrRankingTests(unittest.TestCase):
    def test_baselines_map_to_their_telescopes(self):
        history = onp.array([[10, 20, 30, 40, 50, 60]], dtype=float)

        self.assertEqual(telescope_gd_snr_ranks(history, 1, 200), [4, 3, 2, 1])

    def test_uses_only_the_latest_five_second_sample_window(self):
        history = onp.array(
            [
                [100, 90, 80, 70, 60, 50],
                [10, 20, 30, 40, 50, 60],
                [10, 20, 30, 40, 50, 60],
            ],
            dtype=float,
        )

        self.assertEqual(telescope_gd_snr_ranks(history, 3, 2500), [4, 3, 2, 1])

    def test_ignores_unfilled_rows_at_startup(self):
        history = onp.zeros((100, 6))
        history[-1] = [10, 20, 30, 40, 50, 60]

        self.assertEqual(telescope_gd_snr_ranks(history, 0, 200), [None] * 4)
        self.assertEqual(telescope_gd_snr_ranks(history, 1, 200), [4, 3, 2, 1])

    def test_uses_all_available_rows_if_buffer_is_shorter_than_five_seconds(self):
        history = onp.array(
            [
                [100, 90, 80, 70, 60, 50],
                [10, 20, 30, 40, 50, 60],
            ],
            dtype=float,
        )

        self.assertEqual(telescope_gd_snr_ranks(history, 2, 200), [1] * 4)

    def test_equal_averages_share_a_rank(self):
        history = onp.array([[0, 0, 10, 0, 10, 0]], dtype=float)

        self.assertEqual(telescope_gd_snr_ranks(history, 1, 200), [2, 2, 4, 1])

    def test_nonfinite_baselines_do_not_contribute(self):
        history = onp.array([[onp.nan, onp.inf, -onp.inf, 2, 3, 4]])

        self.assertEqual(telescope_gd_snr_ranks(history, 1, 200), [None, 3, 2, 1])
