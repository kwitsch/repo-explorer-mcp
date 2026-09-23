import unittest

import numpy as np

from judge_train import metrics


class TestMetrics(unittest.TestCase):
    def test_auroc_perfect_inverted_ties(self):
        self.assertAlmostEqual(metrics.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1.0)
        self.assertAlmostEqual(metrics.auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]), 0.0)
        self.assertAlmostEqual(metrics.auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]), 0.5)
        self.assertTrue(np.isnan(metrics.auroc([0.1, 0.2], [1, 1])))

    def test_ece_hand_computed(self):
        # conf=0.7 for both; acc=(1+0)/2=0.5; ece=|0.7-0.5|=0.2
        self.assertAlmostEqual(metrics.ece([0.7, 0.7], [1, 0]), 0.2, places=6)

    def test_query_top1_tie_break(self):
        rows = [
            {"query_id": "q1", "label": "A", "candidate_rank": 2},
            {"query_id": "q1", "label": "B", "candidate_rank": 1},
        ]
        self.assertEqual(metrics.query_top1(rows, [0.5, 0.5]), 0.0)  # tie -> lower rank (B)
        self.assertEqual(metrics.query_top1(rows, [0.9, 0.5]), 1.0)  # A wins

    def test_none_rate(self):
        rows = [
            {"query_id": "q1", "label": "B", "candidate_rank": 1},
            {"query_id": "q1", "label": "B", "candidate_rank": 2},
            {"query_id": "q2", "label": "B", "candidate_rank": 1},
        ]
        self.assertEqual(metrics.none_rate(rows, [0.2, 0.3, 0.9], 50), 0.5)

    def test_select_metrics_empty_selection(self):
        m = metrics.select_metrics([0.1, 0.2], [1, 0], 90)
        self.assertEqual(m["selected"], 0)
        self.assertTrue(np.isnan(m["precision"]))
        self.assertEqual(m["recall"], 0.0)

    def test_pick_threshold_and_fallback(self):
        self.assertEqual(metrics.pick_threshold([0.95, 0.92, 0.1, 0.05], [1, 1, 0, 0]), 30)
        self.assertEqual(metrics.pick_threshold([0.95, 0.95, 0.95, 0.95], [1, 0, 1, 0]), 90)


if __name__ == "__main__":
    unittest.main()
