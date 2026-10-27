import unittest

import numpy as np

from src.train_faithfulness_roberta_lora import (
    metrics_at_threshold,
    risk_coverage,
    select_unfaithful_threshold,
)


class FaithfulnessEvaluationTests(unittest.TestCase):
    def test_selects_validation_threshold_for_unfaithful_f1(self):
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.8, 0.6, 0.2, 0.1])

        threshold, metrics = select_unfaithful_threshold(labels, probabilities)

        self.assertAlmostEqual(threshold, 0.6)
        self.assertEqual(metrics["unfaithful_recall"], 1.0)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_risk_coverage_orders_by_faithful_probability(self):
        labels = np.array([0, 0, 1, 1])
        faithful_probabilities = np.array([0.1, 0.2, 0.8, 0.9])

        curve = risk_coverage(labels, faithful_probabilities)

        self.assertEqual(curve["coverage"], [0.25, 0.5, 0.75, 1.0])
        self.assertEqual(curve["unfaithful_risk"], [0.0, 0.0, 1 / 3, 0.5])
        self.assertGreater(curve["aurc"], 0.0)

    def test_zero_unfaithful_predictions_have_zero_recall(self):
        metrics = metrics_at_threshold(
            np.array([0, 0, 1, 1]),
            np.array([0.2, 0.1, 0.05, 0.01]),
            threshold=0.5,
        )

        self.assertEqual(metrics["unfaithful_recall"], 0.0)
        self.assertEqual(metrics["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
