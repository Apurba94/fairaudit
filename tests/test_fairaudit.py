import math
import unittest

import numpy as np

try:
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
except ImportError:
    confusion_matrix = None

from fairaudit import apply_thresholds, bootstrap, by_group, calibration_by_group, confusion, equalise_opportunity, fairness_metrics, intersections, model_card, permutation_test, rates, threshold_sweep, warnings


def biased(n=20000, seed=0, shift=-0.8):
    """Two groups with the same true risk, where the score is unfairly lower for group B."""
    rng = np.random.default_rng(seed)
    g = np.where(rng.random(n) < 0.5, "A", "B")
    risk = rng.normal(size=n)
    y = rng.random(n) < 1 / (1 + np.exp(-2 * risk))
    z = risk + rng.normal(0, 0.5, n) + np.where(g == "B", shift, 0)
    return y, 1 / (1 + np.exp(-2 * z)), g


class TestRates(unittest.TestCase):
    def test_hand_example(self):
        y = [1, 1, 1, 0, 0, 0, 0, 0]
        p = [1, 1, 0, 1, 0, 0, 0, 0]
        self.assertEqual(confusion(y, p), (2, 1, 4, 1))
        r = rates(y, p)
        self.assertAlmostEqual(r["tpr"], 2 / 3)
        self.assertAlmostEqual(r["fpr"], 1 / 5)
        self.assertAlmostEqual(r["ppv"], 2 / 3)
        self.assertAlmostEqual(r["selection_rate"], 3 / 8)
        self.assertAlmostEqual(r["accuracy"], 6 / 8)
        self.assertAlmostEqual(r["f1"], 2 / 3)

    def test_no_denominator_is_nan(self):
        r = rates([0, 0], [0, 0])
        self.assertTrue(math.isnan(r["tpr"]))
        self.assertTrue(math.isnan(r["ppv"]))
        self.assertEqual(r["fpr"], 0)

    @unittest.skipIf(confusion_matrix is None, "scikit-learn is not installed")
    def test_against_scikit_learn(self):
        rng = np.random.default_rng(1)
        for _ in range(100):
            y, p = rng.integers(0, 2, 200), rng.integers(0, 2, 200)
            r = rates(y, p)
            self.assertAlmostEqual(r["tpr"], recall_score(y, p))
            self.assertAlmostEqual(r["ppv"], precision_score(y, p))
            self.assertAlmostEqual(r["f1"], f1_score(y, p))
            tn, fp, fn, tp = confusion_matrix(y, p).ravel()
            self.assertEqual(confusion(y, p), (tp, fp, tn, fn))


class TestFairness(unittest.TestCase):
    def test_hand_gaps(self):
        y = [1, 1, 0, 0, 1, 1, 0, 0]
        p = [1, 1, 1, 0, 1, 0, 0, 0]
        g = list("AAAABBBB")
        m = fairness_metrics(y, p, g)
        self.assertAlmostEqual(m["demographic_parity_gap"], 0.5)
        self.assertAlmostEqual(m["disparate_impact_ratio"], 1 / 3)
        self.assertAlmostEqual(m["equal_opportunity_gap"], 0.5)
        self.assertAlmostEqual(m["false_positive_gap"], 0.5)
        self.assertAlmostEqual(m["equalized_odds_gap"], 0.5)
        self.assertAlmostEqual(m["predictive_parity_gap"], 1 / 3)

    def test_fair_model_has_small_gaps(self):
        y, s, g = biased(shift=0.0)
        m = fairness_metrics(y, s >= 0.5, g)
        self.assertLess(m["demographic_parity_gap"], 0.03)
        self.assertLess(m["equalized_odds_gap"], 0.03)

    def test_bias_is_found(self):
        y, s, g = biased()
        m = fairness_metrics(y, s >= 0.5, g)
        self.assertGreater(m["equal_opportunity_gap"], 0.2)
        self.assertLess(m["disparate_impact_ratio"], 0.8)

    def test_one_group_gives_nan(self):
        self.assertTrue(math.isnan(fairness_metrics([1, 0], [1, 0], ["a", "a"])["accuracy_gap"]))

    def test_three_groups(self):
        t = by_group([1, 0, 1, 0, 1, 0], [1, 0, 1, 1, 0, 0], list("aabbcc"))
        self.assertEqual(set(t), {"a", "b", "c"})

    def test_intersections_show_hidden_harm(self):
        rng = np.random.default_rng(2)
        n = 8000
        sex, age = rng.integers(0, 2, n), rng.integers(0, 2, n)
        y = rng.random(n) < 0.5
        harmed = (sex == 1) & (age == 1) | (sex == 0) & (age == 0)
        p = np.where(harmed, rng.random(n) < 0.3, rng.random(n) < 0.7) & y
        self.assertLess(fairness_metrics(y, p, sex)["equal_opportunity_gap"], 0.06)
        self.assertGreater(fairness_metrics(y, p, intersections(sex, age))["equal_opportunity_gap"], 0.3)

    def test_calibration(self):
        rng = np.random.default_rng(3)
        n = 40000
        g = rng.integers(0, 2, n)
        s = rng.random(n)
        y = rng.random(n) < s
        e = calibration_by_group(y, np.where(g == 1, s ** 2, s), g)
        self.assertLess(e["0"], 0.02)
        self.assertGreater(e["1"], 0.1)


class TestUncertainty(unittest.TestCase):
    def test_interval_contains_the_truth_and_narrows(self):
        y, s, g = biased(4000)
        p = s >= 0.5
        lo, hi = bootstrap(y, p, g, rounds=300)["equal_opportunity_gap"]
        v = fairness_metrics(y, p, g)["equal_opportunity_gap"]
        self.assertLess(lo, v)
        self.assertGreater(hi, v)
        y2, s2, g2 = biased(400)
        lo2, hi2 = bootstrap(y2, s2 >= 0.5, g2, rounds=300)["equal_opportunity_gap"]
        self.assertGreater(hi2 - lo2, hi - lo)

    def test_permutation_separates_real_from_chance(self):
        y, s, g = biased(2000)
        self.assertLess(permutation_test(y, s >= 0.5, g, rounds=200), 0.02)
        y, s, g = biased(2000, shift=0.0, seed=4)
        self.assertGreater(permutation_test(y, s >= 0.5, g, rounds=200), 0.02)


class TestThresholds(unittest.TestCase):
    def test_sweep(self):
        y, s, g = biased(2000)
        rows = threshold_sweep(y, s, g, 11)
        self.assertEqual(len(rows), 11)
        self.assertEqual(rows[0]["demographic_parity_gap"], 0)

    def test_group_thresholds_equalise_tpr(self):
        y, s, g = biased()
        before = fairness_metrics(y, s >= 0.5, g)["equal_opportunity_gap"]
        t = equalise_opportunity(y, s, g, 0.8)
        p = apply_thresholds(s, g, t)
        table = by_group(y, p, g)
        self.assertLess(fairness_metrics(y, p, g)["equal_opportunity_gap"], 0.01)
        self.assertAlmostEqual(table["A"]["tpr"], 0.8, delta=0.01)
        self.assertGreater(before, 0.2)

    def test_group_with_no_positives(self):
        self.assertEqual(equalise_opportunity([0, 0, 1], [0.1, 0.2, 0.9], ["a", "a", "b"], 0.5)["a"], 1.0)


class TestReporting(unittest.TestCase):
    def test_warnings(self):
        y, s, g = biased(2000)
        text = " ".join(warnings(y, s >= 0.5, g))
        self.assertIn("four-fifths", text)
        small = warnings([1, 0, 1], [1, 0, 1], ["a", "a", "b"])
        self.assertIn("only 1 people", " ".join(small))

    def test_base_rate_warning(self):
        y = [1] * 80 + [0] * 20 + [1] * 20 + [0] * 80
        g = ["a"] * 100 + ["b"] * 100
        self.assertTrue(any("Base rates" in w for w in warnings(y, y, g)))

    def test_model_card(self):
        y, s, g = biased(2000)
        card = model_card("Risk model", "Rank cases for review.", y, s >= 0.5, g, s, not_for=["Automatic decisions"], data="Simulated.", contact="a@example.org", rounds=50)
        for text in ("# Model card: Risk model", "Not intended for:", "| A |", "demographic parity gap", "Calibration by group", "four-fifths", "a@example.org"):
            self.assertIn(text, card)

    def test_card_says_no_flags_is_not_proof(self):
        y = [1, 0] * 100
        card = model_card("M", "Test.", y, y, ["a"] * 100 + ["b"] * 100, rounds=20)
        self.assertIn("does not show that the model is fair", card)


if __name__ == "__main__":
    unittest.main()
