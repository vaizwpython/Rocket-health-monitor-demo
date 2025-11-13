import unittest

from detectors import zscore_anomaly, ewma_anomaly, isolation_forest_flags


class TestDetectors(unittest.TestCase):
    def test_zscore_simple(self):
        s = [0, 0, 0, 10]  # последний — выброс
        pred = zscore_anomaly(s, threshold=3.0)
        self.assertEqual(len(pred), len(s))
        self.assertEqual(pred[-1], 1)

    def test_ewma_simple(self):
        s = [1, 1, 1, 5]
        pred = ewma_anomaly(s, alpha=0.3, k_sigma=2.5)
        self.assertEqual(len(pred), len(s))
        self.assertEqual(pred[-1], 1)

    def test_iforest_shape(self):
        X = [[0.0], [0.1], [0.05], [5.0]]  # простой набор
        pred = isolation_forest_flags(X, contamination=0.2, random_state=0)
        self.assertEqual(len(pred), len(X))
        # хотя бы одна аномалия помечена
        self.assertTrue(any(v == 1 for v in pred))


if __name__ == "__main__":
    unittest.main()
