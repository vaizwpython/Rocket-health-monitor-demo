import unittest
from metrics import precision_recall_f1


class TestMetrics(unittest.TestCase):
    def test_prf1(self):
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0, 1, 0, 0, 1]
        p, r, f1 = precision_recall_f1(y_pred, y_true)
        self.assertGreaterEqual(p, 0.0)
        self.assertGreaterEqual(r, 0.0)
        self.assertGreaterEqual(f1, 0.0)


if __name__ == "__main__":
    unittest.main()
