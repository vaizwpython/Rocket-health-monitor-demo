from typing import Iterable, Tuple


def _counts(pred: Iterable[int], true: Iterable[int]) -> Tuple[int, int, int]:
    tp = fp = fn = 0
    for p, t in zip(pred, true):
        if p == 1 and t == 1:
            tp += 1
        elif p == 1 and t == 0:
            fp += 1
        elif p == 0 and t == 1:
            fn += 1
    return tp, fp, fn


def precision_recall_f1(pred: Iterable[int], true: Iterable[int]) -> Tuple[float, float, float]:
    """
    Point-level precision/recall/F1.
    """
    tp, fp, fn = _counts(pred, true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1
