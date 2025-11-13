import json
import os
from typing import Any, Dict, List

from data_utils import PROJECT_ROOT, load_telemetry
from detectors import zscore_anomaly, ewma_anomaly, isolation_forest_flags
from metrics import precision_recall_f1

OUT_PATH = os.path.join(PROJECT_ROOT, "monitoring_results.json")


def main() -> None:
    # Load telemetry
    ts, X, y_true = load_telemetry()

    # Univariate series for simple detectors
    pressure = [row[0] for row in X]
    vibration = [row[2] for row in X]

    # Run detectors
    preds = {
        "zscore_pressure": zscore_anomaly(pressure, threshold=3.0),
        "ewma_vibration": ewma_anomaly(vibration, alpha=0.3, k_sigma=3.0),
        "iforest_all": isolation_forest_flags(X, contamination=0.15, random_state=42),
    }

    # Compute metrics
    summary: Dict[str, Any] = {"per_detector": {}, "best_by_f1": None}
    best_name = None
    best_f1 = -1.0

    for name, pred in preds.items():
        p, r, f1 = precision_recall_f1(pred, y_true)
        summary["per_detector"][name] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
        }
        if f1 > best_f1:
            best_f1 = f1
            best_name = name

    summary["best_by_f1"] = {"detector": best_name, "f1": round(best_f1, 4)}

    # Print short console summary
    print("=== Rocket health monitoring (point anomalies) ===")
    for name, m in summary["per_detector"].items():
        print(f"{name:18s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
    print(f"\nBest by F1: {summary['best_by_f1']['detector']} (F1={summary['best_by_f1']['f1']:.3f})")

    # Save detailed JSON
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": summary,
                "timestamps": ts,
                "labels_true": y_true,
                "predictions": preds,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nDetailed results saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
