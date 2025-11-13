import csv
import os
from typing import Dict, List, Tuple

from typing import TypedDict


class TelemetryRow(TypedDict):
    t: int
    pressure: float
    temperature: float
    vibration: float
    is_anomaly: int


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "telemetry_small.csv")


def load_telemetry(path: str = DATA_PATH) -> Tuple[List[int], List[List[float]], List[int]]:
    """
    Load telemetry CSV into (timestamps, features, labels).
    Features order: [pressure, temperature, vibration]
    Labels: 0/1 point anomaly labels.
    """
    ts: List[int] = []
    X: List[List[float]] = []
    y: List[int] = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ts.append(int(r["t"]))
            X.append([float(r["pressure"]), float(r["temperature"]), float(r["vibration"])])
            y.append(int(r["is_anomaly"]))

    return ts, X, y
