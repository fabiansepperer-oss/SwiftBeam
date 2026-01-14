#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


@dataclass
class MetricsResult:
    acc: float
    f1: float
    cm: np.ndarray


def compute_classification_metrics(y_true, y_pred) -> MetricsResult:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="binary"))
    cm = confusion_matrix(y_true, y_pred)
    return MetricsResult(acc=acc, f1=f1, cm=cm)


def metrics_to_dict(m: MetricsResult) -> Dict:
    return {
        "acc": m.acc,
        "f1": m.f1,
        "confusion_matrix": m.cm.tolist(),
    }
