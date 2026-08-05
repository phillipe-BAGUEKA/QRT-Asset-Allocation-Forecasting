"""Controlled inference doubles shared by API tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import EXPECTED_FEATURE_COLUMNS


class FakePipeline:
    """Minimal fitted classifier that records inference DataFrames."""

    def __init__(
        self,
        *,
        classes: tuple[int, ...] = (1, 0),
        probabilities: tuple[float, ...] = (0.8, 0.2),
    ) -> None:
        self.classes_ = np.asarray(classes)
        self.probabilities = np.asarray([probabilities], dtype=float)
        self.n_features_in_ = len(EXPECTED_FEATURE_COLUMNS)
        self.feature_names_in_ = np.asarray(EXPECTED_FEATURE_COLUMNS)
        self.received_frames: list[pd.DataFrame] = []

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        self.received_frames.append(frame.copy())
        return self.probabilities.copy()
