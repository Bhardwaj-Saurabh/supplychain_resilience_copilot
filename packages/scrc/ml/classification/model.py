"""Stockout-risk classifier (ML serving, Layer 2).

XGBoost gradient boosting with **isotonic probability calibration** and **SHAP**
attribution. The model emits a typed ``StockoutRiskResult`` carrying the
calibrated probability, a confidence tier, and per-feature SHAP values — the
uncertainty and explainability flow through, never stripped (P2). Training is
seeded for reproducibility.

The plain-language brief is intentionally left empty here: turning SHAP values
into planner prose is the LLM agent's job (ADR-0001), not the model's.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from scrc.contracts import ConfidenceTier, ShapValue, StockoutRiskResult

_DEFAULT_XGB = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "eval_metric": "logloss",
}


def _confidence_tier(probability: float) -> ConfidenceTier:
    """Confidence from distance to the decision boundary.

    A MAPIE conformal prediction set is the production replacement (PRD §4.2);
    this margin-based proxy keeps the contract populated until then.
    """
    margin = abs(probability - 0.5)
    if margin >= 0.35:
        return ConfidenceTier.HIGH
    if margin >= 0.15:
        return ConfidenceTier.MEDIUM
    return ConfidenceTier.LOW


class StockoutClassifier:
    """Trainable, SHAP-explained stockout classifier."""

    def __init__(self, feature_names: Sequence[str], random_state: int = 42, **xgb_params: object):
        self._features = list(feature_names)
        self._random_state = random_state
        self._params = {**_DEFAULT_XGB, **xgb_params}
        self._xgb: XGBClassifier | None = None
        self._calibrated: CalibratedClassifierCV | None = None

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        calibration_fraction: float = 0.25,
    ) -> None:
        """Fit XGBoost, then isotonic-calibrate on a held-out split (prefit)."""
        X = X[self._features]
        x_fit, x_cal, y_fit, y_cal = train_test_split(
            X, y, test_size=calibration_fraction, random_state=self._random_state, stratify=y
        )
        xgb = XGBClassifier(random_state=self._random_state, **self._params)
        xgb.fit(x_fit, y_fit)
        # Calibrate without refitting the booster (FrozenEstimator replaces the
        # removed cv="prefit"); isotonic per PRD §4.2.
        calibrated = CalibratedClassifierCV(FrozenEstimator(xgb), method="isotonic")
        calibrated.fit(x_cal, y_cal)
        self._xgb = xgb
        self._calibrated = calibrated

    def _row(self, features: Mapping[str, float]) -> pd.DataFrame:
        missing = [f for f in self._features if f not in features]
        if missing:
            raise KeyError(f"missing features: {missing}")
        return pd.DataFrame([[features[f] for f in self._features]], columns=self._features)

    def predict(
        self,
        sku_id: str,
        store_id: str,
        features: Mapping[str, float],
        top_k: int = 5,
    ) -> StockoutRiskResult:
        if self._xgb is None or self._calibrated is None:
            raise RuntimeError("model is not trained")
        row = self._row(features)
        probability = float(self._calibrated.predict_proba(row)[0, 1])

        shap_row = self._shap_row(row)
        order = np.argsort(np.abs(shap_row))[::-1][:top_k]
        shap_values = [
            ShapValue(
                feature=self._features[i],
                value=float(row.iloc[0, i]),
                contribution=float(shap_row[i]),
            )
            for i in order
        ]
        return StockoutRiskResult(
            sku_id=sku_id,
            store_id=store_id,
            stockout_probability=probability,
            calibrated=True,
            confidence_tier=_confidence_tier(probability),
            shap_values=shap_values,
        )

    def _shap_row(self, row: pd.DataFrame) -> np.ndarray:
        """SHAP values for the positive class, shape (n_features,)."""
        explainer = shap.TreeExplainer(self._xgb)
        values = np.asarray(explainer.shap_values(row))
        if values.ndim == 3:  # (n, features, classes)
            values = values[..., -1]
        return values.reshape(-1)
