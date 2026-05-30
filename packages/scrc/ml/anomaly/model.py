"""Freight/logistics anomaly detector (ML serving, Layer 2).

scikit-learn Isolation Forest with **KernelSHAP** per-feature attribution. Emits
a typed ``AnomalyResult`` carrying the anomaly score and the top contributing
features (P2). The anomaly score is ``-score_samples`` so that higher means more
anomalous.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest

from scrc.contracts import AnomalyResult, ShapFeature


class FreightAnomalyDetector:
    """Trainable, SHAP-explained anomaly detector over freight/port features."""

    def __init__(
        self,
        feature_names: Sequence[str],
        random_state: int = 42,
        n_background: int = 50,
        **params: object,
    ):
        self._features = list(feature_names)
        self._random_state = random_state
        self._n_background = n_background
        self._params = params
        self._model: IsolationForest | None = None
        self._background: np.ndarray | None = None

    def fit(self, X: pd.DataFrame) -> None:
        # Operate in numpy throughout (fit + predict + SHAP) so sklearn never
        # warns about feature-name mismatches when KernelSHAP probes the model.
        data = X[self._features].to_numpy()
        self._model = IsolationForest(random_state=self._random_state, **self._params)
        self._model.fit(data)
        rng = np.random.default_rng(self._random_state)
        n = min(self._n_background, len(data))
        self._background = data[rng.choice(len(data), size=n, replace=False)]

    def _score(self, data: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return -self._model.score_samples(data)

    def _row(self, features: Mapping[str, float]) -> pd.DataFrame:
        missing = [f for f in self._features if f not in features]
        if missing:
            raise KeyError(f"missing features: {missing}")
        return pd.DataFrame([[features[f] for f in self._features]], columns=self._features)

    def predict(
        self,
        port_ids: Sequence[str],
        features: Mapping[str, float],
        congestion_score: float,
        top_k: int = 3,
        nsamples: int = 100,
    ) -> AnomalyResult:
        if self._model is None or self._background is None:
            raise RuntimeError("detector is not fitted")
        row = self._row(features).to_numpy()
        score = float(self._score(row)[0])
        flag = bool(self._model.predict(row)[0] == -1)

        explainer = shap.KernelExplainer(self._score, self._background)
        shap_row = np.asarray(explainer.shap_values(row, nsamples=nsamples)).reshape(-1)
        order = np.argsort(np.abs(shap_row))[::-1][:top_k]
        top_features = [
            ShapFeature(feature=self._features[i], shap_value=float(shap_row[i])) for i in order
        ]

        return AnomalyResult(
            port_ids=list(port_ids),
            congestion_score=min(max(congestion_score, 0.0), 1.0),
            anomaly_flag=flag,
            anomaly_score=score,
            top_features=top_features,
        )
