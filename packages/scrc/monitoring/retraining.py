"""Drift-triggered retraining and the promotion gate (PRD §8.2, ADR-0007).

Optimises the stockout classifier with Optuna (falling back to a small grid when
Optuna isn't installed — the PRD §13 mitigation), trains a candidate, and scores
it. Promotion mirrors the operational HITL pattern: auto-promote only if the
candidate beats production by a margin on the metric; otherwise require human
approval.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from scrc.ml import StockoutClassifier

_FALLBACK_GRID: list[dict[str, int]] = [
    {"n_estimators": 50, "max_depth": 2},
    {"n_estimators": 150, "max_depth": 2},
    {"n_estimators": 150, "max_depth": 4},
    {"n_estimators": 300, "max_depth": 4},
]


class PromotionMode(StrEnum):
    AUTO_PROMOTE = "auto_promote"
    REQUIRES_APPROVAL = "requires_approval"


def _candidate_auc(
    params: Mapping[str, object],
    feature_names: Sequence[str],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int,
) -> float:
    model = train_candidate(feature_names, params, x_train, y_train, seed=seed)
    return evaluate_candidate(model, x_val, y_val)


def optimise_stockout_params(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: Sequence[str],
    n_trials: int = 15,
    seed: int = 0,
) -> dict[str, object]:
    """Search XGBoost hyperparameters; Optuna if available, else a small grid."""
    x_tr, x_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

    try:
        import optuna

        def objective(trial: object) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),  # type: ignore[attr-defined]
                "max_depth": trial.suggest_int("max_depth", 2, 5),  # type: ignore[attr-defined]
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),  # type: ignore[attr-defined]
            }
            return _candidate_auc(params, feature_names, x_tr, y_tr, x_val, y_val, seed)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        return dict(study.best_params)
    except ImportError:
        best = max(
            _FALLBACK_GRID,
            key=lambda p: _candidate_auc(p, feature_names, x_tr, y_tr, x_val, y_val, seed),
        )
        return dict(best)


def train_candidate(
    feature_names: Sequence[str],
    params: Mapping[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = 0,
) -> StockoutClassifier:
    model = StockoutClassifier(feature_names, random_state=seed, **params)
    model.train(X, y)
    return model


def evaluate_candidate(model: StockoutClassifier, x_val: pd.DataFrame, y_val: pd.Series) -> float:
    probs = [
        model.predict(
            "eval", "eval", {str(k): float(v) for k, v in row.items()}
        ).stockout_probability
        for row in x_val.to_dict(orient="records")
    ]
    return float(roc_auc_score(y_val, probs))


def decide_promotion(
    candidate_auc: float, production_auc: float, margin: float = 0.02
) -> PromotionMode:
    """Auto-promote only if the candidate clears production by ``margin``;
    otherwise route to human approval (same gate as operational decisions)."""
    if candidate_auc >= production_auc + margin:
        return PromotionMode.AUTO_PROMOTE
    return PromotionMode.REQUIRES_APPROVAL
