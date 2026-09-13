"""Baseline clásico XGBoost para contrastar con TabPFN-v2.

A diferencia de TabPFN, este modelo sí se entrena desde cero por gradient
boosting. El desbalance de la clase falla (~3%) se maneja con `scale_pos_weight`,
equivalente a class_weight (FR-005).
"""

from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

RANDOM_STATE = 42


class XGBoostBaseline:
    """Clasificador XGBoost con ponderación de la clase minoritaria."""

    def __init__(self, scale_pos_weight: float, random_state: int = RANDOM_STATE):
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self._estimator = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric="aucpr",
        )

    @classmethod
    def from_class_balance(cls, y_train: np.ndarray, **kwargs) -> "XGBoostBaseline":
        """Construye el modelo con `scale_pos_weight` = negativos / positivos."""
        n_positivos = int((y_train == 1).sum())
        if n_positivos == 0:
            raise ValueError("y_train no contiene ejemplos de la clase falla")
        return cls(scale_pos_weight=(y_train == 0).sum() / n_positivos, **kwargs)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "XGBoostBaseline":
        self._estimator.fit(X_train, y_train)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Devuelve (n_test, 2) con [P(no falla), P(falla)] por muestra."""
        return self._estimator.predict_proba(X_test)

    def save(self, filepath: str | Path) -> Path:
        """Serializa el modelo en formato nativo `.json` de XGBoost (FR-007)."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._estimator.save_model(filepath)
        return filepath
