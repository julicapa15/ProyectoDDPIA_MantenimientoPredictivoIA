"""Envoltorio de TabPFN-v2, el modelo base del proyecto.

TabPFN-v2 es un transformer preentrenado que aprende en contexto: `fit()` solo
fija el conjunto de referencia, no ajusta pesos por gradiente (Principio I de la
constitución). Por eso la interfaz expone `predict_proba(X_train, y_train, X_test)`
en una sola llamada: el conjunto de entrenamiento es parte de la entrada, no un
estado previo del modelo.
"""

from pathlib import Path

import numpy as np
from tabpfn import TabPFNClassifier as _TabPFNEstimator

DEVICE_POR_DEFECTO = "cpu"
RANDOM_STATE = 42


class TabPFNClassifier:
    """Clasificador TabPFN-v2 preentrenado, sin entrenamiento por gradiente."""

    def __init__(self, device: str = DEVICE_POR_DEFECTO, random_state: int = RANDOM_STATE):
        self.device = device
        self.random_state = random_state
        self._estimator = _TabPFNEstimator(device=device, random_state=random_state)

    def predict_proba(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """Predice probabilidades de clase para `X_test` usando `X_train` como contexto.

        Args:
            X_train: Matriz de entrenamiento (n_train, n_features).
            y_train: Etiquetas binarias (n_train,).
            X_test: Matriz a predecir (n_test, n_features).

        Returns:
            Matriz (n_test, 2) con [P(no falla), P(falla)] por muestra.
        """
        self._estimator.fit(X_train, y_train)
        return self._estimator.predict_proba(X_test)

    def save(self, filepath: str | Path) -> Path:
        """Serializa el estado ajustado en formato nativo de TabPFN (FR-007)."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._estimator.save_fit_state(filepath)
        return filepath
