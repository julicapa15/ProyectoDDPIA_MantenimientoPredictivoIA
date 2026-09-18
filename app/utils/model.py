"""Gestión de caché y carga del modelo TabPFN-v2.

Encapsula la inicialización del clasificador y el contexto de referencia
con `@st.cache_resource` para que no se recarguen en cada interacción.
"""

from pathlib import Path

import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split

from src.models import TabPFNClassifier
from src.preprocessing import build_features, load_raw_data

DATASET_PATH = Path("data/raw/ai4i2020.csv")
N_CONTEXTO_WEB = 1000


@st.cache_resource(show_spinner="Cargando datos y preparando TabPFN-v2...")
def load_model_and_context(
    n_samples: int = N_CONTEXTO_WEB,
) -> tuple[TabPFNClassifier, np.ndarray, np.ndarray]:
    """Carga el dataset y prepara un contexto estratificado optimizado para CPU.

    Args:
        n_samples: Número de muestras de referencia a usar en contexto (1000 por defecto
            para lograr tiempos de respuesta de 1-2 segundos en CPU).

    Returns:
        Tupla `(modelo, X_ctx, y_ctx)` lista para inferencia en tiempo real.
    """
    df = load_raw_data(DATASET_PATH)
    X_train, y_train, _, _ = build_features(df)

    if len(X_train) > n_samples:
        X_ctx, _, y_ctx, _ = train_test_split(
            X_train,
            y_train,
            train_size=n_samples,
            stratify=y_train,
            random_state=42,
        )
    else:
        X_ctx, y_ctx = X_train, y_train

    model = TabPFNClassifier()
    model._estimator.fit(X_ctx, y_ctx)
    return model, X_ctx, y_ctx
