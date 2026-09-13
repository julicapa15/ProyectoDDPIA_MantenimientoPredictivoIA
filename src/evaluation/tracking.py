"""Registro de runs en MLflow.

MLflow es el registro oficial de experimentos del proyecto: cada run debe indicar
a qué feature de spec-kit pertenece mediante el tag `spec_id` (Governance de la
constitución, FR-008).
"""

import os

import mlflow
import numpy as np

from src.evaluation.metrics import UMBRAL_POR_DEFECTO, compute_metrics

EXPERIMENTO_POR_DEFECTO = "001-tabpfn-xgboost-baseline"
SPEC_ID = "001"

# MLflow 3.16 dejó el file store (`./mlruns`) en modo mantenimiento y exige un
# backend de base de datos para las funciones actuales.
TRACKING_URI_POR_DEFECTO = "sqlite:///mlflow.db"

NOMBRES_DE_RUN = {
    "tabpfn": "tabpfn-v2",
    "xgboost": "xgboost-baseline",
}


def log_run_to_mlflow(
    model_type: str,
    y_true: np.ndarray,
    y_proba_pos: np.ndarray,
    params: dict | None = None,
    spec_id: str = SPEC_ID,
    experiment: str = EXPERIMENTO_POR_DEFECTO,
    threshold: float = UMBRAL_POR_DEFECTO,
    tracking_uri: str | None = None,
) -> dict[str, float]:
    """Registra un run con sus tags, parámetros y métricas.

    Args:
        model_type: `"tabpfn"` o `"xgboost"`; define el nombre del run y el tag.
        y_true: Etiquetas reales del conjunto de prueba.
        y_proba_pos: Probabilidad predicha de la clase falla.
        params: Hiperparámetros a registrar (p. ej. `scale_pos_weight`).
        spec_id: Feature de spec-kit a la que pertenece el run.
        experiment: Nombre del experimento MLflow.
        threshold: Umbral de decisión usado en F1 y Recall.
        tracking_uri: Backend de MLflow. Por defecto toma `MLFLOW_TRACKING_URI`
            y, si no está definida, `sqlite:///mlflow.db`.

    Returns:
        Las métricas registradas.
    """
    metricas = compute_metrics(y_true, y_proba_pos, threshold=threshold)

    mlflow.set_tracking_uri(
        tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or TRACKING_URI_POR_DEFECTO
    )
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=NOMBRES_DE_RUN.get(model_type, model_type)):
        mlflow.set_tag("spec_id", spec_id)
        mlflow.set_tag("model_type", model_type)
        if params:
            mlflow.log_params(params)
        mlflow.log_metrics(metricas)

    return metricas
