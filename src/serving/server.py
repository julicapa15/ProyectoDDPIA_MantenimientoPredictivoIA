"""Servidor gRPC del servicio de inferencia de mantenimiento predictivo.

Aísla el modelo TabPFN-v2 (y su stack pesado: torch, tabpfn) en un proceso
propio, separado de la interfaz Streamlit — ver `app/utils/model.py`, el
cliente gRPC que lo consume. El vector de 8 features ya llega preprocesado
(ver `proto/mantenimiento.proto`); este servicio solo predice.

Uso: `uv run python -m src.serving.server` (puerto configurable con la
variable de entorno `GRPC_PORT`, por defecto 50051).
"""

import logging
import os
from concurrent import futures

import grpc
import numpy as np

from proto import mantenimiento_pb2, mantenimiento_pb2_grpc
from src.preprocessing.preprocess import COLUMNAS_DUMMY, COLUMNAS_NUMERICAS
from src.serving.model_loader import load_model_and_context

logger = logging.getLogger(__name__)

DEFAULT_PORT = "50051"
MAX_WORKERS = 4
N_FEATURES = len(COLUMNAS_DUMMY) + len(COLUMNAS_NUMERICAS)


class InferenceServicer(mantenimiento_pb2_grpc.InferenceServiceServicer):
    """Implementa el servicio gRPC de inferencia sobre TabPFN-v2."""

    def __init__(self) -> None:
        """Carga el dataset y ajusta el contexto de TabPFN-v2 una sola vez."""
        logger.info("Cargando dataset y ajustando contexto de TabPFN-v2...")
        self._model, _, _ = load_model_and_context()
        logger.info("Modelo listo para inferencia.")

    def PredictFailure(
        self,
        request: mantenimiento_pb2.PredictionRequest,
        context: grpc.ServicerContext,
    ) -> mantenimiento_pb2.PredictionResponse:
        """Predice la probabilidad de falla para el vector de 8 features recibido.

        Args:
            request: Vector `features` (longitud 8) ya preprocesado por el cliente.
            context: Contexto gRPC de la llamada; se usa para rechazar entradas
                inválidas con `INVALID_ARGUMENT`.

        Returns:
            Probabilidades de no-falla y falla, en `[0, 1]`.

        Raises:
            grpc.RpcError: Vía `context.abort(INVALID_ARGUMENT, ...)` si el vector
                no tiene `N_FEATURES` valores o alguno no es finito. Sin esta
                validación, TabPFN fallaría con un error interno (`UNKNOWN`).
        """
        X_new = np.array([request.features], dtype=float)
        if X_new.shape[1] != N_FEATURES:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Se esperaban {N_FEATURES} features, se recibieron {X_new.shape[1]}",
            )
        if not np.isfinite(X_new).all():
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Las features deben ser números finitos (sin NaN ni infinito)",
            )
        proba = self._model.predict_proba_con_contexto(X_new)
        return mantenimiento_pb2.PredictionResponse(
            proba_no_failure=float(proba[0, 0]),
            proba_failure=float(proba[0, 1]),
        )

    def HealthCheck(
        self,
        request: mantenimiento_pb2.HealthCheckRequest,
        context: grpc.ServicerContext,
    ) -> mantenimiento_pb2.HealthCheckResponse:
        """Confirma que el modelo está cargado y listo para inferencia.

        Args:
            request: Vacío.
            context: Contexto gRPC de la llamada (no usado).

        Returns:
            `ready=True` si el servicio llegó a responder (el modelo se carga
            en `__init__`, antes de que el servidor acepte conexiones).
        """
        del request, context
        return mantenimiento_pb2.HealthCheckResponse(ready=True)


def serve(port: str = DEFAULT_PORT) -> None:
    """Arranca el servidor gRPC y bloquea hasta que se detenga.

    Args:
        port: Puerto TCP donde escucha el servicio.
    """
    logging.basicConfig(level=logging.INFO)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS))
    mantenimiento_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("Servicio de inferencia escuchando en el puerto %s", port)
    server.wait_for_termination()


if __name__ == "__main__":
    serve(os.environ.get("GRPC_PORT", DEFAULT_PORT))
