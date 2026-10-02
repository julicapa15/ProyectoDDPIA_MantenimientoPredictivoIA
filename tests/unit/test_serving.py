"""Tests unitarios del servicer gRPC de inferencia (src/serving/server.py).

Solo cubren la validación de entrada, que ocurre antes de usar el modelo: el
servicer se crea sin `__init__`, así que no cargan TabPFN y caben en la RAM del
runner de CI. Las pruebas con el modelo real están en
tests/integration/test_serving_e2e.py y test_grpc_e2e.py.
"""

import grpc
import pytest

from app.utils.preprocessing import preprocess_input
from proto import mantenimiento_pb2
from src.serving.server import N_FEATURES, InferenceServicer


class _ContextoQueAborta:
    """Doble de `grpc.ServicerContext`: `abort` lanza, como hace gRPC de verdad."""

    def __init__(self):
        """Inicializa sin ningún aborto registrado."""
        self.codigo = None

    def abort(self, code, details):
        """Registra el código y corta la ejecución del método, igual que gRPC.

        Args:
            code: `grpc.StatusCode` con el que se aborta la llamada.
            details: Mensaje de error enviado al cliente.

        Raises:
            RuntimeError: Siempre, con `details` como mensaje.
        """
        self.codigo = code
        raise RuntimeError(details)


@pytest.fixture
def servicer_sin_modelo():
    """Servicer sin pasar por `__init__`: la validación no necesita a TabPFN."""
    return InferenceServicer.__new__(InferenceServicer)


@pytest.mark.parametrize(
    "features",
    [[], [0.0] * (N_FEATURES - 1), [0.0] * (N_FEATURES + 1)],
    ids=["vacio", "uno_menos", "uno_mas"],
)
def test_predict_failure_rechaza_vectores_de_largo_incorrecto(servicer_sin_modelo, features):
    # 1. ARRANGE (vector con un número de features distinto al del modelo)
    request = mantenimiento_pb2.PredictionRequest(features=features)
    context = _ContextoQueAborta()

    # 2. ACT (la llamada se corta en `context.abort`)
    with pytest.raises(RuntimeError, match=f"{N_FEATURES} features"):
        servicer_sin_modelo.PredictFailure(request, context)

    # 3. ASSERT (se rechaza como entrada inválida, sin llegar al modelo)
    assert context.codigo == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.parametrize("valor", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_predict_failure_rechaza_valores_no_finitos(servicer_sin_modelo, valor):
    # 1. ARRANGE (vector de largo correcto con un valor no finito)
    features = preprocess_input("M", 298.1, 308.6, 1551, 42.8, 7)[0].tolist()
    features[3] = valor
    context = _ContextoQueAborta()

    # 2. ACT (la llamada se corta en `context.abort`)
    with pytest.raises(RuntimeError, match="finitos"):
        servicer_sin_modelo.PredictFailure(
            mantenimiento_pb2.PredictionRequest(features=features), context
        )

    # 3. ASSERT (se rechaza como entrada inválida, sin llegar al modelo)
    assert context.codigo == grpc.StatusCode.INVALID_ARGUMENT
