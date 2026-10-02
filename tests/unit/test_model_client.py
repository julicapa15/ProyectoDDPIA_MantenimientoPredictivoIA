"""Tests unitarios del cliente gRPC de inferencia (app/utils/model.py).

`check_inference_health`/`predict_failure` apuntan a `INFERENCE_SERVICE_ADDR`,
así que los tests la reasignan con `monkeypatch` en vez de depender del
servicio real: `localhost:1` para el camino sin servidor (falla rápido, sin
timeout largo) y un servidor gRPC real embebido para el camino feliz.
"""

from concurrent import futures

import grpc
import numpy as np
import pytest

import app.utils.model as model_client
from proto import mantenimiento_pb2, mantenimiento_pb2_grpc


def test_get_inference_stub_retorna_un_stub_del_servicio():
    # 1. ARRANGE (dirección sin servidor: abrir el canal no conecta todavía)
    direccion = "localhost:1"

    # 2. ACT (gRPC conecta perezosamente, así que esto no falla)
    stub = model_client.get_inference_stub(direccion)

    # 3. ASSERT
    assert isinstance(stub, mantenimiento_pb2_grpc.InferenceServiceStub)


def test_check_inference_health_es_false_sin_servidor_escuchando(monkeypatch):
    # 1. ARRANGE (puerto localhost sin ningún servicio)
    monkeypatch.setattr(model_client, "INFERENCE_SERVICE_ADDR", "localhost:1")

    # 2. ACT
    listo = model_client.check_inference_health()

    # 3. ASSERT
    assert listo is False


def test_predict_failure_lanza_error_propio_sin_servidor_escuchando(monkeypatch):
    # 1. ARRANGE
    monkeypatch.setattr(model_client, "INFERENCE_SERVICE_ADDR", "localhost:1")
    monkeypatch.setattr(model_client, "TIMEOUT_SEGUNDOS", 2.0)
    X_new = np.zeros((1, 8))

    # 2. ACT + 3. ASSERT (RpcError se traduce a la excepción propia del cliente)
    with pytest.raises(model_client.InferenceServiceUnavailableError):
        model_client.predict_failure(X_new)


@pytest.fixture
def fake_inference_server():
    """Servidor gRPC real con un servicer de juguete (sin cargar TabPFN)."""

    class _ServicerDeJuguete(mantenimiento_pb2_grpc.InferenceServiceServicer):
        def PredictFailure(self, request, context):
            del context
            # Respuesta fija y verificable, no depende de features reales.
            return mantenimiento_pb2.PredictionResponse(proba_no_failure=0.4, proba_failure=0.6)

        def HealthCheck(self, request, context):
            del request, context
            return mantenimiento_pb2.HealthCheckResponse(ready=True)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    mantenimiento_pb2_grpc.add_InferenceServiceServicer_to_server(_ServicerDeJuguete(), server)
    port = server.add_insecure_port("localhost:0")
    server.start()

    yield f"localhost:{port}"

    server.stop(grace=None)


def test_check_inference_health_es_true_con_servidor_real(monkeypatch, fake_inference_server):
    # 1. ARRANGE
    monkeypatch.setattr(model_client, "INFERENCE_SERVICE_ADDR", fake_inference_server)

    # 2. ACT
    listo = model_client.check_inference_health()

    # 3. ASSERT
    assert listo is True


def test_predict_failure_devuelve_la_forma_1_2_esperada_por_results(
    monkeypatch, fake_inference_server
):
    # 1. ARRANGE
    monkeypatch.setattr(model_client, "INFERENCE_SERVICE_ADDR", fake_inference_server)
    X_new = np.zeros((1, 8))

    # 2. ACT
    proba = model_client.predict_failure(X_new)

    # 3. ASSERT (misma forma (1, 2) que esperaba el modelo en proceso)
    assert proba.shape == (1, 2)
    assert proba[0, 0] == pytest.approx(0.4)
    assert proba[0, 1] == pytest.approx(0.6)


def test_health_check_usa_un_timeout_corto_y_no_el_de_la_prediccion():
    # 1. ARRANGE (el banner de estado se consulta en cada rerun de Streamlit)
    timeout_health = model_client.TIMEOUT_HEALTH_SEGUNDOS

    # 2. ACT (comparar con el timeout de la predicción, que sí puede ser largo)
    timeout_prediccion = model_client.TIMEOUT_SEGUNDOS

    # 3. ASSERT (un servicio inalcanzable no congela la página más de unos segundos)
    assert timeout_health <= 5.0
    assert timeout_health < timeout_prediccion


@pytest.fixture
def servidor_que_rechaza_la_peticion():
    """Servidor gRPC real que responde INVALID_ARGUMENT a toda predicción."""

    class _ServicerQueRechaza(mantenimiento_pb2_grpc.InferenceServiceServicer):
        def PredictFailure(self, request, context):
            del request
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "vector mal formado")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    mantenimiento_pb2_grpc.add_InferenceServiceServicer_to_server(_ServicerQueRechaza(), server)
    port = server.add_insecure_port("localhost:0")
    server.start()

    yield f"localhost:{port}"

    server.stop(grace=None)


def test_predict_failure_distingue_peticion_rechazada_de_servicio_caido(
    monkeypatch, servidor_que_rechaza_la_peticion
):
    # 1. ARRANGE (el servicio está arriba pero rechaza la entrada)
    monkeypatch.setattr(model_client, "INFERENCE_SERVICE_ADDR", servidor_que_rechaza_la_peticion)
    X_new = np.zeros((1, 8))

    # 2. ACT (capturar el error que ve la interfaz)
    with pytest.raises(model_client.InferenceServiceError) as excinfo:
        model_client.predict_failure(X_new)

    # 3. ASSERT (no se reporta como "servicio no disponible", y conserva el detalle)
    assert not isinstance(excinfo.value, model_client.InferenceServiceUnavailableError)
    assert "vector mal formado" in str(excinfo.value)
