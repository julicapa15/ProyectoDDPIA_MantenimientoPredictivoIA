"""Tests unitarios del controlador de predicción (capa Controlador del MVC)."""

import inspect

import numpy as np
import pytest

from app.components.results import render_results
from app.controllers import derivar_vista, ejecutar_prediccion
from app.utils.physics import UMBRAL_FALLA, UMBRAL_PRECAUCION
from app.utils.preprocessing import NOMBRE_FEATURES


class ModeloFalso:
    """Doble de prueba que devuelve una probabilidad fija y guarda su entrada."""

    def __init__(self, prob_falla: float):
        """Fija la probabilidad de falla que devolverá la inferencia.

        Args:
            prob_falla: Probabilidad de la clase falla en `[0, 1]`.
        """
        self.prob_falla = prob_falla
        self.recibido = None

    def predict_proba_con_contexto(self, X_new: np.ndarray) -> np.ndarray:
        """Imita la inferencia de TabPFN-v2 y registra el vector recibido.

        Args:
            X_new: Vector de features `(1, 8)` enviado por el controlador.

        Returns:
            Matriz `(1, 2)` con [P(no falla), P(falla)].
        """
        self.recibido = X_new
        return np.array([[1.0 - self.prob_falla, self.prob_falla]])


@pytest.fixture
def entradas():
    return {
        "product_type": "L",
        "air_temp": 300.0,
        "process_temp": 310.0,
        "rpm": 1500.0,
        "torque": 40.0,
        "tool_wear": 100.0,
    }


def test_ejecutar_prediccion_envia_al_modelo_el_vector_de_8_features(entradas):
    # 1. ARRANGE (Modelo falso y entradas válidas del formulario)
    modelo = ModeloFalso(prob_falla=0.42)

    # 2. ACT (Ejecutar el controlador de predicción)
    resultado = ejecutar_prediccion(modelo, entradas)

    # 3. ASSERT (El modelo recibió el vector con la forma del pipeline de entrenamiento)
    assert modelo.recibido.shape == (1, len(NOMBRE_FEATURES))
    assert resultado["X_new"].shape == (1, 8)


def test_ejecutar_prediccion_conserva_las_variables_para_las_metricas_derivadas(entradas):
    # 1. ARRANGE (Modelo falso y entradas válidas del formulario)
    modelo = ModeloFalso(prob_falla=0.10)

    # 2. ACT (Ejecutar el controlador de predicción)
    resultado = ejecutar_prediccion(modelo, entradas)

    # 3. ASSERT (Quedan las 4 variables que luego necesita derivar_vista)
    assert resultado["air_temp"] == entradas["air_temp"]
    assert resultado["process_temp"] == entradas["process_temp"]
    assert resultado["torque"] == entradas["torque"]
    assert resultado["rpm"] == entradas["rpm"]


def test_derivar_vista_calcula_potencia_y_delta_t(entradas):
    # 1. ARRANGE (Resultado de una predicción previa)
    resultado = ejecutar_prediccion(ModeloFalso(prob_falla=0.10), entradas)

    # 2. ACT (Derivar los valores que consume la vista)
    vista = derivar_vista(resultado, UMBRAL_FALLA)

    # 3. ASSERT (ΔT = proceso − aire; potencia = torque · 2π · rpm / 60)
    assert vista["delta_t"] == pytest.approx(10.0)
    assert vista["potencia"] == pytest.approx(40.0 * 2 * np.pi * 1500.0 / 60.0)


def test_derivar_vista_aplica_el_umbral_configurable_sin_reejecutar_la_inferencia(entradas):
    # 1. ARRANGE (Una sola predicción con probabilidad entre ambos umbrales)
    prob = (UMBRAL_PRECAUCION + UMBRAL_FALLA) / 2
    resultado = ejecutar_prediccion(ModeloFalso(prob_falla=prob), entradas)

    # 2. ACT (Derivar la vista dos veces, variando solo el umbral)
    vista_conservadora = derivar_vista(resultado, UMBRAL_FALLA)
    vista_preventiva = derivar_vista(resultado, UMBRAL_PRECAUCION + 0.05)

    # 3. ASSERT (Cambia el nivel de alerta, no la probabilidad predicha)
    assert vista_conservadora["alerta"] == "Precaucion"
    assert vista_preventiva["alerta"] == "Falla inminente"
    assert vista_conservadora["proba"] is vista_preventiva["proba"]


def test_derivar_vista_entrega_exactamente_las_claves_que_espera_la_vista(entradas):
    # 1. ARRANGE (Resultado de una predicción previa)
    resultado = ejecutar_prediccion(ModeloFalso(prob_falla=0.90), entradas)

    # 2. ACT (Derivar los valores que consume la vista)
    vista = derivar_vista(resultado, UMBRAL_FALLA)

    # 3. ASSERT (El contrato con render_results no tiene claves de más ni de menos)
    assert set(vista) == {"proba", "X_new", "alerta", "potencia", "delta_t"}


def test_un_umbral_por_debajo_de_precaucion_es_rechazado(entradas):
    # 1. ARRANGE (Predicción válida y un umbral que colapsaría la banda intermedia)
    resultado = ejecutar_prediccion(ModeloFalso(prob_falla=0.5), entradas)
    umbral_invalido = UMBRAL_PRECAUCION - 0.01

    # 2. ACT (Derivar la vista con ese umbral fuera de rango)
    with pytest.raises(ValueError) as excinfo:
        derivar_vista(resultado, umbral_invalido)

    # 3. ASSERT (El error explica que la banda de Precaucion desaparecería)
    assert "Precaucion" in str(excinfo.value)


def test_las_claves_del_controlador_coinciden_con_los_parametros_de_la_vista(entradas):
    # 1. ARRANGE (Vista derivada y firma real de render_results)
    vista = derivar_vista(ejecutar_prediccion(ModeloFalso(prob_falla=0.5), entradas), UMBRAL_FALLA)

    # 2. ACT (Leer los parámetros que declara la vista)
    parametros = set(inspect.signature(render_results).parameters)

    # 3. ASSERT (Controlador y vista hablan el mismo contrato, sin levantar Streamlit)
    assert set(vista) == parametros


def test_ejecutar_prediccion_probabilidad_en_rango(entradas):
    # 1. ARRANGE (Modelo falso con probabilidad conocida)
    modelo = ModeloFalso(prob_falla=0.42)

    # 2. ACT (Ejecutar la predicción)
    resultado = ejecutar_prediccion(modelo, entradas)

    # 3. ASSERT (La probabilidad está en el rango [0, 1])
    prob_falla = float(resultado["proba"][0, 1])
    assert 0.0 <= prob_falla <= 1.0


def test_ejecutar_prediccion_guarda_X_new_numpy(entradas):
    # 1. ARRANGE (Modelo falso)
    modelo = ModeloFalso(prob_falla=0.15)

    # 2. ACT (Ejecutar la predicción)
    resultado = ejecutar_prediccion(modelo, entradas)

    # 3. ASSERT (X_new es un array numpy de dtype float)
    assert isinstance(resultado["X_new"], np.ndarray)
    assert resultado["X_new"].dtype == float


def test_derivar_vista_delta_t_y_potencia_coherentes(entradas):
    # 1. ARRANGE (Predicción con entradas conocidas)
    resultado = ejecutar_prediccion(ModeloFalso(prob_falla=0.20), entradas)

    # 2. ACT (Derivar la vista)
    vista = derivar_vista(resultado, UMBRAL_FALLA)

    # 3. ASSERT (ΔT = proceso − aire y potencia > 0 con valores reales)
    assert vista["delta_t"] == pytest.approx(entradas["process_temp"] - entradas["air_temp"])
    assert vista["potencia"] > 0.0


def test_derivar_vista_proba_es_numpy(entradas):
    # 1. ARRANGE (Predicción válida)
    resultado = ejecutar_prediccion(ModeloFalso(prob_falla=0.50), entradas)

    # 2. ACT (Derivar la vista)
    vista = derivar_vista(resultado, UMBRAL_FALLA)

    # 3. ASSERT (proba es un array numpy con forma (1, 2))
    assert isinstance(vista["proba"], np.ndarray)
    assert vista["proba"].shape == (1, 2)
