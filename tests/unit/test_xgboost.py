"""T020 [US3] Tests unitarios para el baseline XGBoost."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from src.models import XGBoostBaseline


@pytest.fixture(scope="module")
def datos_sinteticos():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(200, 8))
    y_train = (rng.random(200) < 0.2).astype(int)
    X_test = rng.normal(size=(40, 8))
    return X_train, y_train, X_test


def test_se_instancia_sin_errores():
    # 1. ARRANGE (Peso de la clase minoritaria conocido)
    peso = 30.0

    # 2. ACT (Instanciar el baseline)
    modelo = XGBoostBaseline(scale_pos_weight=peso)

    # 3. ASSERT (Conserva el peso recibido)
    assert modelo.scale_pos_weight == peso


def test_from_class_balance_calcula_el_peso():
    # 1. ARRANGE (Etiquetas con 970 negativos y 30 positivos)
    y = np.array([0] * 970 + [1] * 30)

    # 2. ACT (Construir el modelo desde el balance de clases)
    modelo = XGBoostBaseline.from_class_balance(y)

    # 3. ASSERT (scale_pos_weight = negativos / positivos)
    assert modelo.scale_pos_weight == pytest.approx(970 / 30)


def test_from_class_balance_sin_positivos_lanza_error():
    # 1. ARRANGE (Etiquetas sin ningún caso de falla)
    y = np.zeros(100, dtype=int)

    # 2. ACT (Intentar derivar el peso de clase sin ningún positivo)
    with pytest.raises(ValueError) as excinfo:
        XGBoostBaseline.from_class_balance(y)

    # 3. ASSERT (El error explica que no hay clase falla que ponderar)
    assert "clase falla" in str(excinfo.value)


def test_fit_y_predict_proba(datos_sinteticos):
    # 1. ARRANGE (Datos sintéticos y modelo ponderado por el desbalance)
    X_train, y_train, X_test = datos_sinteticos
    modelo = XGBoostBaseline.from_class_balance(y_train)

    # 2. ACT (Entrenar y predecir probabilidades)
    modelo.fit(X_train, y_train)
    proba = modelo.predict_proba(X_test)

    # 3. ASSERT (Matriz (n, 2) de probabilidades válidas)
    assert proba.shape == (len(X_test), 2)
    assert proba.min() >= 0.0 and proba.max() <= 1.0
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_predecir_sin_entrenar_falla(datos_sinteticos):
    # 1. ARRANGE (Modelo recién instanciado, sin entrenar)
    _, _, X_test = datos_sinteticos
    modelo = XGBoostBaseline(scale_pos_weight=4.0)

    # 2. ACT (Predecir antes de haber llamado a fit)
    with pytest.raises(NotFittedError) as excinfo:
        modelo.predict_proba(X_test)

    # 3. ASSERT (El mensaje indica que falta entrenar el modelo)
    assert "fit" in str(excinfo.value)


def test_es_reproducible(datos_sinteticos):
    # 1. ARRANGE (Mismos datos y misma semilla en dos entrenamientos)
    X_train, y_train, X_test = datos_sinteticos

    # 2. ACT (Entrenar y predecir dos veces por separado)
    predicciones = []
    for _ in range(2):
        modelo = XGBoostBaseline(scale_pos_weight=4.0, random_state=42)
        modelo.fit(X_train, y_train)
        predicciones.append(modelo.predict_proba(X_test))

    # 3. ASSERT (La semilla fija garantiza predicciones idénticas)
    assert np.allclose(predicciones[0], predicciones[1])


def test_fit_retorna_self(datos_sinteticos):
    # 1. ARRANGE (Datos sintéticos y modelo ponderado)
    X_train, y_train, _ = datos_sinteticos
    modelo = XGBoostBaseline.from_class_balance(y_train)

    # 2. ACT (Entrenar y capturar el valor de retorno)
    retorno = modelo.fit(X_train, y_train)

    # 3. ASSERT (fit retorna self para permitir encadenamiento)
    assert retorno is modelo


def test_guarda_modelo_en_disco(datos_sinteticos, tmp_path):
    # 1. ARRANGE (Modelo entrenado y ruta de destino)
    X_train, y_train, _ = datos_sinteticos
    modelo = XGBoostBaseline.from_class_balance(y_train)
    modelo.fit(X_train, y_train)
    ruta_modelo = tmp_path / "modelo_test.json"

    # 2. ACT (Serializar el modelo en disco)
    ruta_guardada = modelo.save(ruta_modelo)

    # 3. ASSERT (El archivo fue creado y tiene contenido)
    assert ruta_guardada.exists()
    assert ruta_guardada.stat().st_size > 0


def test_from_class_balance_random_state_personalizado():
    # 1. ARRANGE (Etiquetas con 970 negativos y 30 positivos)
    y = np.array([0] * 970 + [1] * 30)

    # 2. ACT (Construir el modelo con un random_state personalizado)
    modelo = XGBoostBaseline.from_class_balance(y, random_state=123)

    # 3. ASSERT (El random_state se propagó correctamente)
    assert modelo.random_state == 123
