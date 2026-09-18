"""Tests unitarios para la interfaz Streamlit de mantenimiento predictivo.

Ejercitan las funciones puras del módulo app.main sin necesidad de lanzar
el servidor Streamlit. La estructura AAA sigue las reglas de AGENTS.md.
"""

import math

import numpy as np

from app.main import calculate_power, get_alert_level, preprocess_input

# ── preprocess_input ─────────────────────────────────────────────────────────


def test_preprocess_input_tipo_l():
    # 1. ARRANGE
    product_type = "L"

    # 2. ACT
    resultado = preprocess_input(product_type, 298.1, 308.6, 1551, 42.8, 0)

    # 3. ASSERT
    assert resultado.shape == (1, 8)
    assert resultado[0, 0] == 0.0  # Type_H
    assert resultado[0, 1] == 1.0  # Type_L
    assert resultado[0, 2] == 0.0  # Type_M


def test_preprocess_input_tipo_m():
    # 1. ARRANGE
    product_type = "M"

    # 2. ACT
    resultado = preprocess_input(product_type, 298.1, 308.6, 1551, 42.8, 0)

    # 3. ASSERT
    assert resultado[0, 0] == 0.0
    assert resultado[0, 1] == 0.0
    assert resultado[0, 2] == 1.0


def test_preprocess_input_tipo_h():
    # 1. ARRANGE
    product_type = "H"

    # 2. ACT
    resultado = preprocess_input(product_type, 298.1, 308.6, 1551, 42.8, 0)

    # 3. ASSERT
    assert resultado[0, 0] == 1.0
    assert resultado[0, 1] == 0.0
    assert resultado[0, 2] == 0.0


def test_preprocess_input_valores_numericos():
    # 1. ARRANGE
    air_temp, process_temp, rpm, torque, tool_wear = 299.5, 309.2, 1400, 50.0, 120.0

    # 2. ACT
    resultado = preprocess_input("M", air_temp, process_temp, rpm, torque, tool_wear)

    # 3. ASSERT
    assert resultado[0, 3] == air_temp
    assert resultado[0, 4] == process_temp
    assert resultado[0, 5] == rpm
    assert resultado[0, 6] == torque
    assert resultado[0, 7] == tool_wear


def test_preprocess_input_es_numpy():
    # 1. ARRANGE
    product_type = "L"

    # 2. ACT
    resultado = preprocess_input(product_type, 298.1, 308.6, 1551, 42.8, 0)

    # 3. ASSERT
    assert isinstance(resultado, np.ndarray)
    assert resultado.dtype == float


def test_preprocess_input_one_hot_es_mutuamente_exclusivo():
    # 1. ARRANGE
    tipos = ["L", "M", "H"]

    # 2. ACT
    vectores = [preprocess_input(t, 298, 308, 1500, 40, 0)[0, :3] for t in tipos]

    # 3. ASSERT (cada vector tiene exactamente un 1 entre las primeras 3 posiciones)
    for v in vectores:
        assert v.sum() == 1.0


# ── get_alert_level ──────────────────────────────────────────────────────────


def test_alert_level_normal():
    # 1. ARRANGE
    probabilidad = 0.10

    # 2. ACT
    nivel = get_alert_level(probabilidad)

    # 3. ASSERT
    assert nivel == "Normal"


def test_alert_level_precaucion():
    # 1. ARRANGE
    probabilidad = 0.50

    # 2. ACT
    nivel = get_alert_level(probabilidad)

    # 3. ASSERT
    assert nivel == "Precaucion"


def test_alert_level_falla_inminente():
    # 1. ARRANGE
    probabilidad = 0.85

    # 2. ACT
    nivel = get_alert_level(probabilidad)

    # 3. ASSERT
    assert nivel == "Falla inminente"


def test_alert_level_limite_inferior():
    # 1. ARRANGE
    probabilidad = 0.0

    # 2. ACT
    nivel = get_alert_level(probabilidad)

    # 3. ASSERT
    assert nivel == "Normal"


def test_alert_level_limite_superior():
    # 1. ARRANGE
    probabilidad = 1.0

    # 2. ACT
    nivel = get_alert_level(probabilidad)

    # 3. ASSERT
    assert nivel == "Falla inminente"


def test_alert_level_en_umbral_precaucion():
    # 1. ARRANGE
    probabilidad = 0.30

    # 2. ACT
    nivel = get_alert_level(probabilidad)

    # 3. ASSERT (0.30 no es < 0.30, cae en Precaucion)
    assert nivel == "Precaucion"


def test_alert_level_en_umbral_falla():
    # 1. ARRANGE
    probabilidad = 0.70

    # 2. ACT
    nivel = get_alert_level(probabilidad)

    # 3. ASSERT (0.70 no es < 0.70, cae en Falla inminente)
    assert nivel == "Falla inminente"


# ── calculate_power ──────────────────────────────────────────────────────────


def test_potencia_cero_rpm():
    # 1. ARRANGE
    torque = 50.0
    rpm = 0.0

    # 2. ACT
    potencia = calculate_power(torque, rpm)

    # 3. ASSERT
    assert potencia == 0.0


def test_potencia_cero_torque():
    # 1. ARRANGE
    torque = 0.0
    rpm = 1500.0

    # 2. ACT
    potencia = calculate_power(torque, rpm)

    # 3. ASSERT
    assert potencia == 0.0


def test_potencia_valores_conocidos():
    # 1. ARRANGE (torque=10 Nm, rpm=60 → ω = 2π rad/s → P = 20π ≈ 62.83 W)
    torque = 10.0
    rpm = 60.0
    esperado = 10.0 * 2.0 * math.pi

    # 2. ACT
    potencia = calculate_power(torque, rpm)

    # 3. ASSERT
    assert math.isclose(potencia, esperado, rel_tol=1e-9)


def test_potencia_rpm_altos():
    # 1. ARRANGE
    torque = 42.8
    rpm = 1551.0

    # 2. ACT
    potencia = calculate_power(torque, rpm)

    # 3. ASSERT (P > 0 y en un rango razonable)
    assert potencia > 0
    assert potencia < 100_000  # límite superior de sanity check


def test_potencia_es_escalar():
    # 1. ARRANGE
    torque = 40.0
    rpm = 1500.0

    # 2. ACT
    potencia = calculate_power(torque, rpm)

    # 3. ASSERT
    assert isinstance(potencia, float)
