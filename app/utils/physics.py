"""Funciones puras de ingeniería física para mantenimiento predictivo.

Fórmulas de ΔT y potencia mecánica, testables sin inicializar Streamlit.
"""

import numpy as np

UMBRAL_PRECAUCION = 0.30
UMBRAL_FALLA = 0.70


def calculate_power(torque: float, rpm: float) -> float:
    """Calcula la potencia mecánica en watts.

    Fórmula: P = τ · ω, donde ω = 2π · rpm / 60.

    Args:
        torque: Torque en Newton-metro.
        rpm: Velocidad de rotación en revoluciones por minuto.

    Returns:
        Potencia en watts.
    """
    omega = 2.0 * np.pi * rpm / 60.0
    return torque * omega


def calculate_delta_t(process_temp: float, air_temp: float) -> float:
    """Calcula la diferencia de temperatura entre proceso y aire.

    Args:
        process_temp: Temperatura del proceso en Kelvin.
        air_temp: Temperatura del aire en Kelvin.

    Returns:
        ΔT en Kelvin (proceso − aire).
    """
    return process_temp - air_temp


def get_alert_level(probability: float) -> str:
    """Determina el nivel de alerta según la probabilidad de falla.

    Args:
        probability: Probabilidad predicha de falla en `[0, 1]`.

    Returns:
        `"Normal"`, `"Precaucion"` o `"Falla inminente"`.
    """
    if probability < UMBRAL_PRECAUCION:
        return "Normal"
    if probability < UMBRAL_FALLA:
        return "Precaucion"
    return "Falla inminente"
