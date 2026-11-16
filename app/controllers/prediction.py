"""Controlador de la predicción de falla (capa Controlador del patrón MVC).

Coordina el Modelo (`src/`, `app/utils/`) con la Vista (`app/components/`):
recibe los datos crudos del formulario, ejecuta la inferencia y deriva las
magnitudes que la vista necesita mostrar.

No importa Streamlit a propósito: al ser coordinación pura, se puede probar sin
levantar la interfaz, y la Vista queda reducida a dibujar valores ya resueltos.
La inferencia llega inyectada como función: en la app es el cliente gRPC del
servicio `inference` (`app.utils.model.predict_failure`), en las pruebas un doble.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from app.utils.physics import calculate_delta_t, calculate_power, get_alert_level
from app.utils.preprocessing import preprocess_input


def ejecutar_prediccion(
    predecir: Callable[[np.ndarray], np.ndarray],
    inputs: dict[str, float | str],
) -> dict[str, Any]:
    """Convierte los datos del formulario en una predicción del modelo.

    Args:
        predecir: Función que recibe el vector `(1, 8)` y devuelve la matriz
            `(1, 2)` con `[P(no falla), P(falla)]`.
        inputs: Valores devueltos por `render_machine_form()`, con las claves
            `product_type`, `air_temp`, `process_temp`, `rpm`, `torque` y
            `tool_wear`.

    Returns:
        Diccionario con la predicción (`proba`, matriz `(1, 2)`), el vector
        enviado al modelo (`X_new`, forma `(1, 8)`) y las variables de proceso
        necesarias para derivar después las métricas físicas. Es lo que la app
        persiste entre reejecuciones de Streamlit.
    """
    X_new = preprocess_input(
        inputs["product_type"],
        inputs["air_temp"],
        inputs["process_temp"],
        inputs["rpm"],
        inputs["torque"],
        inputs["tool_wear"],
    )
    return {
        "proba": predecir(X_new),
        "X_new": X_new,
        "air_temp": inputs["air_temp"],
        "process_temp": inputs["process_temp"],
        "torque": inputs["torque"],
        "rpm": inputs["rpm"],
    }


def derivar_vista(resultado: dict[str, Any], umbral_falla: float) -> dict[str, Any]:
    """Deriva de una predicción los valores que la Vista debe mostrar.

    Se separa de `ejecutar_prediccion()` porque el nivel de alerta depende del
    umbral configurable en la barra lateral: mover ese slider debe recalcular la
    alerta sobre la misma probabilidad, sin volver a ejecutar la inferencia.

    Args:
        resultado: Diccionario devuelto por `ejecutar_prediccion()`.
        umbral_falla: Umbral de decisión de Falla inminente en `[0, 1]`.

    Returns:
        Diccionario con las claves que espera `render_results()`: `proba`,
        `X_new`, `alerta`, `potencia` (watts) y `delta_t` (Kelvin).
    """
    prob_falla = float(resultado["proba"][0, 1])
    return {
        "proba": resultado["proba"],
        "X_new": resultado["X_new"],
        "alerta": get_alert_level(prob_falla, umbral_falla),
        "potencia": calculate_power(resultado["torque"], resultado["rpm"]),
        "delta_t": calculate_delta_t(resultado["process_temp"], resultado["air_temp"]),
    }
