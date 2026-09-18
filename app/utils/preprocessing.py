"""Preparación de vectores de entrada para el clasificador TabPFN-v2.

Funciones puras de preprocesamiento y validación de rangos, testables
sin inicializar Streamlit.
"""

import numpy as np

NOMBRE_FEATURES = [
    "Type_H",
    "Type_L",
    "Type_M",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Rangos válidos extraídos del dataset AI4I 2020
RANGOS_VALIDOS = {
    "air_temp": (290.0, 320.0),
    "process_temp": (300.0, 330.0),
    "rpm": (1000.0, 3000.0),
    "torque": (1.0, 100.0),
    "tool_wear": (0.0, 300.0),
}


def preprocess_input(
    product_type: str,
    air_temp: float,
    process_temp: float,
    rpm: float,
    torque: float,
    tool_wear: float,
) -> np.ndarray:
    """Convierte los 6 inputs del formulario en el vector de 8 features del modelo.

    Aplica one-hot encoding a `product_type` (L, M, H) y concatena con las
    5 variables numéricas en el orden exacto que espera el clasificador.

    Args:
        product_type: Tipo de producto (`"L"`, `"M"` o `"H"`).
        air_temp: Temperatura del aire en Kelvin.
        process_temp: Temperatura del proceso en Kelvin.
        rpm: Velocidad de rotación en RPM.
        torque: Torque en Newton-metro.
        tool_wear: Desgaste de herramienta en minutos.

    Returns:
        Array numpy de forma `(1, 8)` con los features en el orden del modelo.
    """
    type_h = 1.0 if product_type == "H" else 0.0
    type_l = 1.0 if product_type == "L" else 0.0
    type_m = 1.0 if product_type == "M" else 0.0
    return np.array(
        [[type_h, type_l, type_m, air_temp, process_temp, rpm, torque, tool_wear]],
        dtype=float,
    )
