"""Formulario interactivo para captura de variables de la máquina.

Vista del patrón MVC: solo captura entradas. Los límites de cada campo salen de
`RANGOS_VALIDOS` (`app/utils/preprocessing.py`), única fuente de verdad, para
que el formulario no pueda aceptar valores fuera del dominio del dataset.
"""

import streamlit as st

from app.utils.preprocessing import RANGOS_VALIDOS


def render_machine_form() -> dict[str, float | str] | None:
    """Renderiza el formulario de entrada de las 6 variables operacionales.

    Returns:
        Diccionario con los valores ingresados si se envió el formulario,
        `None` si aún no se ha presionado "Predecir falla".
    """
    st.subheader("Datos de la máquina")

    with st.form("machine_input", border=True):
        col_tipo, col_desgaste = st.columns([1, 1])
        with col_tipo:
            product_type = st.selectbox("Tipo de producto", ["L", "M", "H"])
        with col_desgaste:
            tool_wear = st.number_input(
                "Desgaste de herramienta [min]",
                min_value=RANGOS_VALIDOS["tool_wear"][0],
                max_value=RANGOS_VALIDOS["tool_wear"][1],
                value=0.0,
                step=1.0,
            )

        col_temp1, col_temp2 = st.columns(2)
        with col_temp1:
            air_temp = st.number_input(
                "Temperatura del aire [K]",
                min_value=RANGOS_VALIDOS["air_temp"][0],
                max_value=RANGOS_VALIDOS["air_temp"][1],
                value=298.1,
                step=0.1,
            )
        with col_temp2:
            process_temp = st.number_input(
                "Temperatura del proceso [K]",
                min_value=RANGOS_VALIDOS["process_temp"][0],
                max_value=RANGOS_VALIDOS["process_temp"][1],
                value=308.6,
                step=0.1,
            )

        col_rpm, col_torque = st.columns(2)
        with col_rpm:
            rpm = st.number_input(
                "Velocidad de rotación [rpm]",
                min_value=RANGOS_VALIDOS["rpm"][0],
                max_value=RANGOS_VALIDOS["rpm"][1],
                value=1500.0,
                step=1.0,
            )
        with col_torque:
            torque = st.number_input(
                "Torque [Nm]",
                min_value=RANGOS_VALIDOS["torque"][0],
                max_value=RANGOS_VALIDOS["torque"][1],
                value=40.0,
                step=0.1,
            )

        submitted = st.form_submit_button("Predecir falla", icon=":material/robot:", type="primary")

    if not submitted:
        return None

    return {
        "product_type": product_type,
        "air_temp": air_temp,
        "process_temp": process_temp,
        "rpm": rpm,
        "torque": torque,
        "tool_wear": tool_wear,
    }
