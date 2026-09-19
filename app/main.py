"""Orquestador principal de la interfaz de mantenimiento predictivo."""

import sys
from pathlib import Path

# Streamlit ejecuta este archivo directamente, así que agrega `app/` al path y no la
# raíz del proyecto. Sin esto, los imports `app.*` de abajo fallan con ModuleNotFoundError.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app.components.form import render_machine_form
from app.components.results import render_results
from app.components.settings import render_alert_sensitivity
from app.controllers import derivar_vista, ejecutar_prediccion
from app.utils.model import load_model_and_context

st.set_page_config(
    page_title="Mantenimiento Predictivo",
    page_icon=":material/engineering:",
    layout="centered",
)

st.title("Mantenimiento Predictivo")
st.caption("Clasificación de fallas con TabPFN-v2 · Dataset AI4I 2020")

umbral_falla = render_alert_sensitivity()

model = load_model_and_context()

inputs = render_machine_form()

if inputs:
    with st.spinner("Ejecutando inferencia con TabPFN-v2..."):
        st.session_state.resultado = ejecutar_prediccion(model, inputs)

if "resultado" in st.session_state:
    # El umbral se aplica aquí y no al predecir: mover el slider recalcula la
    # alerta sobre la misma probabilidad, sin repetir la inferencia.
    vista = derivar_vista(st.session_state.resultado, umbral_falla)
    render_results(
        proba=vista["proba"],
        X_new=vista["X_new"],
        alerta=vista["alerta"],
        potencia=vista["potencia"],
        delta_t=vista["delta_t"],
    )
