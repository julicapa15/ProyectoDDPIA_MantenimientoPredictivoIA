"""Orquestador principal de la interfaz de mantenimiento predictivo."""

import sys
from pathlib import Path

# Streamlit ejecuta este archivo directamente, así que agrega `app/` al path y no la
# raíz del proyecto. Sin esto, los imports `app.*` de abajo fallan con ModuleNotFoundError.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app.components.form import render_machine_form
from app.components.results import render_results
from app.utils.model import load_model_and_context
from app.utils.preprocessing import preprocess_input

st.set_page_config(
    page_title="Mantenimiento Predictivo",
    page_icon=":material/engineering:",
    layout="centered",
)

st.title("Mantenimiento Predictivo")
st.caption("Clasificación de fallas con TabPFN-v2 · Dataset AI4I 2020")

model, X_train, y_train = load_model_and_context()

inputs = render_machine_form()

if inputs:
    with st.spinner("Ejecutando inferencia con TabPFN-v2..."):
        X_new = preprocess_input(
            inputs["product_type"],
            inputs["air_temp"],
            inputs["process_temp"],
            inputs["rpm"],
            inputs["torque"],
            inputs["tool_wear"],
        )
        proba = model._estimator.predict_proba(X_new)

    render_results(
        proba=proba,
        air_temp=inputs["air_temp"],
        process_temp=inputs["process_temp"],
        torque=inputs["torque"],
        rpm=inputs["rpm"],
        X_new=X_new,
    )
