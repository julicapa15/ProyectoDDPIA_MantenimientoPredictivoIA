"""Renderizado de resultados: alertas, métricas y detalles técnicos.

Vista del patrón MVC: solo dibuja valores ya resueltos por
`app/controllers/prediction.py`. No invoca funciones de dominio ni deriva
magnitudes propias.
"""

import numpy as np
import streamlit as st

from app.utils.preprocessing import NOMBRE_FEATURES


def render_results(
    proba: np.ndarray,
    X_new: np.ndarray,
    alerta: str,
    potencia: float,
    delta_t: float,
) -> None:
    """Renderiza la sección de resultados tras la inferencia.

    Muestra el semáforo de alerta, métricas físicas derivadas y detalles
    técnicos de la predicción en expanders.

    Args:
        proba: Matriz de probabilidades `(1, 2)` del modelo.
        X_new: Vector de features `(1, 8)` enviado al modelo.
        alerta: Nivel ya resuelto por el controlador: `"Normal"`,
            `"Precaucion"` o `"Falla inminente"`.
        potencia: Potencia mecánica en watts, ya calculada.
        delta_t: Diferencia de temperatura proceso − aire en Kelvin, ya calculada.
    """
    prob_falla = float(proba[0, 1])

    st.subheader("Resultado")

    if alerta == "Normal":
        st.success(f"**{alerta}** — Probabilidad de falla: {prob_falla:.1%}")
    elif alerta == "Precaucion":
        st.warning(f"**{alerta}** — Probabilidad de falla: {prob_falla:.1%}")
    else:
        st.error(f"**{alerta}** — Probabilidad de falla: {prob_falla:.1%}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ΔT (Proceso − Aire)", f"{delta_t:.1f} K")
    with col2:
        st.metric("Potencia mecánica", f"{potencia:.0f} W")
    with col3:
        st.metric("Prob. de falla", f"{prob_falla:.1%}")

    with st.expander("Detalles de la predicción"):
        st.write("**Vector de entrada (8 features):**")
        st.code(
            dict(zip(NOMBRE_FEATURES, X_new[0], strict=True)),
            language="json",
        )
        st.write(
            f"**Probabilidades completas:** No falla {proba[0, 0]:.4f} · Falla {proba[0, 1]:.4f}"
        )

    with st.expander("Información del modelo TabPFN-v2"):
        st.markdown(
            "- **Arquitectura:** Prior-data Fitted Network"
            " (In-Context Learning con Transformer).\n"
            "- **Modo de inferencia:** Inferencia directa sin ajuste por gradiente.\n"
            "- **Muestras en contexto:** 1,000 registros estratificados"
            " (optimizado para CPU)."
        )
