"""Interfaz web de mantenimiento predictivo con TabPFN-v2.

Permite al usuario ingresar las 6 variables de operación de una máquina y
obtener la probabilidad de falla en tiempo real usando el modelo TabPFN-v2
como clasificador basado en contexto.
"""

import sys
from pathlib import Path

# Garantizar que la raíz del proyecto esté en sys.path para importar `src`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import streamlit as st

from src.models import TabPFNClassifier
from src.preprocessing import build_features, load_raw_data

# ── Constantes ───────────────────────────────────────────────────────────────

DATASET_PATH = ROOT_DIR / "data" / "raw" / "ai4i2020.csv"

UMBRAL_PRECAUCION = 0.30
UMBRAL_FALLA = 0.70

# Columnas en el orden que espera el modelo tras preprocesamiento:
# [Type_H, Type_L, Type_M, Air temp, Process temp, Rotational speed, Torque, Tool wear]
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


# ── Funciones puras (testables sin Streamlit) ────────────────────────────────


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


# ── Recursos cacheados ───────────────────────────────────────────────────────

N_CONTEXTO_WEB = (
    1000  # 1.000 muestras preservan la proporción de falla y permiten inferencia en ~1-2 s en CPU
)


@st.cache_resource(show_spinner="Cargando datos y preparando TabPFN-v2...")
def _load_model_and_context(
    n_samples: int = N_CONTEXTO_WEB,
) -> tuple[TabPFNClassifier, np.ndarray, np.ndarray]:
    """Carga el dataset y prepara un contexto estratificado optimizado para inferencia en CPU.

    Args:
        n_samples: Número de muestras de referencia a usar en contexto (1000 por defecto
            para lograr tiempos de respuesta de 1-2 segundos en CPU).

    Returns:
        Tupla `(modelo, X_ctx, y_ctx)` lista para inferencia en tiempo real.
    """
    df = load_raw_data(DATASET_PATH)
    X_train, y_train, _, _ = build_features(df)

    # Submuestreo estratificado para respuesta rápida en la interfaz web
    if len(X_train) > n_samples:
        from sklearn.model_selection import train_test_split

        X_ctx, _, y_ctx, _ = train_test_split(
            X_train,
            y_train,
            train_size=n_samples,
            stratify=y_train,
            random_state=42,
        )
    else:
        X_ctx, y_ctx = X_train, y_train

    model = TabPFNClassifier()
    model._estimator.fit(X_ctx, y_ctx)
    return model, X_ctx, y_ctx


# ── Interfaz Streamlit ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Mantenimiento Predictivo",
    page_icon=":material/engineering:",
    layout="centered",
)

st.title("Mantenimiento Predictivo")
st.caption("Clasificación de fallas con TabPFN-v2 · Dataset AI4I 2020")

# Carga de recursos (una sola vez por sesión gracias a @st.cache_resource)
model, X_train, y_train = _load_model_and_context()

# ── Formulario de entrada ────────────────────────────────────────────────────

st.subheader("Datos de la máquina")

with st.form("machine_input", border=True):
    col_tipo, col_desgaste = st.columns([1, 1])
    with col_tipo:
        product_type = st.selectbox("Tipo de producto", ["L", "M", "H"])
    with col_desgaste:
        tool_wear = st.number_input(
            "Desgaste de herramienta [min]",
            min_value=0.0,
            max_value=300.0,
            value=0.0,
            step=1.0,
        )

    col_temp1, col_temp2 = st.columns(2)
    with col_temp1:
        air_temp = st.number_input(
            "Temperatura del aire [K]",
            min_value=290.0,
            max_value=320.0,
            value=298.1,
            step=0.1,
        )
    with col_temp2:
        process_temp = st.number_input(
            "Temperatura del proceso [K]",
            min_value=300.0,
            max_value=330.0,
            value=308.6,
            step=0.1,
        )

    col_rpm, col_torque = st.columns(2)
    with col_rpm:
        rpm = st.number_input(
            "Velocidad de rotación [rpm]",
            min_value=1000.0,
            max_value=3000.0,
            value=1500.0,
            step=1.0,
        )
    with col_torque:
        torque = st.number_input(
            "Torque [Nm]",
            min_value=1.0,
            max_value=100.0,
            value=40.0,
            step=0.1,
        )

    submitted = st.form_submit_button("Predecir falla", icon=":material/robot:", type="primary")

# ── Predicción y resultados ──────────────────────────────────────────────────

if submitted:
    with st.spinner("Ejecutando inferencia con TabPFN-v2..."):
        X_new = preprocess_input(product_type, air_temp, process_temp, rpm, torque, tool_wear)
        proba = model._estimator.predict_proba(X_new)
        prob_falla = float(proba[0, 1])
        alerta = get_alert_level(prob_falla)
        potencia = calculate_power(torque, rpm)
        delta_t = process_temp - air_temp

    st.subheader("Resultado")

    # Alerta visual
    if alerta == "Normal":
        st.success(f"**{alerta}** — Probabilidad de falla: {prob_falla:.1%}")
    elif alerta == "Precaucion":
        st.warning(f"**{alerta}** — Probabilidad de falla: {prob_falla:.1%}")
    else:
        st.error(f"**{alerta}** — Probabilidad de falla: {prob_falla:.1%}")

    # Métricas calculadas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ΔT (Proceso − Aire)", f"{delta_t:.1f} K")
    with col2:
        st.metric("Potencia mecánica", f"{potencia:.0f} W")
    with col3:
        st.metric("Prob. de falla", f"{prob_falla:.1%}")

    # Detalles técnicos
    with st.expander("Detalles de la predicción"):
        st.write("**Vector de entrada (8 features):**")
        st.code(
            dict(zip(NOMBRE_FEATURES, X_new[0], strict=False)),
            language="json",
        )
        st.write(
            f"**Probabilidades completas:** No falla {proba[0, 0]:.4f} · Falla {proba[0, 1]:.4f}"
        )

    # Información sobre el modelo
    with st.expander("Información del modelo TabPFN-v2"):
        n_ref = len(X_train)
        st.markdown(
            f"""
            - **Arquitectura:** Prior-data Fitted Network (In-Context Learning con Transformer).
            - **Modo de inferencia:** Inferencia directa sin ajuste por gradiente.
            - **Muestras en contexto:** {n_ref:,} registros estratificados (optimizado para CPU).
            """
        )
