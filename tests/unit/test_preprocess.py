"""T010 [US1] Tests unitarios para preprocess_features()."""

import pandas as pd

from src.preprocessing import preprocess_features

COLUMNAS_PROCESADAS = [
    "Type_H",
    "Type_L",
    "Type_M",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
]


def test_one_hot_genera_tres_dummies(raw_df):
    # 1. ARRANGE (DataFrame crudo con la columna categórica Type)
    df = raw_df

    # 2. ACT (Aplicar el preprocesamiento)
    procesado = preprocess_features(df)

    # 3. ASSERT (Type se expande en una columna por categoría)
    assert {"Type_H", "Type_L", "Type_M"}.issubset(procesado.columns)


def test_dummies_suman_uno_por_fila(raw_df):
    # 1. ARRANGE (DataFrame crudo)
    df = raw_df

    # 2. ACT (Preprocesar y sumar las dummies de cada fila)
    suma_por_fila = preprocess_features(df)[["Type_H", "Type_L", "Type_M"]].sum(axis=1)

    # 3. ASSERT (Cada máquina pertenece a exactamente un tipo)
    assert (suma_por_fila == 1).all()


def test_shape_y_columnas_resultantes(raw_df):
    # 1. ARRANGE (DataFrame crudo de 10.000 filas)
    df = raw_df

    # 2. ACT (Aplicar el preprocesamiento)
    procesado = preprocess_features(df)

    # 3. ASSERT (8 features más el target, en orden fijo)
    assert procesado.shape == (10000, 9)
    assert list(procesado.columns) == COLUMNAS_PROCESADAS


def test_columnas_identificadoras_dropeadas(raw_df):
    # 1. ARRANGE (DataFrame crudo con identificadores)
    df = raw_df

    # 2. ACT (Aplicar el preprocesamiento)
    procesado = preprocess_features(df)

    # 3. ASSERT (Los identificadores no son señal predictiva)
    for columna in ["UDI", "Product ID", "Type"]:
        assert columna not in procesado.columns


def test_submodos_de_falla_dropeados(raw_df):
    # 1. ARRANGE (DataFrame crudo con los 5 submodos de falla)
    df = raw_df

    # 2. ACT (Aplicar el preprocesamiento)
    procesado = preprocess_features(df)

    # 3. ASSERT (Los submodos quedan fuera del alcance de la spec 001)
    for submodo in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
        assert submodo not in procesado.columns


def test_no_muta_el_dataframe_original(raw_df):
    # 1. ARRANGE (Registrar las columnas del DataFrame de entrada)
    columnas_antes = list(raw_df.columns)

    # 2. ACT (Aplicar el preprocesamiento)
    preprocess_features(raw_df)

    # 3. ASSERT (El DataFrame original queda intacto)
    assert list(raw_df.columns) == columnas_antes


def test_dtype_float_en_columnas_numericas(raw_df):
    # 1. ARRANGE (DataFrame crudo)
    df = raw_df

    # 2. ACT (Aplicar el preprocesamiento)
    procesado = preprocess_features(df)

    # 3. ASSERT (Las columnas numéricas se pueden usar como float en sklearn)
    for col in [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]:
        assert pd.api.types.is_numeric_dtype(procesado[col])


def test_dtype_float_en_dummies(raw_df):
    # 1. ARRANGE (DataFrame crudo)
    df = raw_df

    # 2. ACT (Aplicar el preprocesamiento)
    procesado = preprocess_features(df)

    # 3. ASSERT (Las columnas dummy son float, no bool ni int)
    for col in ["Type_H", "Type_L", "Type_M"]:
        assert procesado[col].dtype == float


def test_target_sigue_siendo_binario(raw_df):
    # 1. ARRANGE (DataFrame crudo con target 0/1)
    df = raw_df

    # 2. ACT (Aplicar el preprocesamiento)
    procesado = preprocess_features(df)

    # 3. ASSERT (El target solo contiene 0 y 1 tras el preprocesamiento)
    assert set(procesado["Machine failure"].unique()) <= {0, 1}


def test_no_filas_adicionales_ni_perdidas(raw_df):
    # 1. ARRANGE (DataFrame crudo de 10000 filas)
    df = raw_df
    n_filas_original = len(df)

    # 2. ACT (Aplicar el preprocesamiento)
    procesado = preprocess_features(df)

    # 3. ASSERT (El preprocesamiento no agrega ni elimina filas)
    assert len(procesado) == n_filas_original
