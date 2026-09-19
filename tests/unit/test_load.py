"""T009 [US1] Tests unitarios para load_raw_data()."""

import pandas as pd
import pytest

from src.preprocessing import load_raw_data

COLUMNAS_ESPERADAS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def test_retorna_dataframe_con_10000_registros(dataset_path):
    # 1. ARRANGE (Ruta al CSV crudo de AI4I 2020)
    ruta = dataset_path

    # 2. ACT (Cargar el dataset)
    df = load_raw_data(ruta)

    # 3. ASSERT (Verificar tipo y número de registros)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 10000


def test_columnas_esperadas_existen(dataset_path):
    # 1. ARRANGE (Ruta al CSV crudo)
    ruta = dataset_path

    # 2. ACT (Cargar el dataset)
    df = load_raw_data(ruta)

    # 3. ASSERT (Verificar que están las 14 columnas originales en orden)
    assert list(df.columns) == COLUMNAS_ESPERADAS


def test_sin_valores_faltantes(dataset_path):
    # 1. ARRANGE (Ruta al CSV crudo)
    ruta = dataset_path

    # 2. ACT (Cargar el dataset y contar nulos)
    n_nulos = load_raw_data(ruta).isna().sum().sum()

    # 3. ASSERT (El dataset AI4I 2020 se asume limpio)
    assert n_nulos == 0


def test_target_es_binario(dataset_path):
    # 1. ARRANGE (Ruta al CSV crudo)
    ruta = dataset_path

    # 2. ACT (Cargar y extraer los valores únicos del target)
    valores = set(load_raw_data(ruta)["Machine failure"].unique())

    # 3. ASSERT (Solo debe contener 0 y 1)
    assert valores <= {0, 1}


def test_type_tiene_tres_categorias(dataset_path):
    # 1. ARRANGE (Ruta al CSV crudo)
    ruta = dataset_path

    # 2. ACT (Cargar y extraer las categorías de Type)
    categorias = set(load_raw_data(ruta)["Type"].unique())

    # 3. ASSERT (Los tres tipos de producto documentados)
    assert categorias == {"L", "M", "H"}


def test_archivo_inexistente_lanza_error(tmp_path):
    # 1. ARRANGE (Ruta a un archivo que no existe)
    ruta = tmp_path / "no_existe.csv"

    # 2. ACT (Intentar cargar un archivo que no está en disco)
    with pytest.raises(FileNotFoundError) as excinfo:
        load_raw_data(ruta)

    # 3. ASSERT (El mensaje nombra la ruta que falta, para poder diagnosticarlo)
    assert "no_existe.csv" in str(excinfo.value)


def test_dataset_con_nulos_lanza_error(dataset_path, tmp_path):
    # 1. ARRANGE (Copia del dataset con un valor faltante inyectado)
    df = load_raw_data(dataset_path)
    df.loc[0, "Torque [Nm]"] = None
    ruta_corrupta = tmp_path / "con_nulos.csv"
    df.to_csv(ruta_corrupta, index=False)

    # 2. ACT (Intentar cargar el dataset con el nulo inyectado)
    with pytest.raises(ValueError) as excinfo:
        load_raw_data(ruta_corrupta)

    # 3. ASSERT (El error explica que hay valores faltantes por resolver)
    assert "valores faltantes" in str(excinfo.value)


def test_columnas_faltantes_lanza_error(dataset_path, tmp_path):
    # 1. ARRANGE (Copia del dataset sin la columna 'Torque [Nm]')
    df = load_raw_data(dataset_path)
    df = df.drop(columns=["Torque [Nm]"])
    ruta_incompleta = tmp_path / "sin_torque.csv"
    df.to_csv(ruta_incompleta, index=False)

    # 2. ACT (Intentar cargar el dataset con una columna eliminada)
    with pytest.raises(ValueError) as excinfo:
        load_raw_data(ruta_incompleta)

    # 3. ASSERT (El error lista la columna ausente)
    assert "Torque [Nm]" in str(excinfo.value)


def test_registros_insuficientes_lanza_error(tmp_path):
    # 1. ARRANGE (Un CSV con solo 10 filas y todas las columnas requeridas)
    columnas = [
        "UDI",
        "Product ID",
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]
    df_parcial = pd.DataFrame({col: [0] * 10 for col in columnas})
    ruta_parcial = tmp_path / "parcial.csv"
    df_parcial.to_csv(ruta_parcial, index=False)

    # 2. ACT (Intentar cargar un dataset con menos registros de los esperados)
    with pytest.raises(ValueError) as excinfo:
        load_raw_data(ruta_parcial)

    # 3. ASSERT (El error indica la cantidad esperada vs la encontrada)
    assert "10000" in str(excinfo.value)
    assert "10" in str(excinfo.value)


def test_dataset_vacio_lanza_error(tmp_path):
    # 1. ARRANGE (Un CSV con solo encabezados y cero filas)
    columnas = [
        "UDI",
        "Product ID",
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]
    df_vacio = pd.DataFrame(columns=columnas)
    ruta_vacia = tmp_path / "vacio.csv"
    df_vacio.to_csv(ruta_vacia, index=False)

    # 2. ACT (Intentar cargar un dataset vacío)
    with pytest.raises(ValueError) as excinfo:
        load_raw_data(ruta_vacia)

    # 3. ASSERT (El error indica que no se encontraron los 10000 registros)
    assert "10000" in str(excinfo.value)


def test_csv_con_columnas_extra_se_carga_correctamente(dataset_path, tmp_path):
    # 1. ARRANGE (Copia del dataset con una columna extra no requerida)
    df = load_raw_data(dataset_path)
    df["extra_col"] = 999
    ruta_con_extra = tmp_path / "con_extra.csv"
    df.to_csv(ruta_con_extra, index=False)

    # 2. ACT (Cargar el CSV con la columna adicional)
    resultado = load_raw_data(ruta_con_extra)

    # 3. ASSERT (La columna extra no aparece en el resultado, solo las 14 requeridas)
    assert "extra_col" not in resultado.columns
    assert resultado.shape == (10000, 14)


def test_dtype_target_es_entero(dataset_path):
    # 1. ARRANGE (Ruta al CSV crudo)
    ruta = dataset_path

    # 2. ACT (Cargar el dataset y revisar el tipo de la columna target)
    df = load_raw_data(ruta)

    # 3. ASSERT (Machine failure debe ser numérico entero 0/1)
    assert df["Machine failure"].dtype in ("int64", "int32")
