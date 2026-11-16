"""T011 [US1] Tests unitarios para build_features()."""

import numpy as np

from src.preprocessing import build_features


def test_shapes_de_los_splits(raw_df):
    # 1. ARRANGE (DataFrame crudo de 10.000 registros)
    df = raw_df

    # 2. ACT (Construir los conjuntos de entrenamiento y prueba)
    X_train, y_train, X_test, y_test = build_features(df)

    # 3. ASSERT (Split 80/20 con 8 features)
    assert X_train.shape == (8000, 8)
    assert y_train.shape == (8000,)
    assert X_test.shape == (2000, 8)
    assert y_test.shape == (2000,)


def test_tipos_de_salida(split_data):
    # 1. ARRANGE (Conjuntos ya construidos)
    X_train, y_train, X_test, y_test = split_data

    # 2. ACT (Inspeccionar tipo y valores únicos del target)
    valores_y = set(np.unique(y_train)) | set(np.unique(y_test))

    # 3. ASSERT (X como matrices numpy, y binaria)
    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert valores_y <= {0, 1}


def test_proporcion_de_fallos_en_train(split_data):
    # 1. ARRANGE (Etiquetas del conjunto de entrenamiento)
    _, y_train, _, _ = split_data

    # 2. ACT (Calcular el porcentaje de la clase falla)
    ratio = 100 * y_train.sum() / len(y_train)

    # 3. ASSERT (Debe conservar el ~3,4% del dataset original)
    assert 2 < ratio < 4, f"Proporción de fallos en train fuera de rango: {ratio:.2f}%"


def test_proporcion_de_fallos_en_test(split_data):
    # 1. ARRANGE (Etiquetas del conjunto de prueba)
    _, _, _, y_test = split_data

    # 2. ACT (Calcular el porcentaje de la clase falla)
    ratio = 100 * y_test.sum() / len(y_test)

    # 3. ASSERT (Debe conservar el ~3,4% del dataset original)
    assert 2 < ratio < 4, f"Proporción de fallos en test fuera de rango: {ratio:.2f}%"


def test_estratificacion_preserva_la_misma_proporcion(split_data):
    # 1. ARRANGE (Etiquetas de ambos conjuntos)
    _, y_train, _, y_test = split_data

    # 2. ACT (Comparar las proporciones de la clase falla)
    diferencia = abs(y_train.sum() / len(y_train) - y_test.sum() / len(y_test))

    # 3. ASSERT (La estratificación las mantiene casi idénticas)
    assert diferencia < 0.01


def test_sin_perdida_de_registros(split_data):
    # 1. ARRANGE (Matrices de ambos conjuntos)
    X_train, _, X_test, _ = split_data

    # 2. ACT (Sumar los registros de train y test)
    total = len(X_train) + len(X_test)

    # 3. ASSERT (El split reparte los 10.000 registros sin duplicar ni perder)
    assert total == 10000


def test_es_reproducible(raw_df):
    # 1. ARRANGE (Mismo DataFrame de entrada para dos ejecuciones)
    df = raw_df

    # 2. ACT (Construir los splits dos veces)
    X_train_a, y_train_a, _, _ = build_features(df)
    X_train_b, y_train_b, _, _ = build_features(df)

    # 3. ASSERT (El random_state fijo garantiza el mismo resultado)
    assert np.array_equal(X_train_a, X_train_b)
    assert np.array_equal(y_train_a, y_train_b)


def test_features_no_tienen_nan(split_data):
    # 1. ARRANGE (Matrices de entrenamiento y prueba)
    X_train, y_train, X_test, y_test = split_data

    # 2. ACT (Verificar si hay valores NaN en las matrices)
    nan_en_train = np.isnan(X_train).any()
    nan_en_test = np.isnan(X_test).any()

    # 3. ASSERT (No debe haber valores faltantes en los features)
    assert not nan_en_train
    assert not nan_en_test


def test_y_son_enteros(split_data):
    # 1. ARRANGE (Etiquetas de entrenamiento y prueba)
    _, y_train, _, y_test = split_data

    # 2. ACT (Verificar el tipo de dato de las etiquetas)
    tipo_train = y_train.dtype
    tipo_test = y_test.dtype

    # 3. ASSERT (Las etiquetas deben ser enteras)
    assert np.issubdtype(tipo_train, np.integer)
    assert np.issubdtype(tipo_test, np.integer)


def test_train_y_test_sin_superposicion(split_data):
    # 1. ARRANGE (Matrices de entrenamiento y prueba)
    X_train, _, X_test, _ = split_data

    # 2. ACT (Convertir filas a conjuntos de tuplas para comparar)
    conjunto_train = set(map(tuple, X_train))
    conjunto_test = set(map(tuple, X_test))

    # 3. ASSERT (No hay filas idénticas en ambos conjuntos)
    interseccion = conjunto_train & conjunto_test
    assert len(interseccion) == 0


def test_valores_min_max_coherentes(split_data):
    # 1. ARRANGE (Matrices de entrenamiento y prueba)
    X_train, _, X_test, _ = split_data

    # 2. ACT (Calcular min y max globales)
    minimo = min(X_train.min(), X_test.min())
    maximo = max(X_train.max(), X_test.max())

    # 3. ASSERT (Los valores están en rangos físicamente posibles, rpm puede llegar a ~2886)
    assert minimo >= -1.0
    assert maximo < 5000.0
