"""Tests unitarios para las funciones auxiliares de tracking en MLflow."""

from src.evaluation.tracking import (
    EXTENSION_MODELO,
    LICENCIA_MODELO,
    NOMBRES_DE_RUN,
    REPO_MODELO,
    _detectar_device,
    _hash_dataset,
)


def test_hash_dataset_none_retorna_none():
    # 1. ARRANGE (Ruta None)

    # 2. ACT (Calcular hash con ruta None)
    resultado = _hash_dataset(None)

    # 3. ASSERT (Debe retornar None)
    assert resultado is None


def test_hash_dataset_archivo_inexistente_retorna_none(tmp_path):
    # 1. ARRANGE (Ruta a un archivo que no existe en disco)
    ruta_inexistente = str(tmp_path / "no_existe.csv")

    # 2. ACT (Calcular hash de un archivo inexistente)
    resultado = _hash_dataset(ruta_inexistente)

    # 3. ASSERT (Debe retornar None para archivos fantasma)
    assert resultado is None


def test_hash_dataset_retorna_12caracteres(tmp_path):
    # 1. ARRANGE (Un archivo de texto plano con contenido conocido)
    ruta_archivo = tmp_path / "dataset.csv"
    ruta_archivo.write_text("col1,col2\n1,2\n3,4\n", encoding="utf-8")

    # 2. ACT (Calcular el hash del archivo)
    resultado = _hash_dataset(str(ruta_archivo))

    # 3. ASSERT (El hash tiene exactamente 12 caracteres hexadecimales)
    assert len(resultado) == 12
    assert all(c in "0123456789abcdef" for c in resultado)


def test_hash_dataset_es_reproducible(tmp_path):
    # 1. ARRANGE (Mismo archivo con contenido fijo)
    ruta_archivo = tmp_path / "dataset.csv"
    ruta_archivo.write_text("a,b\n1,2\n", encoding="utf-8")

    # 2. ACT (Calcular el hash dos veces)
    hash_1 = _hash_dataset(str(ruta_archivo))
    hash_2 = _hash_dataset(str(ruta_archivo))

    # 3. ASSERT (El mismo archivo siempre produce el mismo hash)
    assert hash_1 == hash_2


def test_hash_dataset_archivos_diferentes_hashes_diferentes(tmp_path):
    # 1. ARRANGE (Dos archivos con contenido distinto)
    ruta_a = tmp_path / "a.csv"
    ruta_b = tmp_path / "b.csv"
    ruta_a.write_text("x,y\n1,2\n", encoding="utf-8")
    ruta_b.write_text("x,y\n3,4\n", encoding="utf-8")

    # 2. ACT (Calcular el hash de cada uno)
    hash_a = _hash_dataset(str(ruta_a))
    hash_b = _hash_dataset(str(ruta_b))

    # 3. ASSERT (Contenido distinto produce hashes distintos)
    assert hash_a != hash_b


def test_detectar_device_retorna_cpu_o_cuda():
    # 1. ARRANGE (Sin preparación)

    # 2. ACT (Detectar el backend de cómputo disponible)
    device = _detectar_device()

    # 3. ASSERT (Debe ser "cpu" o "cuda", no un valor inesperado)
    assert device in ("cpu", "cuda")


def test_nombres_de_run_contiene_modelos_esperados():
    # 1. ARRANGE (El diccionario de nombres de run)

    # 2. ACT (Verificar las claves del diccionario)
    modelos = set(NOMBRES_DE_RUN.keys())

    # 3. ASSERT (Ambos modelos del proyecto están registrados)
    assert "tabpfn" in modelos
    assert "xgboost" in modelos


def test_extENSION_modelo_y_licencia_cubren_ambos_modelos():
    # 1. ARRANGE (Los diccionarios de extensión y licencia)

    # 2. ACT (Verificar que ambas claves existen en ambos diccionarios)
    modelos = {"tabpfn", "xgboost"}

    # 3. ASSERT (Cada modelo tiene extensión y licencia definida)
    for modelo in modelos:
        assert modelo in EXTENSION_MODELO
        assert modelo in LICENCIA_MODELO
        assert modelo in REPO_MODELO
