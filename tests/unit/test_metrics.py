"""T025 [US4] Tests unitarios para compute_metrics()."""

import numpy as np
import pytest

from src.evaluation import compute_metrics

METRICAS_ESPERADAS = {"f1", "recall", "pr_auc", "accuracy"}


def test_retorna_las_cuatro_metricas():
    # 1. ARRANGE (Etiquetas reales y probabilidades predichas)
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])

    # 2. ACT (Calcular las métricas)
    metricas = compute_metrics(y_true, y_proba)

    # 3. ASSERT (Exactamente F1, Recall, PR-AUC y accuracy)
    assert set(metricas) == METRICAS_ESPERADAS


def test_accuracy_no_es_engañosamente_alta_con_desbalance():
    # 1. ARRANGE (Un modelo que nunca predice falla, con fuerte desbalance)
    rng = np.random.default_rng(7)
    y_true = (rng.random(200) < 0.05).astype(int)
    y_proba = np.zeros(200)

    # 2. ACT (Calcular las métricas)
    metricas = compute_metrics(y_true, y_proba)

    # 3. ASSERT (Accuracy es alta pese a que el modelo es inútil: por eso no es criterio de éxito)
    assert metricas["accuracy"] > 0.9
    assert metricas["recall"] == 0.0
    assert metricas["f1"] == 0.0


def test_prediccion_perfecta():
    # 1. ARRANGE (Probabilidades que separan perfectamente ambas clases)
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.0, 0.1, 0.9, 1.0])

    # 2. ACT (Calcular las métricas)
    metricas = compute_metrics(y_true, y_proba)

    # 3. ASSERT (Las tres métricas alcanzan su valor máximo)
    assert metricas["f1"] == pytest.approx(1.0)
    assert metricas["recall"] == pytest.approx(1.0)
    assert metricas["pr_auc"] == pytest.approx(1.0)


def test_modelo_que_nunca_predice_falla():
    # 1. ARRANGE (Probabilidades siempre por debajo del umbral)
    y_true = np.array([0, 0, 0, 1])
    y_proba = np.array([0.01, 0.02, 0.03, 0.04])

    # 2. ACT (Calcular las métricas)
    metricas = compute_metrics(y_true, y_proba)

    # 3. ASSERT (Un modelo que nunca alerta no tiene mérito, pese a acertar el 75%)
    assert metricas["recall"] == 0.0
    assert metricas["f1"] == 0.0


def test_recall_se_calcula_sobre_la_clase_falla():
    # 1. ARRANGE (Dos fallos reales, de los que el modelo solo detecta uno)
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.1, 0.9, 0.2])

    # 2. ACT (Calcular las métricas)
    metricas = compute_metrics(y_true, y_proba)

    # 3. ASSERT (Detecta 1 de 2 fallos)
    assert metricas["recall"] == pytest.approx(0.5)


def test_todas_las_metricas_en_rango_cero_uno():
    # 1. ARRANGE (Datos aleatorios con desbalance similar al real)
    rng = np.random.default_rng(7)
    y_true = (rng.random(200) < 0.05).astype(int)
    y_proba = rng.random(200)

    # 2. ACT (Calcular las métricas)
    metricas = compute_metrics(y_true, y_proba)

    # 3. ASSERT (Ninguna métrica se sale de [0, 1])
    assert all(0.0 <= valor <= 1.0 for valor in metricas.values())


def test_accuracy_cuenta_aciertos_totales():
    # 1. ARRANGE (Cuatro muestras, tres clasificadas correctamente)
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.3])

    # 2. ACT (Calcular las métricas)
    metricas = compute_metrics(y_true, y_proba)

    # 3. ASSERT (3 de 4 aciertos: 0.75)
    assert metricas["accuracy"] == pytest.approx(0.75)


def test_umbral_configurable():
    # 1. ARRANGE (Un fallo real con probabilidad predicha de 0.4)
    y_true = np.array([0, 1])
    y_proba = np.array([0.3, 0.4])

    # 2. ACT (Evaluar con el umbral por defecto y con uno más sensible)
    con_umbral_alto = compute_metrics(y_true, y_proba, threshold=0.5)
    con_umbral_bajo = compute_metrics(y_true, y_proba, threshold=0.35)

    # 3. ASSERT (Bajar el umbral recupera el fallo que se escapaba)
    assert con_umbral_alto["recall"] == 0.0
    assert con_umbral_bajo["recall"] == 1.0


def test_f1_se_anula_cuando_no_hay_prediccion_positiva():
    # 1. ARRANGE (Probabilidades todas por debajo del umbral por defecto)
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4])

    # 2. ACT (Calcular las métricas)
    metricas = compute_metrics(y_true, y_proba)

    # 3. ASSERT (Sin predicciones positivas, F1 es cero)
    assert metricas["f1"] == 0.0


def test_pr_auc_penaliza_probabilidades_invertidas():
    # 1. ARRANGE (Probabilidades invertidas: alta para no-falla, baja para falla)
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.9, 0.8, 0.1, 0.2])

    # 2. ACT (Calcular las métricas)
    metricas = compute_metrics(y_true, y_proba)

    # 3. ASSERT (PR-AUC por debajo de 0.5 porque el modelo invierte las clases)
    assert metricas["pr_auc"] < 0.5


def test_accuracy_paralela_con_f1_bajo():
    # 1. ARRANGE (Modelo que predice todo como no-falla con 50/50 clases)
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.0, 0.0, 0.0, 0.0])

    # 2. ACT (Calcular las métricas)
    metricas = compute_metrics(y_true, y_proba)

    # 3. ASSERT (Accuracy es 0.5 pero F1 es 0 porque nunca detecta falla)
    assert metricas["accuracy"] == pytest.approx(0.5)
    assert metricas["f1"] == 0.0


def test_recall_maximo_con_todo_sobre_umbral():
    # 1. ARRANGE (Todos los casos de falla tienen alta probabilidad)
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.95])

    # 2. ACT (Calcular las métricas)
    metricas = compute_metrics(y_true, y_proba)

    # 3. ASSERT (Todos los fallos detectados)
    assert metricas["recall"] == pytest.approx(1.0)


def test_metricas_cero_muestras_suficientes():
    # 1. ARRANGE (Una sola falla real, bien detectada)
    y_true = np.array([0, 1])
    y_proba = np.array([0.1, 0.9])

    # 2. ACT (Calcular las métricas)
    metricas = compute_metrics(y_true, y_proba)

    # 3. ASSERT (Con solo 2 muestras, las métricas son validas en [0, 1])
    assert all(0.0 <= v <= 1.0 for v in metricas.values())
