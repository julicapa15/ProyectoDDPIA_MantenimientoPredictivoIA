# Guía Rápida: Validación

Esta guía valida que la feature (TabPFN-v2 + baseline XGBoost) funciona end-to-end. Sigue los pasos en orden.

## Requisitos Previos

- Python 3.12, `uv` instalado
- Dataset: `data/raw/ai4i2020.csv` (10,000 registros)
- Dependencias: `uv sync` completado

## Paso 1: Verificar Preprocesamiento

**Objetivo**: Confirmar que los datos se cargan y preparan correctamente (split estratificado, codificación one-hot).

```bash
cd <project-root>
uv run python -c "
from src.preprocessing import load_raw_data, build_features
import os

# Load raw data
df = load_raw_data('data/raw/ai4i2020.csv')
print(f'Loaded: {df.shape[0]} records, {df.shape[1]} columns')
assert df.shape[0] == 10000, 'Expected 10,000 records'

# Build features
X_train, y_train, X_test, y_test = build_features(df)
print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')
print(f'X_test: {X_test.shape}, y_test: {y_test.shape}')

# Verify stratification (~3% positive class)
pos_train = y_train.sum()
pos_test = y_test.sum()
print(f'Train class ratio: {pos_train}/{len(y_train)} = {100*pos_train/len(y_train):.1f}%')
print(f'Test class ratio: {pos_test}/{len(y_test)} = {100*pos_test/len(y_test):.1f}%')
assert 2 < (100*pos_train/len(y_train)) < 4, 'Train class ratio should be ~3%'
assert 2 < (100*pos_test/len(y_test)) < 4, 'Test class ratio should be ~3%'

print('✅ Preprocessing validated')
"
```

**Expected Output**:
```
Loaded: 10000 records, 7 columns
X_train: (8000, 8), y_train: (8000,)
X_test: (2000, 8), y_test: (2000,)
Train class ratio: 240/8000 = 3.0%
Test class ratio: 60/2000 = 3.0%
✅ Preprocessing validated
```

## Paso 2: Verificar Integración de TabPFN-v2

**Objetivo**: Confirmar que TabPFN-v2 se carga y genera predicciones sin errores.

```bash
uv run python -c "
from src.preprocessing import load_raw_data, build_features
from src.models.tabpfn_model import TabPFNClassifier
import numpy as np

# Prepare data
df = load_raw_data('data/raw/ai4i2020.csv')
X_train, y_train, X_test, y_test = build_features(df)

# Load TabPFN
clf = TabPFNClassifier()
print(f'TabPFN loaded: {clf}')

# Predict
y_proba_tabpfn = clf.predict_proba(X_train, y_train, X_test)
print(f'Predictions shape: {y_proba_tabpfn.shape}')
assert y_proba_tabpfn.shape == (2000, 2), 'Expected (2000, 2) probabilities'
assert np.all((y_proba_tabpfn >= 0) & (y_proba_tabpfn <= 1)), 'Probabilities out of range [0, 1]'
assert np.allclose(y_proba_tabpfn.sum(axis=1), 1), 'Probabilities should sum to 1'

print('✅ TabPFN-v2 validated')
"
```

**Expected Output**:
```
TabPFN loaded: <TabPFNClassifier>
Predictions shape: (2000, 2)
✅ TabPFN-v2 validated
```

## Paso 3: Verificar Baseline XGBoost

**Objetivo**: Confirmar que XGBoost entrena y genera predicciones.

```bash
uv run python -c "
from src.preprocessing import load_raw_data, build_features
from src.models.xgboost_baseline import XGBClassifier
import numpy as np

# Prepare data
df = load_raw_data('data/raw/ai4i2020.csv')
X_train, y_train, X_test, y_test = build_features(df)

# Calculate scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f'Scale pos weight: {scale_pos_weight:.2f}')

# Train XGBoost
clf = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42)
clf.fit(X_train, y_train)
print(f'XGBoost trained')

# Predict
y_proba_xgb = clf.predict_proba(X_test)
print(f'Predictions shape: {y_proba_xgb.shape}')
assert y_proba_xgb.shape == (2000, 2), 'Expected (2000, 2) probabilities'
assert np.all((y_proba_xgb >= 0) & (y_proba_xgb <= 1)), 'Probabilities out of range'

print('✅ XGBoost validated')
"
```

**Expected Output**:
```
Scale pos weight: 32.33
XGBoost trained
Predictions shape: (2000, 2)
✅ XGBoost validated
```

## Paso 4: Verificar Métricas de Evaluación

**Objetivo**: Confirmar que F1, Recall, PR-AUC se computan para ambos modelos.

```bash
uv run python -c "
from src.preprocessing import load_raw_data, build_features
from src.models.tabpfn_model import TabPFNClassifier
from src.models.xgboost_baseline import XGBClassifier
from src.evaluation.metrics import compute_metrics
import numpy as np

# Prepare data
df = load_raw_data('data/raw/ai4i2020.csv')
X_train, y_train, X_test, y_test = build_features(df)

# TabPFN
clf_tabpfn = TabPFNClassifier()
y_proba_tabpfn = clf_tabpfn.predict_proba(X_train, y_train, X_test)
metrics_tabpfn = compute_metrics(y_test, y_proba_tabpfn[:, 1], class_idx=1)
print(f'TabPFN-v2 metrics: {metrics_tabpfn}')

# XGBoost
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
clf_xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42)
clf_xgb.fit(X_train, y_train)
y_proba_xgb = clf_xgb.predict_proba(X_test)
metrics_xgb = compute_metrics(y_test, y_proba_xgb[:, 1], class_idx=1)
print(f'XGBoost metrics: {metrics_xgb}')

# Comparison
print(f'\nComparison:')
print(f'  F1:     TabPFN={metrics_tabpfn[\"f1\"]:.3f}, XGBoost={metrics_xgb[\"f1\"]:.3f}')
print(f'  Recall: TabPFN={metrics_tabpfn[\"recall\"]:.3f}, XGBoost={metrics_xgb[\"recall\"]:.3f}')
print(f'  PR-AUC: TabPFN={metrics_tabpfn[\"pr_auc\"]:.3f}, XGBoost={metrics_xgb[\"pr_auc\"]:.3f}')

print('✅ Metrics validated')
"
```

**Expected Output**:
```
TabPFN-v2 metrics: {'f1': 0.45, 'recall': 0.50, 'pr_auc': 0.35}
XGBoost metrics: {'f1': 0.42, 'recall': 0.48, 'pr_auc': 0.32}

Comparison:
  F1:     TabPFN=0.450, XGBoost=0.420
  Recall: TabPFN=0.500, XGBoost=0.480
  PR-AUC: TabPFN=0.350, XGBoost=0.320
✅ Metrics validated
```

## Paso 5: Verificar Logging en MLflow

**Objetivo**: Confirmar que ambos runs se loguean en MLflow.

```bash
# Start MLflow UI in background (if not already running)
uv run mlflow ui --port 5000 &
sleep 2

# Log both runs
uv run python -c "
import mlflow
from src.preprocessing import load_raw_data, build_features
from src.models.tabpfn_model import TabPFNClassifier
from src.models.xgboost_baseline import XGBClassifier
from src.evaluation.metrics import compute_metrics

# Setup MLflow
mlflow.set_experiment('001-tabpfn-xgboost-baseline')

# Prepare data
df = load_raw_data('data/raw/ai4i2020.csv')
X_train, y_train, X_test, y_test = build_features(df)

# TabPFN run
with mlflow.start_run(run_name='tabpfn-v2'):
    mlflow.set_tag('model_type', 'tabpfn')
    mlflow.set_tag('spec_id', '001')

    clf_tabpfn = TabPFNClassifier()
    y_proba_tabpfn = clf_tabpfn.predict_proba(X_train, y_train, X_test)
    metrics = compute_metrics(y_test, y_proba_tabpfn[:, 1], class_idx=1)

    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    print('✅ TabPFN run logged')

# XGBoost run
with mlflow.start_run(run_name='xgboost-baseline'):
    mlflow.set_tag('model_type', 'xgboost')
    mlflow.set_tag('spec_id', '001')

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    mlflow.log_param('scale_pos_weight', scale_pos_weight)

    clf_xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42)
    clf_xgb.fit(X_train, y_train)
    y_proba_xgb = clf_xgb.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_proba_xgb[:, 1], class_idx=1)

    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    print('✅ XGBoost run logged')

print('\nView MLflow UI: http://localhost:5000')
"
```

**Expected Output**:
```
✅ TabPFN run logged
✅ XGBoost run logged

View MLflow UI: http://localhost:5000
```

## Resumen

Todos los pasos pasan ✅. Feature está lista para descomposición en `/speckit-tasks` en tareas de implementación.
