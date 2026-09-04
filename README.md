# Mantenimiento Predictivo con TabPFN-v2

![Estado](https://img.shields.io/badge/estado-en%20desarrollo-yellow)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Gestor](https://img.shields.io/badge/dependencias-uv-purple)
![Licencia](https://img.shields.io/badge/uso-acad%C3%A9mico-lightgrey)

Sistema de **mantenimiento predictivo** para maquinaria industrial que estima la
probabilidad de fallo de un equipo a partir de sus variables operativas (temperatura,
velocidad de rotación, torque, desgaste de herramienta, etc.).

El modelo base es **TabPFN-v2**, un transformer preentrenado para clasificación
tabular que no requiere entrenamiento tradicional. Como *baselines* opcionales se
contemplan **XGBoost**, **Gradient Boosting** y **TabPFN-Mix**. La solución se
expone mediante una interfaz web en **Streamlit** y se empaqueta con **Docker**.

---

## Integrantes

| Integrante                | 
|---------------------------|
| Juliana Campuzano         |
| Diego Fernando Garcés     |
| Jorge Iván Gonzáles       |
| Juan José Mosquera        |

**Universidad:** Universidad Autónoma de Occidente
**Curso:** Desarrollo de Proyectos en IA — Módulo 2

---

## Tecnologías

| Componente            | Herramienta                                   |
|-----------------------|-----------------------------------------------|
| Modelo base           | TabPFN-v2                                      |
| Baselines opcionales  | XGBoost · Gradient Boosting · TabPFN-Mix       |
| Dataset               | AI4I 2020 Predictive Maintenance Dataset (UCI) |
| Procesamiento         | pandas · NumPy · scikit-learn                  |
| Visualización         | matplotlib                                     |
| Interfaz              | Streamlit                                      |
| Contenedor            | Docker *(pendiente — Módulo 3)*               |
| Gestión de entorno    | uv                                             |
| Pruebas               | pytest                                         |
| Seguimiento de experimentos | MLflow *(previsto)*                      |

---

## Estructura del proyecto

```
.
├── data/
│   ├── raw/            # Datos originales sin procesar (AI4I 2020)
│   └── processed/      # Datos limpios y transformados
├── notebooks/          # Exploración y prototipado
├── src/
│   ├── preprocessing/  # Limpieza, ingeniería de características, splits
│   ├── models/         # TabPFN-v2 y baselines
│   └── evaluation/     # Métricas, curvas, comparativas
├── app/                # Interfaz Streamlit
├── docker/             # Recursos de contenedorización
├── tests/              # Pruebas con pytest
├── docs/               # Documentación del proyecto
├── Dockerfile          # Placeholder (Módulo 3)
├── pyproject.toml      # Dependencias (uv)
└── README.md
```

---

## Instalación

Requisitos: [uv](https://docs.astral.sh/uv/) y Python 3.12.

```bash
# 1. Clonar el repositorio
git clone https://github.com/julicapa15/ProyectoDDPIA_MantenimientoPredictivo.git
cd ProyectoDDPIA_MantenimientoPredictivo

# 2. Crear el entorno e instalar dependencias
uv sync

# 3. (Opcional) Activar el entorno
#    Windows:  .venv\Scripts\activate
#    Linux/Mac: source .venv/bin/activate
```

Para añadir nuevas dependencias:

```bash
uv add <paquete>
```

---

## Ejecución de la aplicación

```bash
uv run streamlit run app/app.py
```

La interfaz quedará disponible en `http://localhost:8501`.

---

## Ejecución con Docker

> **Pendiente.** La contenedorización se implementará en el **Módulo 3**.
> El `Dockerfile` actual es un placeholder.

```bash
# Previsto para el Módulo 3
# docker build -t mantenimiento-predictivo .
# docker run -p 8501:8501 mantenimiento-predictivo
```

---

## Pruebas

```bash
uv run pytest
```

---

## Estado del proyecto

🚧 **En desarrollo — Módulo 2.**

| Fase                                      | Estado        |
|-------------------------------------------|---------------|
| Estructura del repositorio                | ✅ Completado |
| Carga y exploración del dataset AI4I 2020 | ⏳ En curso   |
| Preprocesamiento y *feature engineering*  | ⏳ Pendiente  |
| Modelo base TabPFN-v2                     | ⏳ Pendiente  |
| Baselines (XGBoost / GB / TabPFN-Mix)     | ⏳ Pendiente  |
| Evaluación y comparativa                  | ⏳ Pendiente  |
| Interfaz Streamlit                        | ⏳ Pendiente  |
| Contenedorización con Docker              | ⏳ Módulo 3   |
