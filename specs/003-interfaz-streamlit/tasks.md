# Tareas: Interfaz Web Streamlit con TabPFN-v2

**Entrada**: Especificación desde `specs/003-interfaz-streamlit/spec.md` y `plan.md`

## Formato: `[ID] [Story] Descripción`

---

## Phase 1: Setup y Creación de Estructura de Módulos

- [X] T001 Crear carpeta `specs/003-interfaz-streamlit/` con `spec.md`, `plan.md` y `tasks.md`
- [X] T002 Crear carpetas `app/components/` y `app/utils/`

## Phase 2: Extracción de Utilidades de Dominio (Funciones Puras)

- [X] T003 Extraer funciones físicas (`calculate_power`) a `app/utils/physics.py` con docstrings Google y tests
- [X] T004 Extraer `preprocess_input` y `get_alert_level` a `app/utils/preprocessing.py`
- [X] T005 Extraer la carga de datos y TabPFN con `@st.cache_resource` a `app/utils/model.py`

## Phase 3: Creación de Componentes Visuales

- [X] T006 Crear componente `app/components/form.py` para encapsular la captura de las 6 variables
- [X] T007 Crear componente `app/components/results.py` para renderizar el semáforo, métricas y expanders
- [X] T008 Actualizar `app/main.py` para orquestar los componentes en un archivo limpio y legible (< 60 líneas)

## Phase 4: Pruebas y Validación

- [X] T009 Actualizar imports en `tests/unit/test_app.py` apuntando a los nuevos submódulos
- [X] T010 Ejecutar `uv run pytest tests/unit/test_app.py` asegurando paso de todos los tests
- [X] T011 Ejecutar `uv run ruff check .` y `uv run ruff format .` garantizando 0 errores
- [X] T012 Verificar ejecución local interactiva con `uv run streamlit run app/main.py`
  ([make app](../../Makefile) levanta la app y responde). Esta tarea quedó sin hacer antes
  del merge original de la feature: los 21 tests de `tests/unit/test_app.py` pasaban porque
  ejercitan `app/utils/`, `app/components/` como módulos puros importados por pytest, sin
  pasar por el entrypoint real de Streamlit. Al no correr `streamlit run app/main.py`
  interactivamente antes del merge, el `ModuleNotFoundError` causado por el `sys.path` sin
  ajustar (ver T013) llegó a `develop` sin que ningún test lo detectara.

## Phase 5: Fix del `sys.path` y consolidación del preprocesamiento (rama `fix/streamlit-syspath-jigg`)

- [X] T013 Corregir `ModuleNotFoundError` en `app/main.py`: Streamlit ejecuta el archivo
  directamente (no como parte del paquete `app`), así que la raíz del proyecto no está en
  `sys.path` cuando se resuelven los imports `app.*` / `src.*`. Se agrega
  `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` antes de esos imports.
  `pytest` nunca lo detectó porque `pyproject.toml` ya fija `pythonpath = ["."]` en
  `[tool.pytest.ini_options]`, así que los 21 tests de `tests/unit/test_app.py` importan
  `app.*`/`src.*` con la raíz del proyecto ya en el path por configuración de pytest, sin
  pasar por el arranque real de Streamlit — el bug solo era visible corriendo la app de
  verdad (T012).
- [X] T014 Eliminar la duplicación a mano del orden de features en
  `app/utils/preprocessing.py`: `NOMBRE_FEATURES` ahora se deriva de
  `COLUMNAS_DUMMY + COLUMNAS_NUMERICAS`, importadas de
  `src/preprocessing/preprocess.py`, en vez de repetir la lista de 8 columnas de forma
  independiente. Deja una sola fuente de verdad para el orden que usan tanto el pipeline de
  entrenamiento (`src/`) como el formulario de la app.
- [X] T015 Agregar test de equivalencia parametrizado (`L`/`M`/`H`) en
  `tests/unit/test_app.py` que compara, para una misma fila cruda concreta, el vector de
  8 valores que produce `preprocess_input()` (app) contra el que produce
  `preprocess_features()` (src). Detecta un cruce de valores entre columnas (p. ej. una
  clave mal asignada en `valores_por_columna`) que un test de solo-orden no vería —
  parametrizar sobre los 3 tipos es necesario porque un cruce entre `Type_H`/`Type_L` solo
  se nota cuando el tipo evaluado no vale 0 en las tres columnas dummy.
