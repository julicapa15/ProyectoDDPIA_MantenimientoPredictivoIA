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
- [ ] T012 Verificar ejecución local interactiva con `uv run streamlit run app/main.py`
