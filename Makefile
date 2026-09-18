# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Mantenimiento Predictivo IA — Makefile                                    ║
# ║  Ejecutar `make help` para ver todos los comandos disponibles.             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

.DEFAULT_GOAL := help
SHELL := cmd.exe

# ── Variables ──────────────────────────────────────────────────────────────────

PYTHON   := uv run python
PYTEST   := uv run pytest
RUFF     := uv run ruff
STREAMLIT:= uv run streamlit
APP_FILE := app/main.py
PORT     := 8501

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HELP — Menú de comandos                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

.PHONY: help
help: ## Muestra este menú de ayuda
	@echo.
	@echo  Mantenimiento Predictivo IA — Comandos disponibles
	@echo  ────────────────────────────────────────────────────
	@echo.
	@echo  Uso: make [comando]
	@echo.
	@echo  Desarrollo:
	@echo    install            Instala dependencias con uv sync
	@echo    lint               Verifica estilo con Ruff
	@echo    lint-fix           Corrige problemas de estilo automaticamente
	@echo    format             Formatea el codigo con Ruff
	@echo    format-check       Verifica formato sin modificar (CI)
	@echo    check              lint + format-check (para CI)
	@echo    test               Ejecuta todos los tests
	@echo    test-unit          Solo tests unitarios
	@echo    test-integration   Solo tests de integracion
	@echo    test-app           Tests de la interfaz Streamlit
	@echo    coverage           Tests con reporte de cobertura
	@echo.
	@echo  Aplicacion:
	@echo    app                Lanza la interfaz web (Streamlit)
	@echo    app-reload         Lanza con hot-reload forzado
	@echo    evaluar            Evalua TabPFN + XGBoost (MLflow)
	@echo    evaluar-xgboost    Evalua solo XGBoost
	@echo.
	@echo  Docker:
	@echo    docker-build       Construye la imagen Docker
	@echo    docker-run         Ejecuta la app en contenedor
	@echo.
	@echo  Utilidades:
	@echo    clean              Limpia caches y archivos temporales
	@echo    pre-commit         Ejecuta pre-commit en todos los archivos
	@echo    mlflow             Abre la UI de MLflow
	@echo    status             Estado del proyecto (Python, git, deps)
	@echo.

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DESARROLLO — Linting, formato y tests                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

.PHONY: install
install: ## Instala todas las dependencias con uv sync
	uv sync

.PHONY: lint
lint: ## Verifica estilo con Ruff (sin modificar archivos)
	$(RUFF) check .

.PHONY: lint-fix
lint-fix: ## Corrige automáticamente problemas de estilo con Ruff
	$(RUFF) check --fix .

.PHONY: format
format: ## Formatea el código con Ruff
	$(RUFF) format .

.PHONY: format-check
format-check: ## Verifica formato sin modificar (para CI)
	$(RUFF) format --check .

.PHONY: check
check: lint format-check ## Ejecuta lint + format-check (ideal para CI)

.PHONY: test
test: ## Ejecuta todos los tests con pytest
	$(PYTEST) -v

.PHONY: test-unit
test-unit: ## Ejecuta solo tests unitarios
	$(PYTEST) tests/unit/ -v

.PHONY: test-integration
test-integration: ## Ejecuta solo tests de integración
	$(PYTEST) tests/integration/ -v

.PHONY: test-app
test-app: ## Ejecuta tests de la interfaz Streamlit
	$(PYTEST) tests/unit/test_app.py -v

.PHONY: coverage
coverage: ## Ejecuta tests con reporte de cobertura
	$(PYTEST) --cov=src --cov=app --cov-report=term-missing -v

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  APLICACIÓN — Interfaz web y evaluación de modelos                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

.PHONY: app
app: ## Lanza la interfaz web de Streamlit
	$(STREAMLIT) run $(APP_FILE) --server.port=$(PORT)

.PHONY: app-reload
app-reload: ## Lanza la interfaz con hot-reload forzado
	$(STREAMLIT) run $(APP_FILE) --server.port=$(PORT) --server.runOnSave=true

.PHONY: evaluar
evaluar: ## Evalúa TabPFN-v2 y XGBoost, registra en MLflow
	$(PYTHON) -m scripts.evaluar_modelos

.PHONY: evaluar-xgboost
evaluar-xgboost: ## Evalúa solo XGBoost (sin TabPFN)
	$(PYTHON) -m scripts.evaluar_modelos --sin-tabpfn

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DOCKER — Contenedores                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

.PHONY: docker-build
docker-build: ## Construye la imagen Docker del proyecto
	docker build -t mantenimiento-predictivo .

.PHONY: docker-run
docker-run: ## Ejecuta la app en un contenedor Docker
	docker run -p $(PORT):8501 mantenimiento-predictivo

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  UTILIDADES — Limpieza y mantenimiento                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

.PHONY: clean
clean: ## Limpia archivos temporales, cachés y __pycache__
	@echo Limpiando archivos temporales...
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@if exist .ruff_cache rmdir /s /q .ruff_cache
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
	@echo Limpieza completada.

.PHONY: pre-commit
pre-commit: ## Ejecuta pre-commit en todos los archivos
	uv run pre-commit run --all-files

.PHONY: mlflow
mlflow: ## Abre la UI de MLflow en el navegador
	$(PYTHON) -m mlflow ui

.PHONY: status
status: ## Muestra el estado del proyecto (Python, dependencias, git)
	@echo ── Python ──
	@$(PYTHON) --version
	@echo.
	@echo ── Git ──
	@git branch --show-current
	@git status --short
	@echo.
	@echo ── Dependencias ──
	@uv pip list --quiet 2>nul | findstr /i "streamlit tabpfn xgboost scikit"
