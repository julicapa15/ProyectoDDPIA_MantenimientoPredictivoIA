# syntax=docker/dockerfile:1
#
# Dockerfile — Mantenimiento Predictivo IA
#
# Imagen de producción para la interfaz Streamlit (app/main.py). Usa `uv` para
# instalar las dependencias de forma reproducible a partir de `uv.lock`, en dos
# capas separadas (dependencias vs. código propio) para que un cambio en
# app/ o src/ no obligue a reinstalar TabPFN/XGBoost/PyTorch en cada build.
#
# El dataset AI4I 2020 (data/raw/ai4i2020.csv) está excluido de la imagen
# (ver .dockerignore, igual que en .gitignore) y debe montarse como volumen en
# tiempo de ejecución — ver instrucciones de build/run en el README, sección
# "Ejecución con Docker".

FROM python:3.12-slim

# Binario estático de uv, copiado directo desde su imagen oficial (sin pip).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# 1) Capa de dependencias: se cachea mientras pyproject.toml/uv.lock no cambien,
#    sin importar cambios posteriores en el código de app/ o src/. `--no-dev`
#    excluye pytest/ruff/pre-commit, innecesarios en la imagen de producción.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# 2) Código del proyecto necesario en tiempo de ejecución (tests/, notebooks/,
#    EDA/ y los datos crudos quedan fuera vía .dockerignore).
COPY app/ ./app/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Usuario sin privilegios. TabPFN-v2 descarga sus pesos preentrenados en el
# primer uso bajo $HOME/.cache, por eso HOME apunta a un directorio de un
# usuario real (no root) con permisos de escritura. Se crean además los
# subdirectorios de datos para que el volumen montado en /app/data tenga dueño
# correcto.
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --home-dir /home/app --create-home app \
    && mkdir -p /app/data/raw /app/data/processed \
    && chown -R app:app /app /home/app
ENV HOME=/home/app
USER app

EXPOSE 8501

# Streamlit expone un endpoint de salud propio; se usa Python (ya presente en
# la imagen) en vez de instalar curl solo para esto.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health', timeout=3)" || exit 1

CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
