# Dockerfile — Proyecto de Mantenimiento Predictivo
#
# TODO: implementar en Módulo 3
#
# Plan previsto:
#   - Imagen base: python:3.12-slim
#   - Instalar uv y ejecutar `uv sync --frozen`
#   - Copiar el código del proyecto
#   - Exponer el puerto 8501
#   - CMD: uv run streamlit run app/app.py --server.port=8501 --server.address=0.0.0.0
