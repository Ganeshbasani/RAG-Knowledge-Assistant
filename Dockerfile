FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
# Render builds from the repository root, while the application lives under
# ./rag-knowledge-assistant-main/. Flatten that project into the image root.
WORKDIR /app
COPY rag-knowledge-assistant-main/pyproject.toml ./
COPY rag-knowledge-assistant-main/README.md ./
COPY rag-knowledge-assistant-main/LICENSE ./
COPY rag-knowledge-assistant-main/src ./src
COPY rag-knowledge-assistant-main/ui ./ui
RUN pip install --no-cache-dir -e ".[embeddings]" \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data/uploads \
    && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1
CMD ["sh", "-c", "uvicorn rag_assistant.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
