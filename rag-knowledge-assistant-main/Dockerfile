FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY ui ./ui
COPY tests ./tests

RUN pip install --no-cache-dir -e ".[embeddings,dev]" \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data/uploads \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1

CMD ["sh", "-c", "uvicorn rag_assistant.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
