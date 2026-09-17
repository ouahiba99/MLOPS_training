# A separate, minimal image for the MLflow tracking server — it needs
# mlflow + a Postgres driver, nothing from requirements.txt (no fastapi,
# no sklearn, no pandas). Version pinned to match what the API and
# scripts/log_to_mlflow.py were tested against.
FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow==3.16.0 psycopg2-binary==2.9.9

WORKDIR /mlflow
EXPOSE 5000

HEALTHCHECK --interval=5s --timeout=5s --retries=10 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# No CMD here — docker-compose.yml supplies the full command, since the
# backend-store-uri needs env-var interpolation (postgres user/password).
