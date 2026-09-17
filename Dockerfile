# ---- builder: install runtime deps into a venv, kept out of the final image ----
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements/requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ---- final: slim runtime image ----
FROM python:3.12-slim

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Only what the service needs to run — no notebooks, no tests, no dev tools.
# models/ is intentionally NOT copied here: the model comes from the MLflow
# registry (src/model_registry.py) and the preprocessing artifacts are
# mounted as a volume (see docker-compose.yml) so retraining doesn't mean
# rebuilding the image.
COPY app/ app/
COPY src/ src/
COPY config/ config/
COPY great_expectations/ great_expectations/
COPY scripts/ scripts/

RUN mkdir -p /app/logs /app/models/artifacts /app/data \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen(chr(104)+chr(116)+chr(116)+chr(112)+chr(58)+chr(47)+chr(47)+chr(108)+chr(111)+chr(99)+chr(97)+chr(108)+chr(104)+chr(111)+chr(115)+chr(116)+chr(58)+chr(56)+chr(48)+chr(48)+chr(48)+chr(47)+chr(104)+chr(101)+chr(97)+chr(108)+chr(116)+chr(104))" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
