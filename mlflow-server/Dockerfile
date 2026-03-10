FROM python:3.11-slim

RUN pip install --no-cache-dir "mlflow>=2.14,<4.0" psycopg2-binary

# Expose MLflow default port
EXPOSE 5001

# BACKEND_STORE_URI and DEFAULT_ARTIFACT_ROOT are passed via environment
CMD ["sh", "-c", \
     "mlflow server \
      --backend-store-uri ${BACKEND_STORE_URI} \
      --default-artifact-root ${DEFAULT_ARTIFACT_ROOT} \
      --host 0.0.0.0 \
      --port 5001"]
