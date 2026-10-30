FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements-deploy.txt .

RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY src/ ./src/
COPY api/ ./api/
COPY scripts/init_database.py ./scripts/init_database.py

COPY data/raw/ ./data/raw/

COPY results/final_test/final_model.pt \
     ./results/final_test/final_model.pt

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]