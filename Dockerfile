FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for efficient layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and model code
COPY app/ ./app/
COPY models/ ./models/
COPY src/ ./src/
COPY params.yaml .

ENV PYTHONPATH="/app/src:${PYTHONPATH}"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
