FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/model_torch.py ./model_torch.py
COPY backend/app.py ./app.py
COPY backend/download_models.py ./download_models.py
COPY benchmark/ ./benchmark/

# Download tokenizer + models at build time
RUN python download_models.py

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]