FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p demo/audio demo/photos /tmp/yamnet_cache

EXPOSE 8000

CMD ["uvicorn", "cloud.interface.main:app", "--host", "0.0.0.0", "--port", "8000"]
