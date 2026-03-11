FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    curl \
    fonts-dejavu-core \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --retries 5 --timeout 60 -r requirements.txt

# Validate SDK installed correctly — fail build early if not
RUN python -c "from yandex_ai_studio_sdk import AIStudio; print('SDK OK')"

COPY . .

RUN mkdir -p demo/audio demo/photos /tmp/yamnet_cache

EXPOSE 8000

CMD ["uvicorn", "cloud.interface.main:app", "--host", "0.0.0.0", "--port", "8000"]
