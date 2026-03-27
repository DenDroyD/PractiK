FROM python:3.11-slim

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем скрипт предзагрузки
COPY preload_model.py .

# Запускаем предзагрузку модели
RUN python preload_model.py || echo "⚠️ Предзагрузка не удалась — модель загрузится при запуске"

# Копируем код бота
COPY . .

# Создаем директории
RUN mkdir -p /app/data/docs /app/models_cache && chmod -R 777 /app/data

ENV DATA_DIR=/app/data
ENV PYTHONUNBUFFERED=1
ENV USE_RAG=true

CMD ["python", "bot.py"]