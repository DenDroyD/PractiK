FROM python:3.11-slim

WORKDIR /app

# Минимальные системные зависимости
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Копируем и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

# Создаем папки
RUN mkdir -p /app/data/docs && chmod -R 777 /app/data

ENV DATA_DIR=/app/data
ENV PYTHONUNBUFFERED=1

CMD ["python", "bot.py"]