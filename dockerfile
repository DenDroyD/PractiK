FROM python:3.11-slim

WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements
COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Предзагружаем модель эмбеддингов (выполняется ПРИ СБОРКЕ!)
RUN python -c " \
from sentence_transformers import SentenceTransformer; \
import os; \
print('🚀 Начинаю предзагрузку модели all-MiniLM-L3-v2...'); \
cache_dir = '/app/models_cache'; \
os.makedirs(cache_dir, exist_ok=True); \
model = SentenceTransformer('all-MiniLM-L3-v2', cache_folder=cache_dir, device='cpu'); \
model.encode(['test'], batch_size=1, show_progress_bar=False); \
print('✅ Модель успешно предзагружена в', cache_dir); \
import os; \
total_size = sum(f.stat().st_size for f in os.walk(cache_dir) for _, _, files in _ for f in [os.path.join(_, files)[0]]); \
print(f'💾 Размер кэша: {total_size / 1024 / 1024:.1f} МБ') \
"

# Копируем весь код проекта
COPY . .

# Создаем необходимые директории
RUN mkdir -p /app/data/docs /app/models_cache && \
    chmod -R 777 /app/data

# Переменные окружения по умолчанию
ENV DATA_DIR=/app/data
ENV PYTHONUNBUFFERED=1
ENV USE_RAG=true

# Запускаем бота
CMD ["python", "bot.py"]