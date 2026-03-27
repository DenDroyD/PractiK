FROM python:3.11-slim

WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements
COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# ПРЕДЗАГРУЗКА МОДЕЛИ (выполняется ПРИ СБОРКЕ)
# ============================================
RUN python -c " \
import os; \
import sys; \
from pathlib import Path; \
\
cache_dir = Path('/app/models_cache'); \
cache_dir.mkdir(parents=True, exist_ok=True); \
\
print('🚀 Начинаю предзагрузку модели all-MiniLM-L3-v2...'); \
print(f'📁 Кэш директория: {cache_dir}'); \
\
try: \
    from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer( \
        'all-MiniLM-L3-v2', \
        cache_folder=str(cache_dir), \
        device='cpu', \
        trust_remote_code=True \
    ); \
    model.encode(['test'], batch_size=1, show_progress_bar=False); \
    print('✅ Модель успешно предзагружена!'); \
    \
    total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()); \
    print(f'💾 Размер кэша: {total_size / 1024 / 1024:.1f} МБ'); \
    \
except Exception as e: \
    print(f'⚠️ Предзагрузка не удалась: {e}'); \
    print('💡 Модель будет загружена при первом запуске бота'); \
    sys.exit(0); \
"

# Копируем весь код проекта
COPY . .

# Создаем необходимые директории
RUN mkdir -p /app/data/docs /app/models_cache && \
    chmod -R 777 /app/data

# Переменные окружения
ENV DATA_DIR=/app/data
ENV PYTHONUNBUFFERED=1
ENV USE_RAG=true

# Запускаем бота
CMD ["python", "bot.py"]