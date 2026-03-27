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

# ============================================
# ПРЕДЗАГРУЗКА МОДЕЛИ (с игнорированием ошибок)
# ============================================
RUN python3 << 'EOF' || echo "⚠️ Предзагрузка не удалась — модель загрузится при запуске"
import os
from pathlib import Path

cache_dir = Path('/app/models_cache')
cache_dir.mkdir(parents=True, exist_ok=True)

print('🚀 Предзагрузка модели all-MiniLM-L3-v2...')

try:
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(
        'all-MiniLM-L3-v2',
        cache_folder=str(cache_dir),
        device='cpu',
        trust_remote_code=True
    )
    
    model.encode(['test'], batch_size=1, show_progress_bar=False)
    
    total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
    print(f'✅ Модель предзагружена! Размер: {total_size / 1024 / 1024:.1f} МБ')
    
except Exception as e:
    print(f'⚠️ Предзагрузка не удалась: {e}')
    # Не прерываем сборку!
EOF

# Копируем код
COPY . .

# Директории
RUN mkdir -p /app/data/docs /app/models_cache && chmod -R 777 /app/data

ENV DATA_DIR=/app/data
ENV PYTHONUNBUFFERED=1
ENV USE_RAG=true

CMD ["python", "bot.py"]