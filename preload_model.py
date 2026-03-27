#!/usr/bin/env python3
"""
Предзагрузка модели эмбеддингов для кэширования в Docker-образе
"""
import os
from pathlib import Path

cache_dir = Path('/app/models_cache')
cache_dir.mkdir(parents=True, exist_ok=True)

print('🚀 Предзагрузка модели all-MiniLM-L3-v2...')
print(f'📁 Кэш директория: {cache_dir}')

try:
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(
        'all-MiniLM-L3-v2',
        cache_folder=str(cache_dir),
        device='cpu',
        trust_remote_code=True
    )
    
    # Тестовый прогон
    model.encode(['test'], batch_size=1, show_progress_bar=False)
    
    # Считаем размер кэша
    total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
    print(f'✅ Модель предзагружена успешно!')
    print(f'💾 Размер кэша: {total_size / 1024 / 1024:.1f} МБ')
    
except Exception as e:
    print(f'⚠️ Предзагрузка не удалась: {e}')
    print('💡 Модель будет загружена при первом запуске бота')
    # Не прерываем выполнение — выходим с кодом 0
    exit(0)