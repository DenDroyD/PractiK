#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram RAG-бот для ЛК ПроДвижение
Работает с: Groq API (LLM), HuggingFace (эмбеддинги), ChromaDB (векторная БД)
Оптимизирован для хостинга с 1 ГБ RAM
"""

import os
import sys
import asyncio
import logging
import hashlib
import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

# Telegram
from telegram import Update, constants
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    filters, ContextTypes, ConversationHandler
)

# AI & RAG
import requests
from bs4 import BeautifulSoup
import chromadb
from chromadb.config import Settings

# Утилиты
from functools import wraps
from collections import defaultdict

# ==================== КОНФИГУРАЦИЯ ====================
# Переменные окружения (обязательные)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
ADMIN_USER_ID = int(os.getenv("ADMIN_USER_ID", "0"))  # Telegram ID администратора

# Пути
DATA_DIR = Path("/app/data")
DOCS_DIR = DATA_DIR / "docs"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
LOG_FILE = DATA_DIR / "bot.log"

# Настройки AI
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "leasing_docs_v2"

# Лимиты (под 1 ГБ RAM)
MAX_CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_SEARCH_RESULTS = 5
MAX_HISTORY_MESSAGES = 5
MAX_PROMPT_TOKENS = 3500  # Оставляем запас для ответа (Groq max ~4096 для некоторых моделей)

# Кэширование
CACHE_TTL_EMBEDDINGS = 604800  # 7 дней
CACHE_TTL_RESPONSES = 86400    # 24 часа

# Приоритеты источников (для ранжирования)
SOURCE_PRIORITY = {
    "usloviia-soglasovaniia-sdelok.html": 1.5,
    "processy-lizingovoi-sdelki.html": 1.3,
    "izmeneniia-v-dogovor.html": 1.2,
    "soglasie-na-obrabotku-personalnyx-dannyx-sopd.html": 1.0,
}

# ==================== ЛОГИРОВАНИЕ ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8', mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==================== КЭШИ С TTL ====================
class TTLCache:
    """Простой потокобезопасный кэш с временем жизни"""
    def __init__(self, ttl_seconds: int):
        self._cache: Dict[str, tuple] = {}
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    logger.debug(f"Cache HIT: {key[:50]}...")
                    return value
                else:
                    del self._cache[key]
                    logger.debug(f"Cache EXPIRED: {key[:50]}...")
        return None
    
    async def set(self, key: str, value: Any):
        async with self._lock:
            self._cache[key] = (value, time.time())
            logger.debug(f"Cache SET: {key[:50]}...")
    
    async def clear(self):
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cache cleared: {count} items removed")

# Глобальные кэши
embedding_cache = TTLCache(CACHE_TTL_EMBEDDINGS)
response_cache = TTLCache(CACHE_TTL_RESPONSES)

# ==================== RETRY-ДЕКОРАТОР ====================
def retry_api(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Декоратор для повторных попыток при ошибках API"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    last_exception = e
                    wait_time = delay * (backoff ** attempt)
                    logger.warning(f"{func.__name__} failed (attempt {attempt+1}/{max_attempts}): {e}. Waiting {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                except Exception as e:
                    # Не повторяем непредсказуемые ошибки
                    logger.error(f"{func.__name__} unexpected error: {e}")
                    raise
            logger.error(f"{func.__name__} failed after {max_attempts} attempts")
            raise last_exception
        return wrapper
    return decorator

# ==================== PARSE HTML С СОХРАНЕНИЕМ СТРУКТУРЫ ====================
def parse_html_with_structure(html_content: str, source_filename: str) -> List[Dict]:
    """Извлекает текст из HTML с сохранением заголовков и таблиц"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Удаляем ненужные элементы
    for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
        tag.decompose()
    
    chunks = []
    current_header = ""
    
    for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'table', 'ul', 'ol', 'li']):
        text = element.get_text(separator=' ', strip=True)
        if not text or len(text) < 20:
            continue
            
        if element.name in ['h1', 'h2', 'h3']:
            current_header = text
        elif element.name == 'table':
            # Таблицы обрабатываем целиком, если не слишком большие
            table_text = element.get_text(separator=' | ', strip=True)
            if len(table_text) < 2000:
                chunks.append({
                    "text": f"[Таблица] {current_header}: {table_text}",
                    "metadata": {
                        "source": source_filename,
                        "header": current_header,
                        "type": "table",
                        "priority": SOURCE_PRIORITY.get(source_filename, 1.0)
                    }
                })
        else:
            # Обычный текст
            chunk_text = f"{current_header}: {text}" if current_header else text
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": source_filename,
                    "header": current_header,
                    "type": "text",
                    "priority": SOURCE_PRIORITY.get(source_filename, 1.0)
                }
            })
    
    return chunks

def smart_chunk_texts(chunks: List[Dict], max_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """Разбивает длинные чанки с перекрытием, сохраняя метаданные"""
    result = []
    for chunk in chunks:
        text = chunk["text"]
        if len(text) <= max_size:
            result.append(chunk)
        else:
            # Простое разбиение по предложениям с перекрытием
            sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]
            current = ""
            for i, sent in enumerate(sentences):
                sentence = sent + "."
                if len(current) + len(sentence) <= max_size:
                    current += (" " if current else "") + sentence
                else:
                    if current:
                        result.append({**chunk, "text": current})
                    # Добавляем перекрытие
                    if overlap > 0 and i > 0:
                        prev_sentences = sentences[max(0, i-3):i]
                        current = ". ".join(prev_sentences) + ". " + sentence
                    else:
                        current = sentence
            if current:
                result.append({**chunk, "text": current})
    return result

# ==================== EMBEDDINGS (HuggingFace) ====================
@retry_api(max_attempts=3, delay=1.5, backoff=2.0)
async def get_embedding(text: str) -> List[float]:
    """Получает эмбеддинг через HuggingFace API с кэшированием"""
    # Проверяем кэш
    cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()
    cached = await embedding_cache.get(cache_key)
    if cached is not None:
        return cached
    
    url = f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL}/pipeline/feature-extraction"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": text,
        "options": {"wait_for_model": True, "use_cache": True}
    }
    
    try:
        response = await asyncio.to_thread(
            requests.post, url, headers=headers, json=payload, timeout=30
        )
        response.raise_for_status()
        embedding = response.json()
        
        # HuggingFace может вернуть список списков — берём первый
        if isinstance(embedding, list) and embedding and isinstance(embedding[0], list):
            embedding = embedding[0]
        
        # Кэшируем
        await embedding_cache.set(cache_key, embedding)
        return embedding
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            logger.warning("HuggingFace rate limit exceeded. Waiting 60s...")
            await asyncio.sleep(60)
            return await get_embedding(text)  # Рекурсивная повторная попытка
        raise

# ==================== CHROMADB ====================
def init_chroma() -> chromadb.Collection:
    """Инициализирует ChromaDB в режиме persistent с оптимизацией памяти"""
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_PERSIST_DIR),
        settings=Settings(
            anonymized_telemetry=False,
            # Оптимизация памяти
            chroma_server_grpc_max_recv_message_length=100 * 1024 * 1024,  # 100 MB
        )
    )
    
    # Создаём или получаем коллекцию
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:construct_num_threads": 1,  # Экономия CPU
            "hnsw:search_num_threads": 1,
            "hnsw:ef_search": 100,  # Баланс точности/скорости
            "hnsw:ef_construction": 100,
        }
    )
    return collection

def index_documents(collection: chromadb.Collection, docs_dir: Path = DOCS_DIR):
    """Индексирует HTML-документы в ChromaDB"""
    if not docs_dir.exists():
        logger.error(f"Docs directory not found: {docs_dir}")
        return
    
    html_files = list(docs_dir.glob("*.html"))
    logger.info(f"Found {len(html_files)} HTML files to index")
    
    all_chunks = []
    all_ids = []
    all_metadatas = []
    
    for file_path in html_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Парсим с сохранением структуры
            chunks = parse_html_with_structure(html_content, file_path.name)
            # Умный чанкинг
            chunks = smart_chunk_texts(chunks)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{file_path.stem}_{i}_{hashlib.md5(chunk['text'][:100].encode()).hexdigest()[:8]}"
                all_chunks.append(chunk["text"])
                all_ids.append(chunk_id)
                all_metadatas.append(chunk["metadata"])
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    if not all_chunks:
        logger.warning("No chunks to index")
        return
    
    # Добавляем в ChromaDB (батчами по 100 для стабильности)
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i+batch_size]
        batch_ids = all_ids[i:i+batch_size]
        batch_metas = all_metadatas[i:i+batch_size]
        
        # Получаем эмбеддинги батчем (последовательно, чтобы не превысить лимиты HF)
        embeddings = []
        for chunk_text in batch_chunks:
            emb = asyncio.run(get_embedding(chunk_text))
            embeddings.append(emb)
        
        collection.add(
            ids=batch_ids,
            documents=batch_chunks,
            metadatas=batch_metas,
            embeddings=embeddings
        )
        logger.info(f"Indexed {min(i+batch_size, len(all_chunks))}/{len(all_chunks)} chunks")
    
    logger.info(f"Indexing complete: {len(all_chunks)} chunks from {len(html_files)} files")

# ==================== ПОИСК С ПРИОРИТЕТАМИ ====================
def query_with_priority(collection: chromadb.Collection, query_embedding: List[float], 
                       user_query: str, n_results: int = MAX_SEARCH_RESULTS) -> Dict:
    """Векторный поиск с учётом приоритета источников"""
    # Берём больше результатов для последующей фильтрации
    raw_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results * 2,
        include=["documents", "metadatas", "distances"]
    )
    
    if not raw_results['documents'][0]:
        return raw_results
    
    # Применяем приоритеты
    scored = []
    for doc, meta, dist in zip(
        raw_results['documents'][0],
        raw_results['metadatas'][0],
        raw_results['distances'][0]
    ):
        priority = meta.get('priority', 1.0)
        # Score: чем выше — тем лучше (1-dist = similarity)
        score = (1 - dist) * priority
        scored.append((score, doc, meta, dist))
    
    # Сортируем и берем топ
    scored.sort(key=lambda x: x[0], reverse=True)
    top_n = scored[:n_results]
    
    return {
        'documents': [[item[1] for item in top_n]],
        'metadatas': [[item[2] for item in top_n]],
        'distances': [[item[3] for item in top_n]],
        'scores': [[item[0] for item in top_n]]
    }

# ==================== GROQ API ====================
@retry_api(max_attempts=3, delay=1.0, backoff=2.0)
async def call_groq(prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
    """Вызов Groq API с обработкой ошибок"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Обрезаем промпт, если слишком длинный (грубая оценка: 4 символа ≈ 1 токен)
    if len(prompt) > MAX_PROMPT_TOKENS * 4:
        prompt = prompt[:MAX_PROMPT_TOKENS * 4] + "\n\n[...обрезано из-за лимита длины...]"
        logger.warning("Prompt truncated due to length limit")
    
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }
    
    response = await asyncio.to_thread(
        requests.post, url, headers=headers, json=payload, timeout=60
    )
    response.raise_for_status()
    
    result = response.json()
    return result['choices'][0]['message']['content'].strip()

# ==================== ФОРМИРОВАНИЕ ПРОМПТА ====================
def build_prompt(context_chunks: List[str], user_query: str, 
                 conversation_history: str = "") -> str:
    """Собирает финальный промпт для LLM"""
    context_text = "\n\n---\n\n".join([
        f"[Источник: {chunk.get('source', 'unknown')}]" if isinstance(chunk, dict) else "" 
        + (chunk['text'] if isinstance(chunk, dict) else chunk)
        for chunk in context_chunks
    ])
    
    base_prompt = """Ты — экспертный помощник лизинговой компании «ЛК ПроДвижение». 
Отвечай ТОЛЬКО на основе предоставленного контекста из внутренней документации.

ПРАВИЛА ОТВЕТА:
1. Если информация есть в контексте — дай точный, полный ответ с цифрами и условиями.
2. Если данные противоречивы — укажи на это и приведи оба варианта.
3. Если информации недостаточно — задай УТОЧНЯЮЩИЙ вопрос, а не выдумывай.
4. Числовые значения (лимита, ставки, сроки) выделяй **жирным**.
5. Списки оформляй маркированным списком.
6. Отвечай на русском языке, профессионально, но понятно.

Контекст из документации:
{context}

{history}

Вопрос клиента: {question}

Ответ:"""

    history_part = f"\nИстория диалога:\n{conversation_history}\n" if conversation_history else ""
    
    return base_prompt.format(
        context=context_text,
        history=history_part,
        question=user_query
    )

# ==================== УПРАВЛЕНИЕ ПАМЯТЬЮ ДИАЛОГА ====================
# Хранилище: chat_id -> список сообщений
user_conversations: Dict[int, List[Dict]] = defaultdict(list)

def add_to_history(chat_id: int, role: str, content: str, topic: str = None):
    """Добавляет сообщение в историю диалога"""
    history = user_conversations[chat_id]
    history.append({
        "role": role,
        "content": content,
        "topic": topic,
        "timestamp": time.time()
    })
    # Ограничиваем длину
    if len(history) > MAX_HISTORY_MESSAGES * 2:  # *2 т.к. user+assistant
        user_conversations[chat_id] = history[-MAX_HISTORY_MESSAGES * 2:]

def get_formatted_history(chat_id: int) -> str:
    """Возвращает отформатированную историю для промпта"""
    history = user_conversations.get(chat_id, [])
    if not history:
        return ""
    
    # Берем последние сообщения, исключая текущий вопрос
    messages = [f"{msg['role']}: {msg['content']}" for msg in history[:-1]]
    return "\n".join(messages[-MAX_HISTORY_MESSAGES:])

# ==================== ОБРАБОТЧИКИ TELEGRAM ====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start"""
    user = update.effective_user
    await update.message.reply_text(
        f"Здравствуйте, {user.first_name}! 👋\n\n"
        "Я — бот-помощник ЛК ПроДвижение.\n"
        "Могу ответить на вопросы по:\n"
        "• Условиям лизинга и лимитам\n"
        "• Процессам оформления сделок\n"
        "• Документам и требованиям\n"
        "• Изменениям в договорах\n\n"
        "Просто напишите ваш вопрос — я найду ответ в базе знаний. 📚"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /help"""
    await update.message.reply_text(
        "📋 Доступные команды:\n\n"
        "/start — начать диалог\n"
        "/help — эта справка\n"
        "/reindex — [админ] переиндексировать документы\n"
        "/stats — [админ] статистика использования\n"
        "/clear — очистить историю диалога\n\n"
        "💡 Советы:\n"
        "• Задавайте конкретные вопросы: «Какой минимальный аванс для грузовиков?»\n"
        "• Указывайте тип клиента (ИП/ЮЛ) и выручку для точного ответа\n"
        "• Если ответ неполный — уточните вопрос"
    )

async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /clear — очистка истории"""
    chat_id = update.effective_chat.id
    user_conversations[chat_id] = []
    await response_cache.set(f"response:{chat_id}:*", None)  # Очистка кэша ответов для этого чата
    await update.message.reply_text("🗑️ История диалога очищена. Начнём с чистого листа.")

async def reindex_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /reindex — переиндексация (только админ)"""
    if update.effective_user.id != ADMIN_USER_ID:
        await update.message.reply_text("❌ Доступ только для администратора.")
        return
    
    msg = await update.message.reply_text("🔄 Начинаю переиндексацию документов...")
    
    try:
        # Удаляем старую коллекцию
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except:
            pass  # Коллекция может не существовать
        
        # Создаём новую и индексируем
        collection = init_chroma()
        index_documents(collection)
        
        # Очищаем кэши
        await embedding_cache.clear()
        await response_cache.clear()
        
        await msg.edit_text("✅ Переиндексация завершена успешно!\nКэши очищены.")
    except Exception as e:
        logger.error(f"Reindex error: {e}")
        await msg.edit_text(f"❌ Ошибка при переиндексации: {str(e)}")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /stats — статистика (только админ)"""
    if update.effective_user.id != ADMIN_USER_ID:
        await update.message.reply_text("❌ Доступ только для администратора.")
        return
    
    # Простая статистика
    total_users = len(user_conversations)
    total_messages = sum(len(hist) for hist in user_conversations.values())
    cache_size_emb = len(embedding_cache._cache)
    cache_size_resp = len(response_cache._cache)
    
    stats_text = (
        f"📊 Статистика бота:\n"
        f"• Активных диалогов: {total_users}\n"
        f"• Всего сообщений в памяти: {total_messages}\n"
        f"• Кэш эмбеддингов: {cache_size_emb} записей\n"
        f"• Кэш ответов: {cache_size_resp} записей\n"
        f"• Модель LLM: {GROQ_MODEL}\n"
        f"• Модель эмбеддингов: {EMBEDDING_MODEL.split('/')[-1]}"
    )
    await update.message.reply_text(stats_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Основной обработчик сообщений"""
    user = update.effective_user
    chat_id = update.effective_chat.id
    user_text = update.message.text.strip()
    
    if not user_text:
        return
    
    logger.info(f"User {user.id} ({user.username}): {user_text[:100]}...")
    
    # Добавляем вопрос в историю
    add_to_history(chat_id, "user", user_text)
    
    # Показываем индикатор "печатает..."
    await update.message.chat.send_action(constants.ChatAction.TYPING)
    
    # Проверяем кэш ответов (ключ: chat_id + вопрос)
    cache_key = f"response:{chat_id}:{hashlib.md5(user_text.encode()).hexdigest()}"
    cached_response = await response_cache.get(cache_key)
    if cached_response:
        logger.info(f"Response cache HIT for chat {chat_id}")
        await update.message.reply_text(cached_response)
        add_to_history(chat_id, "assistant", cached_response)
        return
    
    try:
        # 1. Получаем эмбеддинг запроса
        query_embedding = await get_embedding(user_text)
        
        # 2. Ищем релевантные чанки
        collection = init_chroma()
        search_results = query_with_priority(collection, query_embedding, user_text)
        
        if not search_results['documents'][0]:
            # Нет результатов — просим уточнить
            fallback = (
                "❓ Я не нашёл точной информации по вашему вопросу в базе знаний.\n\n"
                "Пожалуйста, уточните:\n"
                "• Тип клиента (ИП или ЮЛ)?\n"
                "• Выручка компании?\n"
                "• Тип предмета лизинга (авто, спецтехника, оборудование)?\n"
                "• Конкретный параметр (лимит, аванс, срок)?"
            )
            await update.message.reply_text(fallback)
            add_to_history(chat_id, "assistant", fallback, topic="clarification_needed")
            return
        
        # 3. Формируем контекст
        context_chunks = search_results['documents'][0]
        history_text = get_formatted_history(chat_id)
        
        # 4. Строим промпт
        prompt = build_prompt(context_chunks, user_text, history_text)
        
        # 5. Вызываем LLM
        start_time = time.time()
        answer = await call_groq(prompt)
        elapsed = time.time() - start_time
        logger.info(f"Groq response in {elapsed:.2f}s, {len(answer)} chars")
        
        # 6. Кэшируем ответ
        await response_cache.set(cache_key, answer)
        
        # 7. Отправляем и сохраняем в историю
        await update.message.reply_text(answer)
        add_to_history(chat_id, "assistant", answer)
        
    except Exception as e:
        logger.error(f"Error handling message: {e}", exc_info=True)
        await update.message.reply_text(
            "⚠️ Произошла ошибка при обработке запроса.\n"
            "Пожалуйста, попробуйте ещё раз или обратитесь к администратору."
        )

# ==================== ЗАПУСК ====================
def main():
    """Точка входа"""
    # Создаём директорию для данных, если нет
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Инициализируем ChromaDB и индексируем документы при первом запуске
    if not CHROMA_PERSIST_DIR.exists() or not list(CHROMA_PERSIST_DIR.glob("*")):
        logger.info("First run: initializing ChromaDB and indexing documents...")
        collection = init_chroma()
        index_documents(collection)
    else:
        logger.info("ChromaDB already initialized, skipping indexing")
    
    # Создаём приложение
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("clear", clear_history))
    
    # Админ-команды
    if ADMIN_USER_ID > 0:
        application.add_handler(CommandHandler("reindex", reindex_command))
        application.add_handler(CommandHandler("stats", stats_command))
    
    # Обработчик текстовых сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Запускаем
    logger.info("Starting bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()