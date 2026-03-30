#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram RAG-бот для ЛК ПроДвижение
Версия: 2.1 (финальная)
Совместимость: Python 3.11+, bothost.ru (1 ГБ RAM)
"""

import os
import sys
import asyncio
import logging
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from functools import wraps

# Telegram
from telegram import Update, constants
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# HTML Parsing
from bs4 import BeautifulSoup

# Vector DB
import chromadb
from chromadb.config import Settings

# HTTP
import requests

# ==================== КОНФИГУРАЦИЯ ====================
# Переменные окружения (задаются в панели bothost.ru)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# ID администратора (ваш: 477810377)
ADMIN_USER_ID = int(os.getenv("ADMIN_USER_ID", "477810377"))

# Пути к данным
DATA_DIR = Path("/app/data")
DOCS_DIR = DATA_DIR / "docs"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
LOG_FILE = DATA_DIR / "bot.log"

# === НАСТРОЙКИ МОДЕЛЕЙ (легко менять через env) ===
# Модель Groq для генерации ответов
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Модель HuggingFace для эмбеддингов
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", 
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Имя коллекции в ChromaDB
COLLECTION_NAME = "leasing_docs_v2"

# === ЛИМИТЫ (оптимизация под 1 ГБ RAM) ===
MAX_CHUNK_SIZE = 800          # Макс. размер чанка текста
CHUNK_OVERLAP = 100           # Перекрытие при разбиении
MAX_SEARCH_RESULTS = 5        # Сколько чанков искать
MAX_HISTORY_MESSAGES = 5      # Сообщений в истории диалога
MAX_PROMPT_TOKENS = 3500      # Лимит токенов в промпте

# === КЭШИРОВАНИЕ ===
CACHE_TTL_EMBEDDINGS = 604800  # 7 дней для эмбеддингов
CACHE_TTL_RESPONSES = 86400    # 24 часа для ответов

# === ПРИОРИТЕТЫ ИСТОЧНИКОВ ===
SOURCE_PRIORITY = {
    "usloviia-soglasovaniia-sdelok.html": 1.5,  # Главный справочник
    "processy-lizingovoi-sdelki.html": 1.3,      # Процессы
    "izmeneniia-v-dogovor.html": 1.2,            # Изменения договора
    "soglasie-na-obrabotku-personalnyx-dannyx-sopd.html": 1.0,
}

# ==================== ЛОГИРОВАНИЕ ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ==================== КЭШ С TTL ====================
class TTLCache:
    """Простой кэш с временем жизни"""
    def __init__(self, ttl_seconds: int):
        self._cache: Dict[str, tuple] = {}
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    return value
                del self._cache[key]
        return None
    
    async def set(self, key: str, value: Any):
        async with self._lock:
            self._cache[key] = (value, time.time())
    
    async def clear(self):
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cache cleared: {count} items")
    
    def __len__(self):
        return len(self._cache)

# Глобальные кэши
embedding_cache = TTLCache(CACHE_TTL_EMBEDDINGS)
response_cache = TTLCache(CACHE_TTL_RESPONSES)

# ==================== RETRY-ДЕКОРАТОР ====================
def retry_api(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Повторные попытки при ошибках API"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    last_exc = e
                    wait = delay * (backoff ** attempt)
                    logger.warning(f"{func.__name__} failed (#{attempt+1}): {e}. Wait {wait:.1f}s")
                    await asyncio.sleep(wait)
                except Exception as e:
                    logger.error(f"{func.__name__} unexpected: {e}")
                    raise
            raise last_exc
        return wrapper
    return decorator

# ==================== PARSE HTML ====================
def parse_html_with_structure(html_content: str, source_filename: str) -> List[Dict]:
    """Извлекает текст из HTML с сохранением заголовков и таблиц"""
    soup = BeautifulSoup(html_content, "lxml")
    
    # Удаляем ненужные элементы
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    
    chunks = []
    current_header = ""
    
    # Специальная обработка таблиц из processy-lizingovoi-sdelki.html
    if source_filename == "processy-lizingovoi-sdelki.html":
        for table in soup.find_all("table"):
            rows = []
            for tr in table.find_all("tr"):
                cells = [td.get_text(separator=" ", strip=True) for td in tr.find_all(["td", "th"])]
                if cells and any(c.strip() for c in cells):
                    rows.append(" | ".join(cells))
            if rows:
                table_text = "\n".join(rows)
                chunks.append({
                    "text": f"[ТАБЛИЦА ПРОЦЕССА] {current_header}\n{table_text}",
                    "metadata": {
                        "source": source_filename,
                        "header": current_header,
                        "type": "process_table",
                        "priority": SOURCE_PRIORITY.get(source_filename, 1.0),
                    },
                })
    
    for element in soup.find_all(["h1", "h2", "h3", "p", "table", "ul", "ol", "li"]):
        text = element.get_text(separator=" ", strip=True)
        if not text or len(text) < 20:
            continue
        
        if element.name in ["h1", "h2", "h3"]:
            current_header = text
        elif element.name == "table" and source_filename != "processy-lizingovoi-sdelki.html":
            table_text = element.get_text(separator=" | ", strip=True)
            if len(table_text) < 2000:
                chunks.append({
                    "text": f"[Таблица] {current_header}: {table_text}",
                    "metadata": {
                        "source": source_filename,
                        "header": current_header,
                        "type": "table",
                        "priority": SOURCE_PRIORITY.get(source_filename, 1.0),
                    },
                })
        else:
            chunk_text = f"{current_header}: {text}" if current_header else text
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": source_filename,
                    "header": current_header,
                    "type": "text",
                    "priority": SOURCE_PRIORITY.get(source_filename, 1.0),
                },
            })
    
    return chunks

def smart_chunk_texts(chunks: List[Dict], max_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """Разбивает длинные чанки с перекрытием"""
    result = []
    for chunk in chunks:
        text = chunk["text"]
        if len(text) <= max_size:
            result.append(chunk)
        else:
            sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
            current = ""
            for i, sent in enumerate(sentences):
                sentence = sent + "."
                if len(current) + len(sentence) <= max_size:
                    current += (" " if current else "") + sentence
                else:
                    if current:
                        result.append({**chunk, "text": current})
                    if overlap > 0 and i > 0:
                        prev = sentences[max(0, i-3):i]
                        current = ". ".join(prev) + ". " + sentence
                    else:
                        current = sentence
            if current:
                result.append({**chunk, "text": current})
    return result

# ==================== EMBEDDINGS (HuggingFace) ====================
@retry_api(max_attempts=3, delay=1.5, backoff=2.0)
async def get_embedding(text: str) -> List[float]:
    """Получает эмбеддинг через HuggingFace API с кэшированием"""
    cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()
    cached = await embedding_cache.get(cache_key)
    if cached is not None:
        return cached
    
    url = f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL}/pipeline/feature-extraction"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": text,
        "options": {"wait_for_model": True, "use_cache": True},
    }
    
    try:
        response = await asyncio.to_thread(
            requests.post, url, headers=headers, json=payload, timeout=30
        )
        response.raise_for_status()
        embedding = response.json()
        
        # Нормализуем ответ
        if isinstance(embedding, list) and embedding and isinstance(embedding[0], list):
            embedding = embedding[0]
        
        await embedding_cache.set(cache_key, embedding)
        return embedding
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            logger.warning("HF rate limit. Waiting 60s...")
            await asyncio.sleep(60)
            return await get_embedding(text)
        raise

# ==================== CHROMADB ====================
def init_chroma() -> chromadb.Collection:
    """Инициализирует ChromaDB с оптимизацией памяти"""
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_PERSIST_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:construct_num_threads": 1,
            "hnsw:search_num_threads": 1,
            "hnsw:ef_search": 100,
            "hnsw:ef_construction": 100,
        },
    )

def index_documents(collection: chromadb.Collection, docs_dir: Path = DOCS_DIR):
    """Индексирует HTML-документы в ChromaDB"""
    if not docs_dir.exists():
        docs_dir.mkdir(parents=True, exist_ok=True)
        logger.warning(f"Created docs directory: {docs_dir}")
        return
    
    html_files = list(docs_dir.glob("*.html"))
    logger.info(f"Found {len(html_files)} HTML files")
    
    if not html_files:
        logger.warning("No HTML files in docs directory")
        return
    
    all_chunks, all_ids, all_metas = [], [], []
    
    for file_path in html_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                html = f.read()
            
            chunks = parse_html_with_structure(html, file_path.name)
            chunks = smart_chunk_texts(chunks)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{file_path.stem}_{i}_{hashlib.md5(chunk['text'][:100].encode()).hexdigest()[:8]}"
                all_chunks.append(chunk["text"])
                all_ids.append(chunk_id)
                all_metas.append(chunk["metadata"])
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    if not all_chunks:
        logger.warning("No chunks to index")
        return
    
    # Добавляем батчами
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i+batch_size]
        batch_ids = all_ids[i:i+batch_size]
        batch_metas = all_metas[i:i+batch_size]
        
        embeddings = []
        for text in batch_chunks:
            emb = asyncio.run(get_embedding(text))
            embeddings.append(emb)
        
        collection.add(ids=batch_ids, documents=batch_chunks, metadatas=batch_metas, embeddings=embeddings)
        logger.info(f"Indexed {min(i+batch_size, len(all_chunks))}/{len(all_chunks)} chunks")
    
    logger.info(f"Indexing complete: {len(all_chunks)} chunks")

# ==================== ПОИСК С ПРИОРИТЕТАМИ ====================
def query_with_priority(collection: chromadb.Collection, query_embedding: List[float], 
                       user_query: str, n_results: int = MAX_SEARCH_RESULTS) -> Dict:
    """Векторный поиск с учётом приоритета источников"""
    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results * 2,
        include=["documents", "metadatas", "distances"],
    )
    
    if not raw["documents"][0]:
        return raw
    
    scored = []
    for doc, meta, dist in zip(raw["documents"][0], raw["metadatas"][0], raw["distances"][0]):
        priority = meta.get("priority", 1.0)
        score = (1 - dist) * priority
        scored.append((score, doc, meta, dist))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:n_results]
    
    return {
        "documents": [[item[1] for item in top]],
        "metadatas": [[item[2] for item in top]],
        "distances": [[item[3] for item in top]],
    }

# ==================== GROQ API ====================
@retry_api(max_attempts=3, delay=1.0, backoff=2.0)
async def call_groq(prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
    """Вызов Groq API"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    
    # Обрезаем промпт, если слишком длинный
    if len(prompt) > MAX_PROMPT_TOKENS * 4:
        prompt = prompt[:MAX_PROMPT_TOKENS * 4] + "\n\n[...обрезано...]"
        logger.warning("Prompt truncated")
    
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 1.0,
    }
    
    response = await asyncio.to_thread(
        requests.post, url, headers=headers, json=payload, timeout=60
    )
    response.raise_for_status()
    
    return response.json()["choices"][0]["message"]["content"].strip()

# ==================== ПРОМПТ ====================
def build_prompt(context_chunks: List[str], user_query: str, history: str = "") -> str:
    """Собирает промпт для LLM"""
    context_text = "\n\n---\n\n".join([
        f"[Источник: {c.get('source', 'unknown')}] " + (c["text"] if isinstance(c, dict) else c)
        for c in context_chunks
    ])
    
    base = """Ты — эксперт лизинговой компании «ЛК ПроДвижение».
Отвечай ТОЛЬКО на основе контекста из документации.

ПРАВИЛА:
1. Есть информация → дай точный ответ с цифрами.
2. Противоречия → укажи и приведи варианты.
3. Нет информации → задай уточняющий вопрос.
4. Числа выделяй **жирным**.
5. Списки оформляй маркированными.
6. Отвечай на русском, профессионально.

Контекст:
{context}

{history}

Вопрос: {question}

Ответ:"""
    
    history_part = f"\nИстория:\n{history}\n" if history else ""
    return base.format(context=context_text, history=history_part, question=user_query)

# ==================== ПАМЯТЬ ДИАЛОГА ====================
user_conversations: Dict[int, List[Dict]] = {}

def add_to_history(chat_id: int, role: str, content: str):
    if chat_id not in user_conversations:
        user_conversations[chat_id] = []
    history = user_conversations[chat_id]
    history.append({"role": role, "content": content, "ts": time.time()})
    if len(history) > MAX_HISTORY_MESSAGES * 2:
        user_conversations[chat_id] = history[-MAX_HISTORY_MESSAGES * 2:]

def get_history(chat_id: int) -> str:
    history = user_conversations.get(chat_id, [])
    if not history:
        return ""
    msgs = [f"{m['role']}: {m['content']}" for m in history[:-1]]
    return "\n".join(msgs[-MAX_HISTORY_MESSAGES:])

# ==================== ОБРАБОТЧИКИ ====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await update.message.reply_text(
        f"Здравствуйте, {user.first_name}! 👋\n\n"
        "Я — бот-помощник ЛК ПроДвижение.\n"
        "Отвечаю на вопросы по:\n"
        "• Условиям лизинга и лимитам\n"
        "• Процессам оформления сделок\n"
        "• Документам и требованиям\n"
        "• Изменениям в договорах\n\n"
        "Напишите вопрос — я найду ответ в базе. 📚"
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📋 Команды:\n"
        "/start — начать\n"
        "/help — справка\n"
        "/reindex — [админ] переиндексировать документы\n"
        "/stats — [админ] статистика\n"
        "/clear — очистить историю\n\n"
        "💡 Советы:\n"
        "• Задавайте конкретные вопросы\n"
        "• Указывайте тип клиента (ИП/ЮЛ) и выручку"
    )

async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_conversations[chat_id] = []
    await response_cache.clear()
    await update.message.reply_text("🗑️ История очищена.")

async def reindex_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_USER_ID:
        await update.message.reply_text("❌ Доступ только администратору.")
        return
    
    msg = await update.message.reply_text("🔄 Переиндексация...")
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        try:
            client.delete_collection(COLLECTION_NAME)
        except:
            pass
        collection = init_chroma()
        index_documents(collection)
        await embedding_cache.clear()
        await response_cache.clear()
        await msg.edit_text("✅ Готово! Кэши очищены.")
    except Exception as e:
        logger.error(f"Reindex error: {e}")
        await msg.edit_text(f"❌ Ошибка: {e}")

async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_USER_ID:
        await update.message.reply_text("❌ Доступ только администратору.")
        return
    
    total_users = len(user_conversations)
    total_msgs = sum(len(h) for h in user_conversations.values())
    
    await update.message.reply_text(
        f"📊 Статистика:\n"
        f"• Диалогов: {total_users}\n"
        f"• Сообщений: {total_msgs}\n"
        f"• Кэш эмбеддингов: {len(embedding_cache)}\n"
        f"• Кэш ответов: {len(response_cache)}\n"
        f"• Модель: {GROQ_MODEL}\n"
        f"• Эмбеддинги: {EMBEDDING_MODEL.split('/')[-1]}"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    chat_id = update.effective_chat.id
    text = update.message.text.strip()
    
    if not text:
        return
    
    logger.info(f"User {user.id}: {text[:100]}...")
    add_to_history(chat_id, "user", text)
    
    await update.message.chat.send_action(constants.ChatAction.TYPING)
    
    # Кэш ответов
    cache_key = f"resp:{chat_id}:{hashlib.md5(text.encode()).hexdigest()}"
    cached = await response_cache.get(cache_key)
    if cached:
        await update.message.reply_text(cached)
        add_to_history(chat_id, "assistant", cached)
        return
    
    try:
        # 1. Эмбеддинг
        query_emb = await get_embedding(text)
        
        # 2. Поиск
        collection = init_chroma()
        results = query_with_priority(collection, query_emb, text)
        
        if not results["documents"][0]:
            fallback = (
                "❓ Не нашёл точной информации.\n"
                "Уточните:\n"
                "• Тип клиента (ИП/ЮЛ)?\n"
                "• Выручка?\n"
                "• Тип предмета лизинга?\n"
                "• Конкретный параметр?"
            )
            await update.message.reply_text(fallback)
            add_to_history(chat_id, "assistant", fallback)
            return
        
        # 3. Промпт и ответ
        context_chunks = results["documents"][0]
        history = get_history(chat_id)
        prompt = build_prompt(context_chunks, text, history)
        
        start_time = time.time()
        answer = await call_groq(prompt)
        logger.info(f"Groq: {time.time()-start_time:.2f}s, {len(answer)} chars")
        
        await response_cache.set(cache_key, answer)
        await update.message.reply_text(answer)
        add_to_history(chat_id, "assistant", answer)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        await update.message.reply_text(
            "⚠️ Ошибка обработки.\nПопробуйте ещё раз или напишите администратору."
        )

# ==================== ЗАПУСК ====================
def main():
    """Точка входа"""
    # Создаём папки
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Индексация при первом запуске
    if not list(CHROMA_PERSIST_DIR.glob("*")):
        logger.info("First run: indexing documents...")
        collection = init_chroma()
        index_documents(collection)
    else:
        logger.info("ChromaDB ready")
    
    # Приложение
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Обработчики
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("clear", clear_history))
    
    if ADMIN_USER_ID > 0:
        app.add_handler(CommandHandler("reindex", reindex_cmd))
        app.add_handler(CommandHandler("stats", stats_cmd))
    
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logger.info("Bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()