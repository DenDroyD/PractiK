#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram RAG-бот для ЛК ПроДвижение
Версия: 2.3 (исправлен SyntaxError f-string)
Совместимость: chromadb==0.4.22, numpy==1.26.4
"""

import os
import logging
import glob
import hashlib
import time
import requests
from telegram import Update, constants
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import groq
import chromadb
from bs4 import BeautifulSoup

# ==================== КОНФИГУРАЦИЯ ====================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
ADMIN_USER_ID = int(os.getenv("ADMIN_USER_ID", "477810377"))

if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN не задан")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY не задан")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN не задан")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Groq клиент
client = groq.Groq(api_key=GROQ_API_KEY)

# Пути
CHROMA_PATH = "/app/data/chroma_db"
DOCS_DIR = "/app/data/docs"
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

# ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="docs")

# Настройки
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
MAX_CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_SEARCH_RESULTS = 5

# Кэш ответов (TTL 24 часа)
response_cache = {}
CACHE_TTL = 86400

# История диалогов
user_history = {}

# ==================== ЭМБЕДДИНГИ ====================
def get_embedding(text: str):
    """Получает эмбеддинг через HuggingFace API"""
    url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/pipeline/feature-extraction"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": [text],
        "options": {"wait_for_model": True}
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()[0]
    except Exception as e:
        logger.exception(f"Ошибка эмбеддинга: {e}")
        return None

# ==================== ПАРАСИНГ HTML ====================
def extract_text_from_html(html_path):
    """Извлекает текст из HTML с сохранением структуры"""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
        
        # Удаляем ненужные элементы
        for tag in soup(['script', 'style', 'comment']):
            tag.decompose()
        
        # Особая обработка таблиц из processy-lizingovoi-sdelki.html
        if 'processy-lizingovoi-sdelki' in html_path:
            tables_text = []
            for table in soup.find_all('table'):
                rows = []
                for tr in table.find_all('tr'):
                    cells = [td.get_text(separator=' ', strip=True) for td in tr.find_all(['td', 'th'])]
                    if cells and any(c.strip() for c in cells):
                        rows.append(' | '.join(cells))
                if rows:
                    tables_text.append('\n'.join(rows))
            if tables_text:
                return '[ТАБЛИЦА ПРОЦЕССА]\n' + '\n\n'.join(tables_text)
        
        text = soup.get_text(separator='\n')
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return '\n'.join(lines)
    except Exception as e:
        logger.error(f"Ошибка парсинга {html_path}: {e}")
        return None

# ==================== ИНДЕКСАЦИЯ ====================
def index_documents():
    """Индексирует HTML-документы в ChromaDB"""
    files = glob.glob(os.path.join(DOCS_DIR, "*.html"))
    if not files:
        logger.warning(f"В {DOCS_DIR} нет HTML-файлов")
        return
    
    logger.info(f"Найдено {len(files)} файлов. Начинаем индексацию...")
    
    for file_path in files:
        text = extract_text_from_html(file_path)
        if not text:
            continue
        
        # Разбиение на чанки
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + MAX_CHUNK_SIZE, len(text))
            chunks.append(text[start:end])
            start += MAX_CHUNK_SIZE - CHUNK_OVERLAP
        
        # Эмбеддинги
        embeddings = []
        for chunk in chunks:
            emb = get_embedding(chunk)
            if emb is None:
                logger.error(f"Не удалось получить эмбеддинг для {file_path}")
                continue
            embeddings.append(emb)
        
        if not embeddings:
            continue
        
        file_name = os.path.basename(file_path)
        ids = [f"{file_name}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file_name} for _ in chunks]
        
        collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Индексирован {file_name}: {len(chunks)} чанков")
    
    logger.info(f"Индексация завершена. Всего записей: {collection.count()}")

# ==================== ОБРАБОТЧИКИ TELEGRAM ====================
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
    if chat_id in user_history:
        del user_history[chat_id]
    await update.message.reply_text("🗑️ История очищена.")

async def reindex_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_USER_ID:
        await update.message.reply_text("❌ Доступ только администратору.")
        return
    
    msg = await update.message.reply_text("🔄 Переиндексация...")
    try:
        chroma_client.delete_collection(name="docs")
        global collection
        collection = chroma_client.get_or_create_collection(name="docs")
        index_documents()
        response_cache.clear()
        await msg.edit_text("✅ Готово! Документы переиндексированы.")
    except Exception as e:
        logger.error(f"Reindex error: {e}")
        await msg.edit_text(f"❌ Ошибка: {e}")

async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_USER_ID:
        await update.message.reply_text("❌ Доступ только администратору.")
        return
    
    await update.message.reply_text(
        f"📊 Статистика:\n"
        f"• Записей в базе: {collection.count()}\n"
        f"• Активных диалогов: {len(user_history)}\n"
        f"• Кэш ответов: {len(response_cache)}\n"
        f"• Модель: {GROQ_MODEL}"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    chat_id = update.effective_chat.id
    user_text = update.message.text.strip()
    
    if not user_text:
        return
    
    logger.info(f"User {user.id}: {user_text[:100]}...")
    
    await update.message.chat.send_action(constants.ChatAction.TYPING)
    
    # Кэш ответов
    cache_key = f"{chat_id}:{hashlib.md5(user_text.encode()).hexdigest()}"
    current_time = time.time()
    
    if cache_key in response_cache:
        cached_answer, cached_time = response_cache[cache_key]
        if current_time - cached_time < CACHE_TTL:
            logger.info(f"Cache HIT for chat {chat_id}")
            await update.message.reply_text(cached_answer)
            return
        else:
            del response_cache[cache_key]
    
    # Эмбеддинг запроса
    query_embedding = get_embedding(user_text)
    if query_embedding is None:
        await update.message.reply_text("❌ Ошибка при обращении к сервису эмбеддингов. Попробуйте позже.")
        return
    
    # Поиск в базе
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=MAX_SEARCH_RESULTS,
        include=["documents", "metadatas"]
    )
    
    if not results['documents'][0]:
        fallback = (
            "❓ Не нашёл точной информации.\n"
            "Уточните:\n"
            "• Тип клиента (ИП/ЮЛ)?\n"
            "• Выручка?\n"
            "• Тип предмета лизинга?\n"
            "• Конкретный параметр?"
        )
        await update.message.reply_text(fallback)
        return
    
    # Формируем контекст
    context_chunks = results['documents'][0]
    sources = list(set(meta['source'] for meta in results['metadatas'][0]))
    context_text = "\n\n---\n\n".join(context_chunks)
    
    # История диалога
    history = user_history.get(chat_id, [])[-6:]
    history_text = "\n".join([f"{h['role']}: {h['content']}" for h in history]) if history else ""
    
    # Промпт
    prompt = (
        "Ты — эксперт лизинговой компании «ЛК ПроДвижение».\n"
        "Отвечай ТОЛЬКО на основе контекста из документации.\n\n"
        "ПРАВИЛА:\n"
        "1. Есть информация → дай точный ответ с цифрами.\n"
        "2. Противоречия → укажи и приведи варианты.\n"
        "3. Нет информации → задай уточняющий вопрос.\n"
        "4. Числа выделяй **жирным**.\n"
        "5. Списки оформляй маркированными.\n"
        "6. Отвечай на русском, профессионально.\n\n"
        f"Контекст:\n{context_text}\n\n"
        f"{'История:\n' + history_text + '\n\n' if history_text else ''}"
        f"Вопрос: {user_text}\n\n"
        "Ответ:"
    )
    
    # Groq API
    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024
        )
        answer = completion.choices[0].message.content
        
        response_cache[cache_key] = (answer, current_time)
        
        if chat_id not in user_history:
            user_history[chat_id] = []
        user_history[chat_id].append({"role": "user", "content": user_text})
        user_history[chat_id].append({"role": "assistant", "content": answer})
        if len(user_history[chat_id]) > 10:
            user_history[chat_id] = user_history[chat_id][-10:]
        
        # Источники - ИСПРАВЛЕНО (без backslash в f-string)
        sources_str = ", ".join(sources)
        source_line = f"\n\n📄 Источники: {sources_str}"
        final_answer = answer + source_line
        
        await update.message.reply_text(final_answer)
        
    except Exception as e:
        logger.exception(f"Ошибка Groq: {e}")
        await update.message.reply_text("❌ Ошибка при обращении к Groq. Попробуйте позже.")

def main():
    """Точка входа"""
    logger.info("Запуск бота...")
    
    if collection.count() == 0:
        logger.info("Коллекция пуста, запускаем индексацию...")
        index_documents()
    else:
        logger.info(f"Коллекция содержит {collection.count()} записей")
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("clear", clear_history))
    
    if ADMIN_USER_ID > 0:
        app.add_handler(CommandHandler("reindex", reindex_cmd))
        app.add_handler(CommandHandler("stats", stats_cmd))
    
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logger.info("Бот запущен")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()