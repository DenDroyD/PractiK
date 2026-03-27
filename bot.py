#!/usr/bin/env python3
import os
import sys
import logging
import asyncio
import numpy as np
from pathlib import Path
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from bs4 import BeautifulSoup

import groq

# ===== НАСТРОЙКИ =====
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
USE_RAG = os.getenv("USE_RAG", "false").lower() == "true"
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
DOCS_DIR = Path(DATA_DIR) / "docs"
MODEL_NAME = "all-MiniLM-L3-v2"

# Проверка переменных
if not TELEGRAM_TOKEN:
    print("❌ TELEGRAM_TOKEN не задан", file=sys.stderr)
    sys.exit(1)
if not GROQ_API_KEY:
    print("❌ GROQ_API_KEY не задан", file=sys.stderr)
    sys.exit(1)

# ===== ЛОГИРОВАНИЕ =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/app/bot.log", encoding="utf-8", mode="a")
    ]
)
logger = logging.getLogger(__name__)

# Создаем директории
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# ===== GROQ КЛИЕНТ =====
try:
    groq_client = groq.Groq(api_key=GROQ_API_KEY)
    logger.info("✅ Groq клиент инициализирован")
except Exception as e:
    logger.error(f"❌ Groq ошибка: {e}")
    groq_client = None

# ===== ЗАГРУЗКА HTML ДОКУМЕНТОВ =====
def load_html_documents(docs_dir: Path) -> list:
    """Загружает все HTML файлы и извлекает текст"""
    documents = []
    if not docs_dir.exists():
        logger.warning(f"📁 Папка документов не найдена: {docs_dir}")
        return documents
    
    html_files = list(docs_dir.glob("*.html")) + list(docs_dir.glob("*.htm"))
    logger.info(f"📚 Найдено HTML файлов: {len(html_files)}")
    
    for file_path in html_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Парсинг HTML
            soup = BeautifulSoup(content, 'html.parser')
            
            # Удаляем скрипты и стили
            for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                tag.decompose()
            
            # Извлекаем текст
            text = soup.get_text(separator=' ', strip=True)
            
            # Разбиваем на части по 1000 символов (для лучшей точности поиска)
            chunks = [text[i:i+1000] for i in range(0, len(text), 1000) if text[i:i+1000].strip()]
            
            for i, chunk in enumerate(chunks):
                documents.append({
                    "text": chunk.strip(),
                    "source": file_path.name,
                    "chunk": i + 1
                })
            
            logger.info(f"✅ Загружен {file_path.name}: {len(chunks)} частей")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки {file_path.name}: {e}")
    
    logger.info(f"📖 Всего загружено частей документов: {len(documents)}")
    return documents

# ===== МОДЕЛЬ ЭМБЕДДИНГОВ И ВЕКТОРНЫЙ ПОИСК =====
embedder = None
vector_store = None
documents = []

if USE_RAG:
    logger.info("=" * 60)
    logger.info("🔄 ИНИЦИАЛИЗАЦИЯ RAG СИСТЕМЫ")
    logger.info("=" * 60)
    
    # 1. Загрузка модели
    try:
        logger.info(f"🚀 Загрузка модели {MODEL_NAME}...")
        from sentence_transformers import SentenceTransformer
        
        cache_dir = Path(DATA_DIR) / "models_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        embedder = SentenceTransformer(
            MODEL_NAME,
            cache_folder=str(cache_dir),
            device="cpu"
        )
        
        # Тестовый прогон
        _ = embedder.encode(["test"], batch_size=1, show_progress_bar=False)
        logger.info(f"✅ Модель {MODEL_NAME} загружена успешно")
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки модели: {e}")
        logger.warning("💡 RAG будет отключён")
        embedder = None
        USE_RAG = False
    
    # 2. Загрузка документов (если модель загрузилась)
    if embedder:
        logger.info("📂 Загрузка документов...")
        documents = load_html_documents(DOCS_DIR)
        
        if documents:
            # 3. Создание эмбеддингов для всех документов
            try:
                logger.info("🧮 Создание векторных представлений...")
                doc_texts = [doc["text"] for doc in documents]
                
                # Создаём эмбеддинги пачками по 16 документов
                all_embeddings = []
                for i in range(0, len(doc_texts), 16):
                    batch = doc_texts[i:i+16]
                    embeddings = embedder.encode(
                        batch,
                        batch_size=16,
                        show_progress_bar=True,
                        convert_to_numpy=True
                    )
                    all_embeddings.extend(embeddings)
                    logger.info(f"📊 Обработано {min(i+16, len(doc_texts))}/{len(doc_texts)} частей")
                
                # Сохраняем как numpy array
                vector_store = np.array(all_embeddings)
                logger.info(f"✅ Векторное хранилище создано: {vector_store.shape}")
                
            except Exception as e:
                logger.error(f"❌ Ошибка создания эмбеддингов: {e}")
                vector_store = None
        else:
            logger.warning("⚠️ Документы не найдены. Добавьте HTML файлы в папку data/docs/")
            vector_store = None
else:
    logger.info("ℹ️  RAG отключён (USE_RAG=false)")

# ===== ФУНКЦИЯ ПОИСКА =====
def search_documents(query: str, top_k: int = 3) -> list:
    """Ищет наиболее релевантные документы"""
    if not embedder or vector_store is None or len(vector_store) == 0:
        return []
    
    try:
        # Создаём эмбеддинг запроса
        query_embedding = embedder.encode([query], convert_to_numpy=True)[0]
        
        # Косинусное сходство
        similarities = np.dot(vector_store, query_embedding) / (
            np.linalg.norm(vector_store, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Топ-K наиболее похожих
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Порог релевантности
                results.append({
                    **documents[idx],
                    "score": float(similarities[idx])
                })
        
        logger.info(f"🔍 Найдено {len(results)} релевантных частей")
        return results
        
    except Exception as e:
        logger.error(f"❌ Ошибка поиска: {e}")
        return []

# ===== ОБРАБОТЧИКИ КОМАНД =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start"""
    status_rag = "✅ включён" if (USE_RAG and embedder and vector_store is not None) else "❌ выключен"
    docs_count = len(documents) if documents else 0
    
    await update.message.reply_text(
        f"🤖 Бот запущен!\n\n"
        f"🧠 RAG (поиск по документам): {status_rag}\n"
        f"📚 Загружено документов: {docs_count}\n"
        f"🔗 Groq API: {'✅' if groq_client else '❌'}\n\n"
        f"Отправьте сообщение для теста."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текстовых сообщений"""
    user_text = update.message.text
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    
    logger.info(f"👤 User {user_id} (chat {chat_id}): {user_text[:100]}")

    # 🔹 RAG: поиск релевантных документов
    context_text = ""
    if USE_RAG and embedder and vector_store is not None:
        try:
            relevant_docs = search_documents(user_text, top_k=3)
            
            if relevant_docs:
                context_parts = []
                for doc in relevant_docs:
                    context_parts.append(
                        f"[{doc['source']} (часть {doc['chunk']}, релевантность: {doc['score']:.2f})]\n"
                        f"{doc['text']}"
                    )
                context_text = "\n\n".join(context_parts)
                logger.info(f"📎 Добавлен контекст из {len(relevant_docs)} документов")
            else:
                logger.info("ℹ️  Релевантные документы не найдены")
                
        except Exception as e:
            logger.error(f"❌ Ошибка RAG поиска: {e}")

    # 🔹 Формируем запрос к Groq
    if context_text:
        system_prompt = (
            "Вы — помощник, который отвечает на вопросы на основе предоставленных документов.\n"
            "Используйте только информацию из контекста. Если ответа нет в контексте, скажите об этом.\n"
            "Будьте кратки и точны."
        )
        
        groq_message = {
            "role": "user",
            "content": f"Контекст из документов:\n\n{context_text}\n\nВопрос пользователя: {user_text}"
        }
    else:
        system_prompt = "Вы — полезный помощник. Отвечайте кратко и по делу."
        groq_message = {"role": "user", "content": user_text}

    # 🔹 Запрос к Groq
    if not groq_client:
        await update.message.reply_text("❌ Groq API не доступен")
        return

    try:
        logger.info(f"🔄 Отправка запроса к Groq...")
        
        completion = await asyncio.wait_for(
            asyncio.to_thread(
                lambda: groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        groq_message
                    ],
                    temperature=0.7,
                    max_tokens=512,
                    timeout=30
                )
            ),
            timeout=40
        )
        
        answer = completion.choices[0].message.content
        await update.message.reply_text(answer)
        logger.info(f"✅ Ответ отправлен (длина: {len(answer)} симв.)")

    except asyncio.TimeoutError:
        logger.error("⏰ Таймаут запроса к Groq")
        await update.message.reply_text("⏰ Превышено время ожидания ответа. Попробуйте позже.")
    except Exception as e:
        logger.exception(f"❌ Ошибка Groq API: {e}")
        await update.message.reply_text(f"❌ Ошибка: {str(e)[:200]}")

# ===== ЗАПУСК БОТА =====
def main():
    logger.info("=" * 60)
    logger.info("🚀 ЗАПУСК БОТА")
    logger.info(f"📁 DATA_DIR: {DATA_DIR}")
    logger.info(f"📂 DOCS_DIR: {DOCS_DIR}")
    logger.info(f"🧠 RAG: {'включён' if USE_RAG else 'выключен'}")
    logger.info(f"✅ Embedder: {'загружен' if embedder else 'не загружен'}")
    logger.info(f"✅ Vector Store: {'создан' if vector_store is not None else 'не создан'}")
    logger.info(f"✅ Groq: {'подключен' if groq_client else 'ошибка'}")
    logger.info("=" * 60)
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logger.info("✅ Бот запущен и слушает обновления...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()