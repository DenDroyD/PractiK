#!/usr/bin/env python3
import os
import sys
import logging
import asyncio
from pathlib import Path
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

import groq

# ===== НАСТРОЙКИ =====
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
MODEL_NAME = "all-MiniLM-L3-v2"

# Проверка переменных окружения
if not TELEGRAM_TOKEN:
    print("❌ ОШИБКА: TELEGRAM_TOKEN не задан!", file=sys.stderr)
    sys.exit(1)
if not GROQ_API_KEY:
    print("❌ ОШИБКА: GROQ_API_KEY не задан!", file=sys.stderr)
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
CACHE_DIR = Path(DATA_DIR) / "models_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ===== ИНИЦИАЛИЗАЦИЯ КЛИЕНТОВ =====
try:
    groq_client = groq.Groq(api_key=GROQ_API_KEY)
    logger.info("✅ Groq клиент инициализирован")
except Exception as e:
    logger.error(f"❌ Ошибка инициализации Groq: {e}")
    groq_client = None

# ===== ЗАГРУЗКА МОДЕЛИ ЭМБЕДДИНГОВ =====
embedder = None
embedder_error = None

def _load_embedding_model():
    """Загрузка модели эмбеддингов"""
    global embedder, embedder_error
    try:
        logger.info(f"🚀 Начинаю загрузку модели {MODEL_NAME}...")
        
        # Пробуем импортировать sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            logger.warning(f"⚠️ sentence-transformers не установлен: {e}")
            logger.info("💡 Бот будет работать без RAG (векторного поиска)")
            embedder_error = "sentence-transformers не установлен"
            return False
        
        # Загружаем модель
        embedder = SentenceTransformer(
            MODEL_NAME,
            cache_folder=str(CACHE_DIR),
            device="cpu"
        )
        
        # Тестовый прогон
        _ = embedder.encode(["test"], batch_size=1, show_progress_bar=False)
        
        logger.info(f"✅ Модель {MODEL_NAME} загружена успешно!")
        return True
        
    except Exception as e:
        embedder_error = str(e)
        logger.exception(f"❌ Ошибка загрузки модели: {e}")
        logger.warning("💡 Бот продолжит работу без эмбеддингов")
        return False

# Загружаем модель при старте (в отдельном потоке)
logger.info("🔄 Инициализация модели эмбеддингов...")
try:
    loop = asyncio.get_event_loop()
    model_loaded = loop.run_in_executor(None, _load_embedding_model)
    # Ждем завершения загрузки (но не более 120 секунд)
    loop.run_until_complete(asyncio.wait_for(model_loaded, timeout=120))
except asyncio.TimeoutError:
    logger.error("⏰ Таймаут загрузки модели (120 сек)")
    embedder_error = "Таймаут загрузки модели"
except Exception as e:
    logger.error(f"❌ Ошибка при инициализации модели: {e}")
    embedder_error = str(e)

# ===== ОБРАБОТЧИКИ КОМАНД =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start"""
    status_model = "✅ загружена" if embedder else f"❌ не загружена"
    status_groq = "✅ подключен" if groq_client else "❌ ошибка"
    
    await update.message.reply_text(
        f"🤖 Бот запущен!\n\n"
        f"🧠 Модель эмбеддингов: {status_model}\n"
        f"🔗 Groq API: {status_groq}\n"
        f"📁 Данные: {DATA_DIR}\n\n"
        f"Отправьте сообщение для теста."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текстовых сообщений"""
    user_text = update.message.text
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    
    logger.info(f"👤 User {user_id} (chat {chat_id}): {user_text[:100]}")

    # 🔹 Шаг 1: Создание эмбеддинга (если модель доступна)
    if embedder:
        try:
            embedding = await asyncio.to_thread(
                lambda: embedder.encode([user_text], batch_size=1, show_progress_bar=False)[0]
            )
            logger.debug(f"🧮 Эмбеддинг создан: {len(embedding)} dim")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка создания эмбеддинга: {e}")
    else:
        logger.debug("⚠️ Модель эмбеддингов недоступна, пропускаем")

    # 🔹 Шаг 2: Запрос к Groq
    if not groq_client:
        await update.message.reply_text("❌ Groq API не доступен. Проверьте ключ.")
        return

    try:
        logger.info(f"🔄 Отправка запроса к Groq (модель: llama-3.1-8b-instant)...")
        
        completion = await asyncio.wait_for(
            asyncio.to_thread(
                lambda: groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": user_text}],
                    temperature=0.7,
                    max_tokens=512,
                    timeout=30
                )
            ),
            timeout=40
        )
        
        answer = completion.choices[0].message.content
        await update.message.reply_text(answer)
        logger.info(f"✅ Ответ отправлен пользователю {user_id}")

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
    logger.info(f"🧠 Модель: {MODEL_NAME}")
    logger.info(f"✅ Embedder: {'загружен' if embedder else 'не загружен'}")
    logger.info(f"✅ Groq: {'подключен' if groq_client else 'ошибка'}")
    logger.info("=" * 60)
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Регистрируем обработчики
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logger.info("✅ Бот запущен и слушает обновления...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()