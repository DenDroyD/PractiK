#!/usr/bin/env python3
import os
import logging
import asyncio
from pathlib import Path
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

import groq
from sentence_transformers import SentenceTransformer

# ===== НАСТРОЙКИ =====
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "all-MiniLM-L3-v2"  # лёгкая модель: 61 МБ, 384-dim векторы [[40]]
CACHE_DIR = Path("models_cache")  # кэш для модели
CACHE_DIR.mkdir(exist_ok=True)

if not TELEGRAM_TOKEN:
    raise ValueError("❌ TELEGRAM_TOKEN не задан в переменных окружения")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY не задан в переменных окружения")

# ===== ЛОГИРОВАНИЕ =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log", encoding="utf-8", mode="a")
    ]
)
logger = logging.getLogger(__name__)

# ===== ИНИЦИАЛИЗАЦИЯ КЛИЕНТОВ =====
groq_client = groq.Groq(api_key=GROQ_API_KEY)

# ===== ЗАГРУЗКА МОДЕЛИ ЭМБЕДДИНГОВ (с кэшированием) =====
embedder = None
embedder_error = None

def _load_embedding_model():
    """Синхронная функция загрузки — будет вызвана в потоке"""
    global embedder, embedder_error
    try:
        logger.info(f"🚀 Загрузка модели {MODEL_NAME} в {CACHE_DIR}...")
        embedder = SentenceTransformer(
            MODEL_NAME,
            cache_folder=str(CACHE_DIR),  # кэшируем модель локально
            device="cpu"  # явно указываем CPU для совместимости
        )
        # Тестовый прогон для "прогрева"
        _ = embedder.encode(["test"], batch_size=1, show_progress_bar=False)
        logger.info("✅ Модель эмбеддингов загружена и готова к работе")
        return True
    except Exception as e:
        embedder_error = str(e)
        logger.exception(f"❌ Ошибка загрузки модели: {e}")
        return False

# Загружаем модель при старте (в отдельном потоке, чтобы не блокировать)
logger.info("🔄 Инициализация модели эмбеддингов...")
loop = asyncio.get_event_loop()
model_loaded = loop.run_in_executor(None, _load_embedding_model)

# ===== ОБРАБОТЧИКИ КОМАНД =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start"""
    status = "✅ загружена" if embedder else f"❌ не загружена: {embedder_error}" if embedder_error else "🔄 загружается..."
    await update.message.reply_text(
        f"🤖 Бот запущен!\n"
        f"🧠 Модель эмбеддингов: {status}\n"
        f"🔗 Groq API: подключён\n"
        f"\nОтправьте сообщение для теста."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текстовых сообщений"""
    user_text = update.message.text
    user_id = update.effective_user.id
    logger.info(f"👤 User {user_id}: {user_text[:100]}")

    # 🔹 Шаг 1: Создание эмбеддинга (если модель доступна) — В ОТДЕЛЬНОМ ПОТОКЕ!
    embedding = None
    if embedder:
        try:
            # ⚠️ Критично: блокирующую операцию выполняем в потоке!
            embedding = await asyncio.to_thread(
                lambda: embedder.encode([user_text], batch_size=1, show_progress_bar=False)[0]
            )
            logger.debug(f"🧮 Эмбеддинг создан: {len(embedding)} dim")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка создания эмбеддинга: {e}")
            await update.message.reply_text("⚠️ Предупреждение: не удалось создать эмбеддинг, но запрос к ИИ продолжен.")

    # 🔹 Шаг 2: Запрос к Groq (с таймаутом)
    try:
        # Для RAG: здесь можно добавить контекст из векторной БД
        # Сейчас — простой прокси-запрос к модели
        completion = await asyncio.wait_for(
            asyncio.to_thread(
                lambda: groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": user_text}],
                    temperature=0.7,
                    max_tokens=512,
                    timeout=30  # таймаут запроса
                )
            ),
            timeout=40  # общий таймаут ожидания
        )
        answer = completion.choices[0].message.content
        await update.message.reply_text(answer)
        logger.info(f"✅ Ответ отправлен пользователю {user_id}")

    except asyncio.TimeoutError:
        logger.error("⏰ Таймаут запроса к Groq")
        await update.message.reply_text("⏰ Превышено время ожидания ответа. Попробуйте позже.")
    except Exception as e:
        logger.exception(f"❌ Ошибка Groq API: {e}")
        await update.message.reply_text("❌ Ошибка при обращении к ИИ. Проверьте логи или попробуйте позже.")

# ===== ЗАПУСК БОТА =====
def main():
    logger.info("🚀 Запуск бота...")
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Регистрируем обработчики
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # 🔥 Важно: разрешаем параллельную обработку обновлений
    # По умолчанию PTB обрабатывает обновления последовательно (blocking)
    # Для продакшена можно добавить: app.run_polling(allowed_updates=Update.ALL_TYPES)
    
    logger.info("✅ Бот запущен и слушает обновления...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()