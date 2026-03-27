#!/usr/bin/env python3
import os
import sys
import logging
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

import groq

# ===== НАСТРОЙКИ =====
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
USE_RAG = os.getenv("USE_RAG", "false").lower() == "true"  # Флаг для RAG

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
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ===== GROQ КЛИЕНТ =====
try:
    groq_client = groq.Groq(api_key=GROQ_API_KEY)
    logger.info("✅ Groq клиент инициализирован")
except Exception as e:
    logger.error(f"❌ Groq ошибка: {e}")
    groq_client = None

# ===== ЭМБЕДДИНГИ (опционально) =====
embedder = None
if USE_RAG:
    logger.info("🔄 Попытка загрузить модель эмбеддингов...")
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer('all-MiniLM-L3-v2', device='cpu')
        # Тестовый прогон
        _ = embedder.encode(["test"], batch_size=1, show_progress_bar=False)
        logger.info("✅ Модель эмбеддингов загружена")
    except Exception as e:
        logger.warning(f"⚠️ Не удалось загрузить модель: {e}")
        logger.warning("💡 Бот продолжит работу БЕЗ векторного поиска")
        embedder = None
else:
    logger.info("ℹ️  RAG отключен (USE_RAG=false). Бот работает в базовом режиме.")

# ===== ОБРАБОТЧИКИ =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_rag = "✅ включён" if (USE_RAG and embedder) else "❌ выключен"
    await update.message.reply_text(
        f"🤖 Бот запущен!\n"
        f"🧠 RAG (векторный поиск): {status_rag}\n"
        f"🔗 Groq API: {'✅' if groq_client else '❌'}\n\n"
        f"Отправьте сообщение для теста."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    user_id = update.effective_user.id
    logger.info(f"👤 User {user_id}: {user_text[:100]}")

    # 🔹 Опционально: создание эмбеддинга
    if USE_RAG and embedder:
        try:
            # Блокирующая операция — в отдельном потоке!
            embedding = await asyncio.to_thread(
                lambda: embedder.encode([user_text], batch_size=1, show_progress_bar=False)[0]
            )
            logger.debug(f"🧮 Эмбеддинг: {len(embedding)} dim")
            # Здесь позже будет поиск по векторной БД
        except Exception as e:
            logger.warning(f"⚠️ Ошибка эмбеддинга: {e}")

    # 🔹 Запрос к Groq
    if not groq_client:
        await update.message.reply_text("❌ Groq API не доступен")
        return

    try:
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
        logger.info(f"✅ Ответ отправлен")

    except asyncio.TimeoutError:
        logger.error("⏰ Таймаут Groq")
        await update.message.reply_text("⏰ Превышено время ожидания. Попробуйте позже.")
    except Exception as e:
        logger.exception(f"❌ Ошибка: {e}")
        await update.message.reply_text(f"❌ Ошибка: {str(e)[:200]}")

# ===== ЗАПУСК =====
def main():
    logger.info("🚀 Запуск бота...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("✅ Бот слушает обновления")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()