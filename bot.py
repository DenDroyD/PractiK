import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import groq
from sentence_transformers import SentenceTransformer

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN не задан")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY не задан")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = groq.Groq(api_key=GROQ_API_KEY)

# Загрузка лёгкой модели эмбеддингов
logger.info("🚀 Начинаем загрузку модели all-MiniLM-L3-v2...")
try:
    embedder = SentenceTransformer('all-MiniLM-L3-v2')
    logger.info("✅ Модель эмбеддингов загружена успешно")
except Exception as e:
    logger.exception("❌ Ошибка загрузки модели эмбеддингов")
    embedder = None

async def start(update: Update, context):
    if embedder:
        await update.message.reply_text("Бот работает с лёгкой моделью эмбеддингов. Задай вопрос.")
    else:
        await update.message.reply_text("Модель эмбеддингов не загружена, но Groq работает.")

async def handle_message(update: Update, context):
    user_text = update.message.text
    logger.info(f"Пользователь: {user_text}")

    # Если модель загружена, тестируем эмбеддинг (один раз)
    if embedder:
        try:
            embedding = embedder.encode([user_text])
            logger.info(f"Эмбеддинг создан, размер: {len(embedding[0])}")
        except Exception as e:
            logger.exception("Ошибка при создании эмбеддинга")
            await update.message.reply_text("Ошибка при создании эмбеддинга.")
            return

    # Вызов Groq
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": user_text}],
            temperature=0.7,
            max_tokens=256
        )
        answer = completion.choices[0].message.content
        await update.message.reply_text(answer)
    except Exception as e:
        logger.exception("Ошибка при запросе к Groq")
        await update.message.reply_text("Ошибка при обращении к Groq. Попробуйте позже.")

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()