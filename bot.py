import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from sentence_transformers import SentenceTransformer

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN не задан")
# Groq пока не используем, но переменную проверим
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY не задан")

logging.basicConfig(level=logging.INFO)

# Пробуем загрузить модель
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    logging.info("Модель эмбеддингов загружена")
except Exception as e:
    logging.exception("Ошибка загрузки модели")
    raise

async def start(update: Update, context):
    await update.message.reply_text("Модель загружена, бот работает.")

async def handle_message(update: Update, context):
    # Пока просто эхо
    await update.message.reply_text("Я пока только тестирую модель эмбеддингов. Она загружена.")

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()