import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN не задан")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY не задан")

logging.basicConfig(level=logging.INFO)

# Импорт модели будет отложен
# embedder = None

async def start(update: Update, context):
    await update.message.reply_text("Модель эмбеддингов загрузится при первом сообщении.")

async def handle_message(update: Update, context):
    global embedder
    # Загружаем модель при первом сообщении
    if 'embedder' not in context.bot_data:
        try:
            from sentence_transformers import SentenceTransformer
            logging.info("Загрузка модели эмбеддингов...")
            embedder = SentenceTransformer('all-MiniLM-L6-v2')
            context.bot_data['embedder'] = embedder
            logging.info("Модель загружена")
            await update.message.reply_text("Модель эмбеддингов загружена. Теперь можно задавать вопросы.")
        except Exception as e:
            logging.exception("Ошибка загрузки модели")
            await update.message.reply_text("Ошибка загрузки модели. Попробуйте позже.")
            return

    embedder = context.bot_data['embedder']

    # Пока просто эхо
    await update.message.reply_text("Модель загружена, бот работает.")

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()