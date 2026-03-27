import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

# Токен из переменной окружения
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN не задан")

logging.basicConfig(level=logging.INFO)

async def start(update: Update, context):
    await update.message.reply_text("Привет! Я бот-эхо. Отправь мне любое сообщение, и я повторю его.")

async def echo(update: Update, context):
    user_text = update.message.text
    await update.message.reply_text(f"Вы сказали: {user_text}")

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    app.run_polling()

if __name__ == "__main__":
    main()