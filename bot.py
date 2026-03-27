import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import groq

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN не задан")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY не задан")

logging.basicConfig(level=logging.INFO)

client = groq.Groq(api_key=GROQ_API_KEY)

async def start(update: Update, context):
    await update.message.reply_text("Привет! Я бот, который отвечает с помощью Groq. Задай вопрос.")

async def handle_message(update: Update, context):
    user_text = update.message.text
    logging.info(f"Пользователь: {user_text}")
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # быстрая модель
            messages=[{"role": "user", "content": user_text}],
            temperature=0.7,
            max_tokens=256
        )
        answer = completion.choices[0].message.content
        await update.message.reply_text(answer)
    except Exception as e:
        logging.exception("Ошибка при запросе к Groq")
        await update.message.reply_text("Ошибка при обращении к Groq. Попробуйте позже.")

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()