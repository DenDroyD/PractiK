import os
import logging
from flask import Flask, request
import telegram
import asyncio

app = Flask(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
DOMAIN = os.getenv("DOMAIN")

logging.basicConfig(level=logging.INFO)

if not TELEGRAM_TOKEN:
    logging.error("TELEGRAM_TOKEN не задан")
    exit(1)

bot = telegram.Bot(token=TELEGRAM_TOKEN)

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        update = telegram.Update.de_json(request.get_json(force=True), bot)
        if update.message:
            chat_id = update.message.chat_id
            text = update.message.text
            logging.info(f"Получено сообщение: {text}")
            # Отвечаем эхом
            bot.send_message(chat_id, f"Вы сказали: {text}")
        return "ok"
    except Exception as e:
        logging.exception("Ошибка в вебхуке")
        return "error", 500

@app.route('/')
def home():
    return "OK"

async def set_webhook():
    if not DOMAIN:
        logging.error("DOMAIN не задан")
        return
    webhook_url = f"https://{DOMAIN}/webhook"
    await bot.set_webhook(webhook_url)
    logging.info(f"✅ Вебхук установлен: {webhook_url}")

if __name__ == "__main__":
    # Устанавливаем вебхук (один раз)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(set_webhook())

    port = int(os.getenv("PORT", 3000))
    app.run(host="0.0.0.0", port=port)