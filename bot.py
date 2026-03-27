import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

# Загрузка переменных окружения
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Проверка токена
if not TELEGRAM_TOKEN:
    logger.error("TELEGRAM_TOKEN не задан")
    exit(1)

# Обработчик команды /start
async def start(update: Update, context):
    await update.message.reply_text("Привет! Я бот-помощник. Задай вопрос, и я отвечу.")

# Обработчик текстовых сообщений (эхо для теста)
async def echo(update: Update, context):
    user_text = update.message.text
    logger.info(f"Получено сообщение: {user_text}")
    await update.message.reply_text(f"Вы сказали: {user_text}")

def main():
    # Создаём приложение
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Добавляем обработчики
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Запускаем polling
    logger.info("Бот запущен в режиме Long Polling")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()