import os
import logging
import glob
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import groq
from fastembed import TextEmbedding
import chromadb
from bs4 import BeautifulSoup

# ========== НАСТРОЙКА ПЕРЕМЕННЫХ ==========
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN не задан")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY не задан")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== ИНИЦИАЛИЗАЦИЯ КЛИЕНТОВ ==========
groq_client = groq.Groq(api_key=GROQ_API_KEY)

# FastEmbed (лёгкие эмбеддинги)
logger.info("Загрузка модели эмбеддингов FastEmbed...")
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
logger.info("Модель эмбеддингов загружена")

# ChromaDB (векторная БД)
CHROMA_PATH = "./chroma_db"
DOCS_DIR = "/app/data/docs"          # папка с документами (создаётся автоматически)

os.makedirs(DOCS_DIR, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="docs")

# ========== ФУНКЦИИ РАБОТЫ С ДОКУМЕНТАМИ ==========
def extract_text_from_html(html_path):
    """Извлечь текст из HTML-файла, удалив скрипты и стили."""
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
        for tag in soup(['script', 'style', 'comment']):
            tag.decompose()
        text = soup.get_text(separator='\n')
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return '\n'.join(lines)

def index_documents():
    """Проиндексировать все HTML-файлы из DOCS_DIR в ChromaDB."""
    html_files = glob.glob(os.path.join(DOCS_DIR, "*.html"))
    if not html_files:
        logger.warning(f"В папке {DOCS_DIR} нет HTML-файлов. Индексация не выполнена.")
        return

    logger.info(f"Найдено {len(html_files)} HTML-файлов. Начинаем индексацию...")
    for file_path in html_files:
        text = extract_text_from_html(file_path)
        if not text:
            logger.warning(f"Не удалось извлечь текст из {file_path}")
            continue

        # Разбиваем текст на чанки (простой способ: по 1000 символов с перекрытием)
        chunk_size = 1000
        overlap = 200
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])

        # Генерируем эмбеддинги для всех чанков
        embeddings = list(embedder.embed(chunks))

        file_name = os.path.basename(file_path)
        ids = [f"{file_name}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file_name} for _ in chunks]

        collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Индексирован {file_name}: {len(chunks)} чанков")

# Индексация при первом запуске (если коллекция пуста)
if collection.count() == 0:
    logger.info("Коллекция пуста, начинаем индексацию документов...")
    index_documents()
else:
    logger.info(f"Коллекция уже содержит {collection.count()} записей")

# ========== ОБРАБОТЧИКИ TELEGRAM ==========
async def start(update: Update, context):
    await update.message.reply_text(
        "Привет! Я ИИ-агент, обученный на вашей документации. Задайте вопрос, и я найду ответ."
    )

async def handle_message(update: Update, context):
    user_text = update.message.text
    logger.info(f"Пользователь {update.effective_user.id}: {user_text}")

    # 1. Преобразуем вопрос в эмбеддинг
    query_embedding = list(embedder.embed([user_text]))[0]

    # 2. Ищем похожие чанки
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas"]
    )

    if not results['documents'][0]:
        await update.message.reply_text("Извините, не нашёл информации по вашему вопросу.")
        return

    # 3. Собираем контекст
    context_chunks = results['documents'][0]
    sources = list(set(meta['source'] for meta in results['metadatas'][0]))
    context_text = "\n\n".join(context_chunks)

    # 4. Формируем промпт
    system_prompt = (
        "Ты — полезный помощник, который отвечает на вопросы, используя только предоставленный контекст. "
        "Если ответа нет в контексте, скажи, что не знаешь. Не добавляй информацию из своего знания.\n\n"
        "Контекст:\n{context}\n\nВопрос: {question}\n\nОтвет:"
    )
    prompt = system_prompt.format(context=context_text, question=user_text)

    # 5. Вызываем Groq
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",   # можно заменить на "llama-3.3-70b-versatile"
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        logger.exception("Ошибка при запросе к Groq")
        await update.message.reply_text("Произошла ошибка при генерации ответа. Попробуйте позже.")
        return

    source_line = f"\n\n📄 Источники: {', '.join(sources)}"
    final_answer = answer + source_line
    await update.message.reply_text(final_answer)

# ========== ЗАПУСК БОТА ==========
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Бот запущен в режиме long polling")
    app.run_polling()

if __name__ == "__main__":
    main()