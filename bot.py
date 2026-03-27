import os
import logging
import glob
from dotenv import load_dotenv
import telegram
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not TELEGRAM_TOKEN:
    logger.error("❌ TELEGRAM_TOKEN не задан")
    exit(1)
if not GROQ_API_KEY:
    logger.error("❌ GROQ_API_KEY не задан")
    exit(1)

# Инициализация Groq
groq_client = Groq(api_key=GROQ_API_KEY)

# Конфигурация векторной БД
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "docs"
DOCS_DIR = "documents"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)
TOP_K = 5

def extract_text_and_links(html_path):
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
        for tag in soup(['script', 'style', 'comment']):
            tag.decompose()
        links = [a['href'] for a in soup.find_all('a', href=True)]
        text = soup.get_text(separator='\n')
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = '\n'.join(lines)
        return text, list(set(links))

def ensure_collection():
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
        logger.info("Коллекция уже существует")
        return collection
    except chromadb.errors.InvalidCollectionException:
        logger.info("Коллекция не найдена, создаём новую")
        collection = chroma_client.create_collection(COLLECTION_NAME)

        html_files = glob.glob(os.path.join(DOCS_DIR, "*.html"))
        if not html_files:
            logger.warning(f"Папка {DOCS_DIR} пуста или не содержит HTML-файлов.")
            return collection

        logger.info(f"Найдено {len(html_files)} HTML-файлов. Индексация...")
        for file_path in html_files:
            logger.info(f"Индексируем {file_path}...")
            text, links = extract_text_and_links(file_path)
            if not text:
                continue
            chunks = text_splitter.split_text(text)
            if not chunks:
                continue

            embeddings = embedder.encode(chunks).tolist()
            file_name = os.path.basename(file_path)
            ids = [f"{file_name}_{i}" for i in range(len(chunks))]
            metadatas = [{"source": file_name, "links": ";".join(links)} for _ in chunks]

            collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"  Добавлено {len(chunks)} чанков")
        return collection

collection = ensure_collection()

async def start(update, context):
    await update.message.reply_text("Привет! Я бот-помощник по документации. Задай вопрос, и я найду ответ в нашей базе знаний.")

async def handle_message(update, context):
    user_text = update.message.text
    logger.info(f"Пользователь {update.effective_user.id}: {user_text}")

    # Эмбеддинг вопроса
    query_embedding = embedder.encode([user_text]).tolist()[0]

    # Поиск
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    if not results['documents'][0]:
        await update.message.reply_text("Извините, не нашёл информации по вашему вопросу.")
        return

    context_chunks = results['documents'][0]
    sources = list(set(meta['source'] for meta in results['metadatas'][0]))
    context_text = "\n\n".join(context_chunks)

    system_prompt = (
        "Ты — полезный помощник, который отвечает на вопросы, используя только предоставленный контекст. "
        "Если ответа нет в контексте, скажи, что не знаешь. Не добавляй информацию из своего знания.\n\n"
        "Контекст:\n{context}\n\nВопрос: {question}\n\nОтвет:"
    )
    prompt = system_prompt.format(context=context_text, question=user_text)

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Ошибка Groq: {e}")
        await update.message.reply_text("Произошла ошибка при генерации ответа. Попробуйте позже.")
        return

    source_line = f"\n\n📄 Источники: {', '.join(sources)}"
    final_answer = answer + source_line
    await update.message.reply_text(final_answer)

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Бот запущен в режиме long polling")
    app.run_polling(allowed_updates=telegram.Update.ALL_TYPES)

if __name__ == "__main__":
    main()