import os
import logging
import glob
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import groq
import chromadb
from bs4 import BeautifulSoup

# --- Конфигурация ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN не задан")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY не задан")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN не задан")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = groq.Groq(api_key=GROQ_API_KEY)

CHROMA_PATH = "./chroma_db"
DOCS_DIR = "/app/data/docs"
os.makedirs(DOCS_DIR, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="docs")

# --- Эмбеддинги через Hugging Face (мультиязычная модель) ---
def get_embedding(text: str):
    url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/pipeline/feature-extraction"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": [text],   # обязательно список!
        "options": {"wait_for_model": True}
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()[0]   # результат — список списков
    except Exception as e:
        logger.exception("Ошибка получения эмбеддинга от Hugging Face")
        return None

def extract_text_from_html(html_path):
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
        for tag in soup(['script', 'style', 'comment']):
            tag.decompose()
        text = soup.get_text(separator='\n')
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return '\n'.join(lines)

def index_documents():
    files = glob.glob(os.path.join(DOCS_DIR, "*.html"))
    if not files:
        logger.warning(f"В {DOCS_DIR} нет HTML-файлов")
        return

    logger.info(f"Найдено {len(files)} файлов. Начинаем индексацию...")
    for file_path in files:
        text = extract_text_from_html(file_path)
        if not text:
            continue
        chunk_size = 1000
        overlap = 200
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap

        embeddings = []
        for chunk in chunks:
            emb = get_embedding(chunk)
            if emb is None:
                logger.error(f"Не удалось получить эмбеддинг для {file_path}")
                return
            embeddings.append(emb)

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

if collection.count() == 0:
    logger.info("Коллекция пуста, запускаем индексацию...")
    index_documents()
else:
    logger.info(f"Коллекция уже содержит {collection.count()} записей")

async def start(update: Update, context):
    await update.message.reply_text("Привет! Я RAG-бот. Задай вопрос по документации (лизинг, страховки, агенты, поставщики).")

async def handle_message(update: Update, context):
    user_text = update.message.text
    logger.info(f"Пользователь {update.effective_user.id}: {user_text}")

    query_embedding = get_embedding(user_text)
    if query_embedding is None:
        await update.message.reply_text("Ошибка при обращении к сервису эмбеддингов. Попробуйте позже.")
        return

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,   # увеличили до 5
        include=["documents", "metadatas"]
    )

    if not results['documents'][0]:
        await update.message.reply_text("Извините, не нашёл информации по вашему вопросу.")
        return

    # Логирование найденных чанков для отладки
    logger.info(f"Найдено {len(results['documents'][0])} чанков")
    for i, doc in enumerate(results['documents'][0]):
        src = results['metadatas'][0][i]['source']
        logger.info(f"Чанк {i+1} (источник {src}): {doc[:300]}")

    context_chunks = results['documents'][0]
    sources = list(set(meta['source'] for meta in results['metadatas'][0]))
    context_text = "\n\n".join(context_chunks)

    prompt = (
        "Ты — полезный помощник, который отвечает на вопросы, используя предоставленный контекст. "
        "Если в контексте есть информация, дай полный и точный ответ, по возможности развёрнутый. "
        "Если информации нет, скажи, что не знаешь, и предложи уточнить вопрос.\n\n"
        "Контекст:\n{context}\n\nВопрос: {question}\n\nОтвет:"
    ).format(context=context_text, question=user_text)

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        logger.exception("Ошибка Groq")
        await update.message.reply_text("Ошибка при обращении к Groq. Попробуйте позже.")
        return

    source_line = f"\n\n📄 Источники: {', '.join(sources)}"
    final_answer = answer + source_line
    await update.message.reply_text(final_answer)

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()