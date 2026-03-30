import os
import logging
import glob
import asyncio
import time
import requests
import shutil
from functools import lru_cache
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import groq
import chromadb
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

# --- Конфигурация ---
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_USER_ID", os.getenv("ADMIN_ID", 0)))

if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN не задан")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY не задан")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN не задан")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Инициализация Groq ---
groq_client = groq.Groq(api_key=GROQ_API_KEY)

# --- Настройки путей ---
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db")
DOCS_DIR = os.path.join(DATA_DIR, "docs")
os.makedirs(DOCS_DIR, exist_ok=True)

# --- Инициализация ChromaDB ---
def init_chroma(force_recreate=False):
    if force_recreate and os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
            logger.info(f"🧹 Папка {CHROMA_PATH} удалена для чистой инициализации")
        except Exception as e:
            logger.warning(f"Не удалось удалить {CHROMA_PATH}: {e}")

    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection_name = "leasing_docs_v2"

    if force_recreate:
        try:
            client.delete_collection(collection_name)
            logger.info(f"✅ Коллекция {collection_name} удалена")
        except Exception:
            pass

    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"✅ Коллекция {collection_name} готова")
        return collection
    except Exception as e:
        logger.error(f"Ошибка при работе с коллекцией: {e}")
        try:
            client.delete_collection(collection_name)
        except:
            pass
        collection = client.create_collection(collection_name)
        logger.info(f"✅ Коллекция {collection_name} создана заново (с настройками по умолчанию)")
        return collection

chroma_client = None
collection = None

# --- Глобальные переменные для BM25 ---
corpus = []
bm25 = None
tokenized_corpus = []

# --- Настройки поиска ---
TOP_K_VECTOR = 20          # увеличено для лучшего охвата
TOP_K_FINAL = 5
ALPHA = 0.6

# --- Словарь для расшифровки сокращений ---
ABBREVIATIONS = {
    'ГТ': ['грузовой транспорт', 'грузовой', 'грузовик', 'грузоперевозки'],
    'ЛА': ['легковой автомобиль', 'легковой', 'легковушка'],
    'ЛКТ': ['легкий коммерческий транспорт', 'коммерческий транспорт', 'фургон', 'микроавтобус'],
    'СТ': ['спецтехника', 'специальная техника', 'строительная техника'],
    'ЛП': ['лизингополучатель', 'клиент', 'арендатор'],
    'ЮЛ': ['юридическое лицо', 'организация', 'компания'],
    'ИП': ['индивидуальный предприниматель'],
    'ДЛ': ['договор лизинга'],
    'ПЛ': ['предмет лизинга', 'оборудование', 'техника'],
    'БКК': ['большой кредитный комитет'],
    'МКК': ['малый кредитный комитет'],
    'УКА': ['управление кредитного анализа'],
    'СЭБ': ['служба экономической безопасности'],
    'LTV': ['loan to value', 'соотношение займа и стоимости'],
    'ROE': ['рентабельность собственного капитала'],
}

def expand_query(query: str) -> str:
    """Заменяет сокращения в запросе на полные варианты для улучшения поиска."""
    words = query.split()
    expanded_words = []
    for word in words:
        word_upper = word.upper()
        if word_upper in ABBREVIATIONS:
            expanded_words.append(word)
            expanded_words.extend(ABBREVIATIONS[word_upper])
        else:
            expanded_words.append(word)
    return " ".join(expanded_words) + " " + query

# --- Разбиение текста на чанки ---
headers_to_split_on = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

def extract_text_from_html(html_path):
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
        for tag in soup(['script', 'style', 'comment']):
            tag.decompose()
        text = soup.get_text(separator='\n')
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return '\n'.join(lines)

def chunk_text(text):
    if any(h in text for h in ["#", "##", "###"]):
        try:
            docs = md_splitter.split_text(text)
            chunks = []
            for doc in docs:
                header = doc.metadata.get("H1") or doc.metadata.get("H2") or doc.metadata.get("H3")
                full = f"{header}\n{doc.page_content}" if header else doc.page_content
                chunks.append(full)
            return chunks
        except Exception:
            return text_splitter.split_text(text)
    else:
        return text_splitter.split_text(text)

@lru_cache(maxsize=100)
def get_embedding(text: str):
    url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/pipeline/feature-extraction"
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = {"inputs": [text], "options": {"wait_for_model": True}}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()[0]
    except Exception as e:
        logger.exception("Ошибка получения эмбеддинга от Hugging Face")
        return None

def update_bm25():
    global bm25, tokenized_corpus
    if not corpus:
        bm25 = None
        tokenized_corpus = []
        return
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

def index_documents():
    global corpus, collection
    files = glob.glob(os.path.join(DOCS_DIR, "*.html"))
    if not files:
        logger.warning(f"В {DOCS_DIR} нет HTML-файлов")
        return

    logger.info(f"Найдено {len(files)} файлов. Начинаем индексацию...")
    all_chunks = []
    all_embeddings = []
    all_ids = []
    all_metadatas = []
    for file_path in files:
        text = extract_text_from_html(file_path)
        if not text:
            continue
        chunks = chunk_text(text)
        if not chunks:
            continue

        embeddings = []
        for chunk in chunks:
            emb = get_embedding(chunk)
            if emb is None:
                logger.error(f"Не удалось получить эмбеддинг для {file_path}, пропускаем")
                return
            embeddings.append(emb)

        file_name = os.path.basename(file_path)
        ids = [f"{file_name}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file_name} for _ in chunks]

        all_chunks.extend(chunks)
        all_embeddings.extend(embeddings)
        all_ids.extend(ids)
        all_metadatas.extend(metadatas)
        logger.info(f"Индексирован {file_name}: {len(chunks)} чанков")

    if not all_chunks:
        return

    try:
        chroma_client.delete_collection(collection.name)
        collection = chroma_client.create_collection(collection.name)
    except Exception:
        pass
    collection.add(
        embeddings=all_embeddings,
        documents=all_chunks,
        metadatas=all_metadatas,
        ids=all_ids
    )
    corpus = all_chunks
    update_bm25()
    logger.info(f"Индексация завершена. Всего чанков: {len(all_chunks)}")

# --- Память диалога ---
user_histories = {}

# --- Обработчики команд ---
async def start(update: Update, context):
    await update.message.reply_text(
        "Привет! Я RAG-бот. Задай вопрос по документации (лизинг, страховки, агенты, поставщики). "
        "Я помню историю нашего разговора, могу уточнить детали."
    )

async def reindex(update: Update, context):
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("У вас нет прав для этой команды.")
        return
    await update.message.reply_text("Начинаю переиндексацию...")
    try:
        index_documents()
        await update.message.reply_text("Переиндексация завершена.")
    except Exception as e:
        logger.exception("Ошибка при переиндексации")
        await update.message.reply_text(f"Ошибка: {e}")

def hybrid_search(query: str):
    expanded_query = expand_query(query)
    logger.debug(f"Расширенный запрос: {expanded_query}")

    query_embedding = get_embedding(expanded_query)
    if query_embedding is None:
        logger.warning("Не удалось получить эмбеддинг запроса")
        return []

    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K_VECTOR * 2,
        include=["documents", "metadatas", "distances"]
    )
    if not vector_results['documents'][0]:
        return []

    text_to_score = {}
    if bm25 and corpus:
        tokenized_query = expanded_query.split()
        scores = bm25.get_scores(tokenized_query)
        for i, doc in enumerate(corpus):
            text_to_score[doc] = scores[i]

    combined = []
    for i, doc in enumerate(vector_results['documents'][0]):
        similarity = 1 - vector_results['distances'][0][i]
        bm25_score = text_to_score.get(doc, 0.0)
        combined_score = ALPHA * similarity + (1 - ALPHA) * bm25_score
        combined.append((combined_score, doc, vector_results['metadatas'][0][i]))

    combined.sort(key=lambda x: x[0], reverse=True)
    return combined[:TOP_K_FINAL]

def call_groq_with_retry(messages, retries=3):
    for i in range(retries):
        try:
            completion = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.2,
                max_tokens=2048
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.warning(f"Ошибка Groq, попытка {i+1}/{retries}: {e}")
            if i == retries - 1:
                raise
            time.sleep(2 ** i)

async def handle_message(update: Update, context):
    chat_id = update.effective_chat.id
    user_text = update.message.text
    logger.info(f"Пользователь {update.effective_user.id}: {user_text}")

    history = user_histories.get(chat_id, [])
    history.append(("user", user_text))
    if len(history) > 10:
        history = history[-10:]
    user_histories[chat_id] = history

    search_results = hybrid_search(user_text)
    if not search_results:
        await update.message.reply_text("Извините, не нашёл информации по вашему вопросу.")
        return

    context_chunks = [res[1] for res in search_results]
    sources = list(set(res[2]['source'] for res in search_results))
    context_text = "\n\n".join(context_chunks)

    history_str = ""
    if len(history) > 1:
        prev = history[:-1]
        lines = []
        for role, text in prev:
            if role == "user":
                lines.append(f"Пользователь: {text}")
            else:
                lines.append(f"Ассистент: {text}")
        history_str = "\n".join(lines) + "\n"

    prompt = (
        f"{history_str}"
        f"Ты — полезный помощник, который отвечает на вопросы, используя только предоставленный контекст.\n"
        f"Если в контексте есть информация, дай полный, точный и развёрнутый ответ.\n"
        f"Если информация противоречива, сообщи об этом. Если ответ требует числовых данных, выдели их **жирным**.\n"
        f"Если ответ можно представить списком, используй маркированный список.\n"
        f"Если для ответа не хватает информации, задай уточняющий вопрос, а не выдумывай ответ.\n\n"
        f"Контекст:\n{context_text}\n\n"
        f"Вопрос: {user_text}\n\n"
        f"Ответ:"
    )

    try:
        answer = call_groq_with_retry([{"role": "user", "content": prompt}])
    except Exception as e:
        logger.exception("Ошибка Groq")
        await update.message.reply_text("Произошла ошибка при генерации ответа. Попробуйте позже.")
        return

    if answer.startswith("Уточните") or answer.startswith("Уточни"):
        await update.message.reply_text(answer)
        return

    history.append(("assistant", answer))
    user_histories[chat_id] = history

    source_line = f"\n\n📄 Источники: {', '.join(sources)}"
    final_answer = answer + source_line
    await update.message.reply_text(final_answer)

def main():
    global chroma_client, collection
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = init_chroma(force_recreate=False)

    if collection.count() == 0:
        logger.info("Коллекция пуста, запускаем индексацию...")
        index_documents()
    else:
        logger.info(f"Коллекция уже содержит {collection.count()} записей")
        all_data = collection.get(include=["documents"])
        global corpus
        corpus = all_data.get('documents', [])
        update_bm25()

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reindex", reindex))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Бот запущен в режиме long polling")
    app.run_polling(allowed_updates=telegram.Update.ALL_TYPES)

if __name__ == "__main__":
    main()