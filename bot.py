import os
import logging
import glob
import asyncio
import time
import requests
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
ADMIN_ID = int(os.getenv("ADMIN_ID", 0))  # ваш Telegram ID для админских команд

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

# --- Инициализация ChromaDB ---
CHROMA_PATH = "./chroma_db"
DOCS_DIR = "/app/data/docs"   # папка с вашими HTML-файлами
os.makedirs(DOCS_DIR, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="docs")

# --- Глобальные переменные для BM25 ---
corpus = []          # список всех чанков (текстов)
bm25 = None          # объект BM25Okapi
tokenized_corpus = []  # токенизированная версия

# --- Настройки поиска ---
TOP_K_VECTOR = 10    # сколько чанков брать из векторного поиска
TOP_K_FINAL = 5      # сколько оставить после гибридного ранжирования
ALPHA = 0.6          # вес векторного поиска (1 - ALPHA вес BM25)

# --- Разбиение текста на чанки ---
# Сначала пробуем разбить по markdown-заголовкам
headers_to_split_on = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
# Обычный сплиттер на случай, если нет заголовков
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
    # Если текст содержит markdown-заголовки, разбиваем по ним
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
            # fallback
            return text_splitter.split_text(text)
    else:
        return text_splitter.split_text(text)

@lru_cache(maxsize=100)
def get_embedding(text: str):
    """Получение эмбеддинга через HF Inference API с кэшированием."""
    url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/pipeline/feature-extraction"
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = {"inputs": [text], "options": {"wait_for_model": True}}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()[0]   # список float
    except Exception as e:
        logger.exception("Ошибка получения эмбеддинга от Hugging Face")
        return None

def update_bm25():
    """Обновляет BM25-индекс на основе глобального corpus."""
    global bm25, tokenized_corpus
    if not corpus:
        bm25 = None
        tokenized_corpus = []
        return
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

def index_documents():
    """Индексирует все HTML-файлы в DOCS_DIR."""
    global corpus
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

        # Получаем эмбеддинги для всех чанков
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

    # Очищаем коллекцию и добавляем новые данные
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
    # Обновляем глобальный корпус для BM25
    corpus = all_chunks
    update_bm25()
    logger.info(f"Индексация завершена. Всего чанков: {len(all_chunks)}")

# При старте проверяем коллекцию
if collection.count() == 0:
    logger.info("Коллекция пуста, запускаем индексацию...")
    index_documents()
else:
    logger.info(f"Коллекция уже содержит {collection.count()} записей")
    # Для BM25 нужно загрузить все чанки в память (не обязательно, но удобно)
    all_data = collection.get(include=["documents"])
    corpus = all_data.get('documents', [])
    update_bm25()

# --- Память диалога ---
user_histories = {}  # chat_id -> список кортежей (role, text)

# --- Обработчики команд ---
async def start(update: Update, context):
    await update.message.reply_text(
        "Привет! Я RAG-бот. Задай вопрос по документации (лизинг, страховки, агенты, поставщики). "
        "Я помню историю нашего разговора, могу уточнить детали."
    )

async def reindex(update: Update, context):
    """Принудительная переиндексация (только для администратора)."""
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
    """Выполняет гибридный поиск: векторный + BM25."""
    # 1. Векторный поиск
    query_embedding = get_embedding(query)
    if query_embedding is None:
        logger.warning("Не удалось получить эмбеддинг запроса")
        return []

    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K_VECTOR,
        include=["documents", "metadatas", "distances"]
    )
    if not vector_results['documents'][0]:
        return []

    # 2. BM25 (если есть корпус)
    bm25_scores = {}
    if bm25 and corpus:
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)
        # Сопоставляем оценку с ID чанка (по порядку в corpus)
        # Но corpus может не соответствовать порядку, в котором вернулись результаты из Chroma.
        # Поэтому будем искать совпадение по тексту.
        # Упростим: для каждого чанка из векторного поиска найдём его BM25 оценку по индексу в corpus.
        # Построим словарь: текст чанка -> BM25 оценка (приблизительно)
        # Если чанк не найден в corpus (из-за разных токенизаций), оценка будет 0.
        text_to_score = {}
        for i, doc in enumerate(corpus):
            text_to_score[doc] = scores[i]
    else:
        text_to_score = {}

    # 3. Комбинируем
    combined = []
    for i, doc in enumerate(vector_results['documents'][0]):
        similarity = 1 - vector_results['distances'][0][i]  # чем меньше расстояние, тем лучше
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

    # --- 1. История диалога ---
    history = user_histories.get(chat_id, [])
    # Добавляем текущий вопрос
    history.append(("user", user_text))
    # Ограничиваем историю (например, последние 10 сообщений, но не более 5 пар)
    if len(history) > 10:
        history = history[-10:]
    user_histories[chat_id] = history

    # --- 2. Поиск ---
    search_results = hybrid_search(user_text)
    if not search_results:
        await update.message.reply_text("Извините, не нашёл информации по вашему вопросу.")
        return

    context_chunks = [res[1] for res in search_results]
    sources = list(set(res[2]['source'] for res in search_results))
    context_text = "\n\n".join(context_chunks)

    # --- 3. Формирование промпта с историей ---
    # Преобразуем историю в строку (без последнего вопроса, он уже есть в user_text)
    history_str = ""
    if len(history) > 1:  # есть предыдущие сообщения
        # Формируем строку вида "Пользователь: ...\nАссистент: ..."
        # Убираем последний элемент (текущий вопрос)
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

    # --- 4. Вызов Groq ---
    try:
        answer = call_groq_with_retry([{"role": "user", "content": prompt}])
    except Exception as e:
        logger.exception("Ошибка Groq")
        await update.message.reply_text("Произошла ошибка при генерации ответа. Попробуйте позже.")
        return

    # --- 5. Обработка уточняющих вопросов ---
    if answer.startswith("Уточните") or answer.startswith("Уточни"):
        # Просто отправляем ответ, не сохраняя его в историю как завершённый
        await update.message.reply_text(answer)
        # Не добавляем ответ в историю, чтобы бот не «запоминал» уточнение как факт
        return

    # --- 6. Сохранение ответа в историю ---
    history.append(("assistant", answer))
    user_histories[chat_id] = history

    source_line = f"\n\n📄 Источники: {', '.join(sources)}"
    final_answer = answer + source_line
    await update.message.reply_text(final_answer)

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reindex", reindex))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Бот запущен в режиме long polling")
    app.run_polling(allowed_updates=telegram.Update.ALL_TYPES)

if __name__ == "__main__":
    main()