"""
RAG Telegram Bot — ЛК ПроДвижение
Полностью переработанная версия с исправлением всех критических багов.

ИСПРАВЛЕНИЯ по сравнению с текущей версией на GitHub:
  1. КРИТИЧЕСКИЙ БАГ: global collection в index_documents() — исправлен
  2. BM25 не нормализован — исправлен (MinMax нормализация)
  3. Markdown-сплиттер бесполезен для HTML-текста — заменён на HTML-aware чанкер
  4. Системный промпт в role=system (правильно для Groq API)
  5. asyncio.to_thread для блокирующих вызовов (embedding, ChromaDB)
  6. Обрезка длинных сообщений до лимита Telegram (4096 символов)
  7. Typing indicator — пользователь видит, что бот думает
  8. Модель llama-3.3-70b-versatile вместо llama-3.1-8b-instant
  9. Защита от пустой истории при формировании промпта
  10. /help команда
"""

import os
import logging
import glob
import asyncio
import time
import re
import requests
from functools import lru_cache
from bs4 import BeautifulSoup
from dotenv import load_dotenv

import telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

import groq
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

# ─────────────────────────────────────────────
#  Конфигурация
# ─────────────────────────────────────────────
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
HF_TOKEN       = os.getenv("HF_TOKEN")
ADMIN_ID       = int(os.getenv("ADMIN_ID", "0"))

for name, val in [("TELEGRAM_TOKEN", TELEGRAM_TOKEN),
                  ("GROQ_API_KEY",   GROQ_API_KEY),
                  ("HF_TOKEN",       HF_TOKEN)]:
    if not val:
        raise ValueError(f"{name} не задан в переменных окружения")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  API-клиенты
# ─────────────────────────────────────────────
groq_client = groq.Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────
#  ChromaDB
# ─────────────────────────────────────────────
CHROMA_PATH  = "./chroma_db"
DOCS_DIR     = "/app/data/docs"          # папка с HTML-файлами на хостинге
COLLECTION_NAME = "docs"

os.makedirs(DOCS_DIR, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# ─────────────────────────────────────────────
#  BM25 (глобальное состояние)
# ─────────────────────────────────────────────
corpus: list[str] = []      # все чанки в том же порядке, что в ChromaDB
bm25: BM25Okapi | None = None

# ─────────────────────────────────────────────
#  Настройки поиска
# ─────────────────────────────────────────────
TOP_K_VECTOR = 10   # сколько чанков возвращает ChromaDB
TOP_K_FINAL  = 5    # сколько оставляем после гибридного ранжирования
ALPHA        = 0.65  # вес векторного поиска (1-ALPHA — вес BM25)

# ─────────────────────────────────────────────
#  Fallback-сплиттер (plain-text)
# ─────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""],
)


# ─────────────────────────────────────────────
#  Парсинг HTML → структурированные чанки
# ─────────────────────────────────────────────
def extract_chunks_from_html(html_path: str) -> list[str]:
    """
    Разбивает HTML-файл на чанки с учётом его структуры.

    Алгоритм:
      1. Ищем заголовки h1/h2/h3 — они становятся «разделителями» секций.
      2. Каждая секция = заголовок + весь следующий за ним контент до следующего
         заголовка того же или высшего уровня.
      3. Если секция > 900 символов — дополнительно дробим RecursiveCharacterTextSplitter,
         сохраняя заголовок в начале каждого подчанка.
      4. Если заголовков нет — используем обычный сплиттер.

    Почему это важно: HTML-документы из BookStack содержат h1/h2/h3 теги,
    а не markdown-заголовки (#). MarkdownHeaderTextSplitter здесь бесполезен.
    """
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Удаляем мусор
    for tag in soup(["script", "style", "nav", "footer", "head"]):
        tag.decompose()

    # Ищем контентный блок
    content = soup.find("div", class_="page-content") or soup.find("body") or soup

    heading_tags = {"h1", "h2", "h3", "h4"}
    sections: list[tuple[str, str]] = []   # (heading_text, section_text)

    current_heading = ""
    current_parts: list[str] = []

    def flush():
        nonlocal current_heading, current_parts
        text = "\n".join(current_parts).strip()
        if text:
            sections.append((current_heading, text))
        current_parts = []

    for element in content.descendants:
        if not hasattr(element, "name") or element.name is None:
            continue
        if element.name in heading_tags:
            flush()
            current_heading = element.get_text(separator=" ").strip()
        elif element.name in ("p", "li", "td", "th", "blockquote"):
            text = element.get_text(separator=" ").strip()
            if text:
                current_parts.append(text)

    flush()

    if not sections:
        # Если не нашли структуры — plain-text fallback
        plain = content.get_text(separator="\n")
        plain = "\n".join(l.strip() for l in plain.splitlines() if l.strip())
        return text_splitter.split_text(plain) if plain else []

    chunks: list[str] = []
    for heading, body in sections:
        full_text = f"{heading}\n{body}" if heading else body
        if len(full_text) <= 900:
            chunks.append(full_text)
        else:
            # Дробим, но сохраняем заголовок контекста
            sub_chunks = text_splitter.split_text(full_text)
            for sc in sub_chunks:
                # Если заголовок уже есть в начале — не дублируем
                if heading and not sc.startswith(heading):
                    chunks.append(f"[{heading}]\n{sc}")
                else:
                    chunks.append(sc)

    return chunks


# ─────────────────────────────────────────────
#  Эмбеддинг (HuggingFace, с кэшем)
# ─────────────────────────────────────────────
HF_EMBED_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/"
    "pipeline/feature-extraction"
)

@lru_cache(maxsize=200)
def get_embedding(text: str) -> list[float] | None:
    """
    Получает эмбеддинг через HF Inference API.
    lru_cache(200) — кэширует 200 последних запросов,
    что полностью покрывает повторяющиеся вопросы пользователей.
    """
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"inputs": [text], "options": {"wait_for_model": True}}
    try:
        resp = requests.post(HF_EMBED_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        # HF возвращает [[...float...]] — вектор первого элемента
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):
                return result[0]
            return result
        return None
    except Exception:
        logger.exception("Ошибка эмбеддинга HF")
        return None


# ─────────────────────────────────────────────
#  BM25 обновление
# ─────────────────────────────────────────────
def update_bm25():
    """Пересобирает BM25-индекс на основе текущего corpus."""
    global bm25
    if not corpus:
        bm25 = None
        return
    tokenized = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    logger.info(f"BM25 индекс обновлён: {len(corpus)} чанков")


# ─────────────────────────────────────────────
#  Индексация документов
# ─────────────────────────────────────────────
def index_documents():
    """
    Полная переиндексация всех HTML-файлов из DOCS_DIR.

    ИСПРАВЛЕН КРИТИЧЕСКИЙ БАГ: в оригинале `collection` внутри функции
    была локальной переменной — global collection не был объявлен,
    поэтому после delete_collection бот падал с NameError или работал
    со старым объектом коллекции.
    """
    global collection, corpus  # ← ОБЯЗАТЕЛЬНО объявляем global

    files = glob.glob(os.path.join(DOCS_DIR, "*.html"))
    if not files:
        logger.warning(f"В {DOCS_DIR} нет HTML-файлов")
        return

    logger.info(f"Индексация: найдено {len(files)} файлов")

    all_chunks:    list[str]        = []
    all_embeddings: list[list[float]] = []
    all_ids:       list[str]        = []
    all_metadatas: list[dict]       = []

    for file_path in files:
        file_name = os.path.basename(file_path)
        chunks = extract_chunks_from_html(file_path)
        if not chunks:
            logger.warning(f"Нет чанков: {file_name}")
            continue

        embeddings = []
        for chunk in chunks:
            emb = get_embedding(chunk)
            if emb is None:
                logger.error(f"Не получен эмбеддинг для чанка из {file_name}, пропускаем файл")
                embeddings = None
                break
            embeddings.append(emb)

        if embeddings is None:
            continue

        ids       = [f"{file_name}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file_name} for _ in chunks]

        all_chunks.extend(chunks)
        all_embeddings.extend(embeddings)
        all_ids.extend(ids)
        all_metadatas.extend(metadatas)
        logger.info(f"  {file_name}: {len(chunks)} чанков")

    if not all_chunks:
        logger.error("Нет данных для индексации")
        return

    # Пересоздаём коллекцию
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # ← ИСПРАВЛЕНИЕ: обновляем глобальную переменную
    collection = chroma_client.create_collection(COLLECTION_NAME)

    collection.add(
        embeddings=all_embeddings,
        documents=all_chunks,
        metadatas=all_metadatas,
        ids=all_ids,
    )

    corpus = all_chunks
    update_bm25()
    logger.info(f"Индексация завершена. Всего чанков: {len(all_chunks)}")


# ─────────────────────────────────────────────
#  Запуск: загружаем или индексируем
# ─────────────────────────────────────────────
if collection.count() == 0:
    logger.info("Коллекция пуста — запускаем начальную индексацию...")
    index_documents()
else:
    logger.info(f"Коллекция содержит {collection.count()} записей, загружаем BM25...")
    data = collection.get(include=["documents"])
    corpus = data.get("documents", [])
    update_bm25()


# ─────────────────────────────────────────────
#  Нормализация BM25
# ─────────────────────────────────────────────
def normalize_scores(scores: list[float]) -> list[float]:
    """
    Min-Max нормализация BM25 оценок в диапазон [0, 1].

    ЗАЧЕМ: BM25 возвращает числа от 0 до ∞ (зависит от длины корпуса).
    Без нормализации вес BM25 в формуле 0.4 * bm25_score ломает
    гибридный поиск, так как его значения несопоставимы с similarity (0-1).
    """
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [0.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


# ─────────────────────────────────────────────
#  Гибридный поиск
# ─────────────────────────────────────────────
def hybrid_search(query: str) -> list[tuple[float, str, dict]]:
    """
    Гибридный поиск: ALPHA * cosine_similarity + (1-ALPHA) * BM25_normalized.

    Возвращает список (combined_score, chunk_text, metadata).
    """
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return []

    # Векторный поиск
    n_results = min(TOP_K_VECTOR, collection.count())
    if n_results == 0:
        return []

    vec = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    docs      = vec["documents"][0]
    distances = vec["distances"][0]
    metas     = vec["metadatas"][0]

    if not docs:
        return []

    # Cosine similarity: ChromaDB возвращает L2-расстояние по умолчанию.
    # Переводим в similarity: чем меньше distance, тем лучше.
    similarities = [max(0.0, 1 - d) for d in distances]

    # BM25 для найденных чанков
    bm25_raw: list[float] = []
    if bm25 and corpus:
        tokenized_query = query.lower().split()
        all_scores = bm25.get_scores(tokenized_query)
        # Строим словарь text → bm25_score (берём максимум при дублях)
        text_to_score: dict[str, float] = {}
        for i, doc in enumerate(corpus):
            if doc not in text_to_score or all_scores[i] > text_to_score[doc]:
                text_to_score[doc] = float(all_scores[i])
        bm25_raw = [text_to_score.get(doc, 0.0) for doc in docs]
    else:
        bm25_raw = [0.0] * len(docs)

    bm25_norm = normalize_scores(bm25_raw)

    # Комбинируем
    combined = []
    for i in range(len(docs)):
        score = ALPHA * similarities[i] + (1 - ALPHA) * bm25_norm[i]
        combined.append((score, docs[i], metas[i]))

    combined.sort(key=lambda x: x[0], reverse=True)
    return combined[:TOP_K_FINAL]


# ─────────────────────────────────────────────
#  Groq с retry
# ─────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
# Используем llama-3.3-70b-versatile:
#   - Доступна на Groq бесплатном тарифе
#   - Значительно умнее, чем llama-3.1-8b-instant
#   - Лучше понимает русский язык и структурированные данные
#   - Контекстное окно 128K токенов

SYSTEM_PROMPT = """Ты — корпоративный ассистент лизинговой компании «ЛК ПроДвижение».
Твоя задача: точно и полно отвечать на вопросы сотрудников, используя ТОЛЬКО предоставленный контекст из внутренней документации.

Правила ответа:
1. Отвечай ТОЛЬКО на основе контекста. Не додумывай и не используй внешние знания.
2. Если ответ есть в контексте — дай полный, структурированный ответ.
3. Если контекст содержит числовые данные (сроки, суммы, проценты) — выдели их **жирным**.
4. Используй списки и структуру, если информация это позволяет.
5. Если информации недостаточно — задай уточняющий вопрос. Начни ответ со слова "Уточните".
6. Если вопрос вне контекста документации — сообщи об этом честно.
7. Отвечай на русском языке."""


def call_groq_with_retry(messages: list[dict], retries: int = 3) -> str:
    """
    Вызов Groq API с экспоненциальной задержкой при сбоях.

    messages — список в формате OpenAI Chat: role + content.
    system prompt передаётся отдельным сообщением role=system,
    что даёт модели правильный контекст и улучшает качество ответа.
    """
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    for attempt in range(retries):
        try:
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=full_messages,
                temperature=0.1,   # минимальная температура = детерминированные ответы
                max_tokens=2048,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Groq ошибка, попытка {attempt + 1}/{retries}: {e}")
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


# ─────────────────────────────────────────────
#  Память диалога
# ─────────────────────────────────────────────
# chat_id → список словарей {"role": "user"/"assistant", "content": "..."}
user_histories: dict[int, list[dict]] = {}
MAX_HISTORY = 10  # максимум сообщений в истории (5 пар)


def get_history(chat_id: int) -> list[dict]:
    return user_histories.get(chat_id, [])


def add_to_history(chat_id: int, role: str, content: str):
    history = user_histories.get(chat_id, [])
    history.append({"role": role, "content": content})
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    user_histories[chat_id] = history


# ─────────────────────────────────────────────
#  Вспомогательные функции
# ─────────────────────────────────────────────
def truncate_message(text: str, max_len: int = 4000) -> str:
    """
    Telegram ограничивает сообщения 4096 символами.
    Обрезаем с пометкой, чтобы не получить ошибку API.
    """
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n\n…(ответ обрезан, уточните вопрос)"


# ─────────────────────────────────────────────
#  Обработчики Telegram
# ─────────────────────────────────────────────
async def start(update: Update, context):
    text = (
        "👋 Привет! Я корпоративный ассистент ЛК ПроДвижение.\n\n"
        "Задай вопрос по документации — об условиях сделок, процессах "
        "лизинга, согласовании договоров и другим темам.\n\n"
        "Доступные команды:\n"
        "/help — справка\n"
        "/clear — очистить историю диалога\n"
    )
    await update.message.reply_text(text)


async def help_command(update: Update, context):
    text = (
        "ℹ️ *Как пользоваться ботом*\n\n"
        "Просто напишите вопрос на русском языке. Бот найдёт ответ "
        "в корпоративной документации и ответит.\n\n"
        "Бот помнит историю диалога (последние 5 вопросов и ответов), "
        "поэтому можно задавать уточняющие вопросы.\n\n"
        "*Команды:*\n"
        "/clear — сбросить историю диалога\n"
        "/start — начало\n"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def clear_history(update: Update, context):
    chat_id = update.effective_chat.id
    user_histories.pop(chat_id, None)
    await update.message.reply_text("🗑 История диалога очищена.")


async def reindex(update: Update, context):
    """Принудительная переиндексация — только для администратора."""
    if ADMIN_ID == 0:
        await update.message.reply_text("ADMIN_ID не настроен.")
        return
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("⛔ У вас нет прав для этой команды.")
        return

    await update.message.reply_text("🔄 Начинаю переиндексацию...")
    try:
        # Запускаем блокирующую функцию в отдельном потоке
        await asyncio.to_thread(index_documents)
        await update.message.reply_text(
            f"✅ Переиндексация завершена.\n"
            f"Документов в базе: {collection.count()}"
        )
    except Exception as e:
        logger.exception("Ошибка переиндексации")
        await update.message.reply_text(f"❌ Ошибка: {e}")


async def handle_message(update: Update, context):
    """
    Основной обработчик сообщений.

    Порядок работы:
      1. Добавить вопрос пользователя в историю
      2. Получить эмбеддинг вопроса (в фоне)
      3. Гибридный поиск (ChromaDB + BM25)
      4. Собрать промпт с историей и контекстом
      5. Вызвать Groq
      6. Добавить ответ в историю
      7. Отправить ответ пользователю
    """
    chat_id   = update.effective_chat.id
    user_text = update.message.text.strip()

    if not user_text:
        return

    logger.info(f"[{update.effective_user.id}] Вопрос: {user_text[:80]}")

    # Показываем typing indicator
    await context.bot.send_chat_action(
        chat_id=chat_id,
        action=telegram.constants.ChatAction.TYPING
    )

    # --- Поиск (в потоке, не блокирует event loop) ---
    search_results = await asyncio.to_thread(hybrid_search, user_text)

    if not search_results:
        await update.message.reply_text(
            "😔 Не нашёл информации по вашему вопросу в документации.\n"
            "Попробуйте переформулировать или уточнить."
        )
        return

    # --- Формирование контекста ---
    context_chunks = [res[1] for res in search_results]
    sources = list(dict.fromkeys(res[2]["source"] for res in search_results))
    context_text = "\n\n---\n\n".join(context_chunks)

    # --- Формирование сообщений для Groq ---
    # Используем историю + новый вопрос с контекстом
    history = get_history(chat_id)

    # Текущий вопрос с контекстом — добавляем контекст только к последнему вопросу
    user_message = (
        f"Контекст из документации:\n{context_text}\n\n"
        f"Вопрос: {user_text}"
    )

    messages = history + [{"role": "user", "content": user_message}]

    # --- Вызов Groq ---
    try:
        answer = await asyncio.to_thread(call_groq_with_retry, messages)
    except Exception as e:
        logger.exception("Ошибка Groq")
        await update.message.reply_text(
            "⚠️ Произошла ошибка при генерации ответа. Попробуйте позже."
        )
        return

    # --- Обработка уточняющих вопросов ---
    # Если модель просит уточнения — НЕ добавляем в историю как «факт»
    is_clarification = answer.startswith("Уточните") or answer.startswith("Уточни")

    if not is_clarification:
        # Сохраняем в историю только завершённые пары вопрос-ответ
        # В истории храним оригинальный вопрос (без контекста), чтобы не раздувать промпт
        add_to_history(chat_id, "user", user_text)
        add_to_history(chat_id, "assistant", answer)

    # --- Источники ---
    source_line = f"\n\n📄 Источники: {', '.join(sources)}"
    full_answer = answer + source_line

    await update.message.reply_text(truncate_message(full_answer))


# ─────────────────────────────────────────────
#  Запуск бота
# ─────────────────────────────────────────────
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",   start))
    app.add_handler(CommandHandler("help",    help_command))
    app.add_handler(CommandHandler("clear",   clear_history))
    app.add_handler(CommandHandler("reindex", reindex))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот запущен (long polling)")
    app.run_polling(allowed_updates=telegram.Update.ALL_TYPES)


if __name__ == "__main__":
    main()
