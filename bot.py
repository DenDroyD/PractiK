#!/usr/bin/env python3
import os, sys, logging, asyncio, time
from pathlib import Path
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from bs4 import BeautifulSoup
import groq

# ===== НАСТРОЙКИ =====
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
USE_RAG = os.getenv("USE_RAG", "false").lower() == "true"  # ← ПО УМОЛЧАНИЮ FALSE!
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
DOCS_DIR = Path(DATA_DIR) / "docs"

if not TELEGRAM_TOKEN or not GROQ_API_KEY:
    print("❌ Заполните TELEGRAM_TOKEN и GROQ_API_KEY", file=sys.stderr)
    sys.exit(1)

# ===== ЛОГИ =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# ===== GROQ =====
try:
    groq_client = groq.Groq(api_key=GROQ_API_KEY)
    logger.info("✅ Groq подключён")
except:
    groq_client = None

# ===== RAG (опционально, с защитой) =====
embedder = None
docs_cache = []

if USE_RAG:
    logger.info("🔄 Попытка включить RAG...")
    try:
        from sentence_transformers import SentenceTransformer
        cache_dir = Path(DATA_DIR) / "models_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("⏳ Загрузка модели (может занять 1-2 мин при первом запуске)...")
        embedder = SentenceTransformer('all-MiniLM-L3-v2', cache_folder=str(cache_dir), device='cpu')
        
        # Загрузка документов
        for f in DOCS_DIR.glob("*.html"):
            try:
                text = BeautifulSoup(f.read_text(encoding='utf-8'), 'html.parser').get_text(separator=' ', strip=True)
                for i in range(0, len(text), 800):
                    docs_cache.append({"text": text[i:i+800].strip(), "source": f.name})
                logger.info(f"✅ Загружен {f.name}")
            except Exception as e:
                logger.warning(f"⚠️ Не загружен {f.name}: {e}")
        
        logger.info(f"📚 Документов в кэше: {len(docs_cache)}")
        
    except Exception as e:
        logger.error(f"❌ RAG не активирован: {e}")
        logger.info("💡 Бот будет работать без поиска по документам")
        USE_RAG = False

# ===== ОБРАБОТЧИКИ =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"🤖 Бот работает!\n"
        f"🧠 RAG: {'✅' if USE_RAG and embedder else '❌'}\n"
        f"📚 Документов: {len(docs_cache)}\n"
        f"Задайте вопрос."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    logger.info(f"👤 Вопрос: {text[:100]}")
    
    # 🔹 Поиск по документам (если RAG включён)
    context_add = ""
    if USE_RAG and embedder and docs_cache:
        try:
            q_emb = embedder.encode([text])[0]
            scores = [(i, sum(q_emb * embedder.encode([d["text"]])[0])) for i, d in enumerate(docs_cache)]
            top = sorted(scores, key=lambda x: x[1], reverse=True)[:2]
            if top and top[0][1] > 0.3:
                context_add = "\n\nКонтекст:\n" + "\n".join([docs_cache[i]["text"] for i, _ in top])
        except: pass
    
    # 🔹 Запрос к Groq
    if not groq_client:
        await update.message.reply_text("❌ Groq не доступен")
        return
    
    try:
        resp = await asyncio.to_thread(
            lambda: groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": context_add + "\n\n" + text if context_add else text}],
                max_tokens=512, timeout=30
            )
        )
        await update.message.reply_text(resp.choices[0].message.content)
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        await update.message.reply_text("⚠️ Ошибка ответа")

# ===== ЗАПУСК =====
def main():
    logger.info("🚀 Старт...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()