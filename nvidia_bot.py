import os
import time
import schedule
import telebot
from dotenv import load_dotenv
import joblib
import pandas as pd
from datetime import datetime

# LangChain
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ======================== CONFIG ========================
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    raise ValueError("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not found in .env")

bot = telebot.TeleBot(BOT_TOKEN)

# Render-specific settings
os.environ["PYTHONIOENCODING"] = "utf-8"
USE_WEBHOOK = os.getenv("USE_WEBHOOK", "False").lower() == "true"
WEBHOOK_URL = os.getenv("RENDER_WEBHOOK_URL")

llm = ChatOpenAI(model="gpt-5.4", temperature=0.3)

# ======================== LOAD COMPONENTS ========================
# RAG (Chroma)
vectorstore = None
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path="./chroma_db_v2", settings=Settings(allow_reset=True))
    vectorstore = Chroma(client=client, collection_name="nvidia_annual_reports_2014_2025", embedding_function=embeddings)
    print("✅ RAG Loaded Successfully")
except Exception as e:
    print(f"⚠️ RAG load failed (will work without): {e}")

ddg_search = DuckDuckGoSearchRun()

# Prophet Model
try:
    checkpoint = joblib.load("nvidia_price_model.pkl")
    prophet_model = checkpoint['prophet_model']
    last_close = checkpoint['last_close']
    print("✅ Prophet Model Loaded")
except:
    prophet_model = None
    last_close = 0
    print("⚠️ Prophet model not found")

df = pd.read_csv("nvda_2014_to_2026.csv", skiprows=[1])
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# ======================== HELPER FUNCTIONS ========================
def researcher_answer(query: str) -> str:
    context = ""
    if vectorstore:
        docs = vectorstore.similarity_search(query, k=4)
        context = "\n\n".join([doc.page_content[:500] for doc in docs])

    news = ddg_search.run(f"NVIDIA {query} latest news OR Blackwell OR Huawei")[:600]

    response = llm.invoke([
        SystemMessage(content="You are a senior NVIDIA strategy analyst. Give concise, intelligent answers."),
        HumanMessage(content=f"Question: {query}\n\nFrom Reports:\n{context}\n\nNews:\n{news}")
    ]).content

    return response[:3800] + "\n\n... (truncated)" if len(response) > 3800 else response


def trader_forecast() -> str:
    if not prophet_model:
        return "Forecast service unavailable."
    try:
        future = prophet_model.make_future_dataframe(periods=7, freq='B')
        future['MA7'] = df['Close'].rolling(7).mean().iloc[-1]
        future['MA30'] = df['Close'].rolling(30).mean().iloc[-1]
        future['Vol7'] = df['Close'].tail(7).std()
        future['Volume_MA7'] = df['Volume'].tail(7).mean()

        forecast = prophet_model.predict(future)
        pred = forecast.tail(7)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        pred['ds'] = pred['ds'].dt.strftime('%Y-%m-%d')

        final_pred = pred['yhat'].iloc[-1]
        upside = (final_pred / last_close - 1) * 100

        trade_rec = "🟢 STRONG BUY" if upside > 4 else "🟡 BUY" if upside > 1.8 else "🔴 SELL/CAUTION" if upside < -2.5 else "⚪ HOLD"

        lines = [f"{row['ds']}: **${row['yhat']:.2f}**" for _, row in pred.iterrows()]

        return f"""**🚀 NVIDIA 7-DAY FORECAST**
Current: **${last_close:.2f}** → Day 7: **${final_pred:.2f}** ({upside:+.1f}%)
**Recommendation:** {trade_rec}

Predictions:
""" + "\n".join(lines)
    except Exception as e:
        return f"Forecast error: {str(e)}"


# ======================== BOT HANDLERS ========================
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "✅ **NVIDIA Bot is online!**\n\nCommands:\n/forecast - 7-day price prediction\n/news - Latest news")

@bot.message_handler(commands=['forecast'])
def send_forecast(message):
    bot.reply_to(message, trader_forecast())

@bot.message_handler(commands=['news'])
def send_news(message):
    bot.reply_to(message, ddg_search.run("NVIDIA latest news")[:1000])

@bot.message_handler(func=lambda m: True)
def handle_message(message):
    bot.send_chat_action(message.chat.id, 'typing')
    query = message.text.strip()

    if any(k in query.lower() for k in ["forecast", "price", "predict", "7 day"]):
        reply = trader_forecast()
    else:
        reply = researcher_answer(query)

    bot.reply_to(message, reply)


# ======================== DAILY ALERT ========================
def send_daily_alert():
    forecast = trader_forecast()
    news = ddg_search.run("NVIDIA latest news OR Blackwell OR earnings")[:600]

    alert = f"🚨 **NVIDIA DAILY ALERT** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n{forecast}\n\n📰 News:\n{news}"
    try:
        bot.send_message(CHAT_ID, alert)
        print(f"✅ Daily alert sent at {datetime.now()}")
    except Exception as e:
        print(f"Alert error: {e}")


# ======================== START ========================
if __name__ == "__main__":
    print("🚀 Nvidia_bot starting on Render...")

    send_daily_alert()                     # Send one immediately
    schedule.every(60).minutes.do(send_daily_alert)

    if USE_WEBHOOK and WEBHOOK_URL:
        bot.remove_webhook()
        time.sleep(1)
        bot.set_webhook(url=WEBHOOK_URL)
        print(f"✅ Webhook set to: {WEBHOOK_URL}")
        # Keep process alive
        while True:
            schedule.run_pending()
            time.sleep(60)
    else:
        print("🔄 Using polling mode...")
        while True:
            try:
                schedule.run_pending()
                bot.polling(none_stop=True, interval=1, timeout=30)
            except Exception as e:
                print(f"Polling error: {e}")
                time.sleep(10)