import os
import time
import schedule
import telebot
from dotenv import load_dotenv
import joblib
import pandas as pd
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    print("❌ Missing Telegram credentials!")
    exit(1)

bot = telebot.TeleBot(BOT_TOKEN)
llm = ChatOpenAI(model="gpt-5.4", temperature=0.3)

print("🚀 Starting Nvidia_bot...")

# Load components
vectorstore = None
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2", model_kwargs={"device": "cpu"})
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path="./chroma_db_v2", settings=Settings(allow_reset=True))
    vectorstore = Chroma(client=client, collection_name="nvidia_annual_reports_2014_2025", embedding_function=embeddings)
    print("✅ RAG Loaded")
except Exception as e:
    print(f"⚠️ RAG failed: {e}")

ddg_search = DuckDuckGoSearchRun()

checkpoint = joblib.load("nvidia_price_model.pkl")
prophet_model = checkpoint['prophet_model']
last_close = checkpoint['last_close']

df = pd.read_csv("nvda_2014_to_2026.csv", skiprows=[1])
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print("✅ All components loaded successfully")

# ======================== HANDLERS ========================
@bot.message_handler(commands=['start', 'help'])
def welcome(message):
    bot.reply_to(message, "✅ **NVIDIA Bot is online!**\nAsk me anything about price, technology, competition, or strategy.")

@bot.message_handler(func=lambda m: True)
def handle_message(message):
    bot.send_chat_action(message.chat.id, 'typing')
    query = message.text.strip()

    # Smart routing
    if any(word in query.lower() for word in ["predict", "forecast", "price", "3 months", "stock"]):
        reply = trader_forecast()
    else:
        reply = researcher_answer(query)

    bot.reply_to(message, reply)


def trader_forecast():
    try:
        future = prophet_model.make_future_dataframe(periods=7, freq='B')
        future['MA7'] = df['Close'].rolling(7).mean().iloc[-1]
        future['MA30'] = df['Close'].rolling(30).mean().iloc[-1]
        future['Vol7'] = df['Close'].tail(7).std()
        future['Volume_MA7'] = df['Volume'].tail(7).mean()

        forecast = prophet_model.predict(future)
        pred = forecast.tail(7)[['ds', 'yhat']].copy()
        pred['ds'] = pred['ds'].dt.strftime('%Y-%m-%d')

        final = pred['yhat'].iloc[-1]
        upside = (final / last_close - 1) * 100
        rec = "🟢 STRONG BUY" if upside > 4 else "🟡 BUY" if upside > 1.8 else "⚪ HOLD" if upside > -5 else "🔴 CAUTION"

        return f"""**🚀 NVIDIA 7-DAY FORECAST**
Current: **${last_close:.2f}** → Day 7: **${final:.2f}** ({upside:+.1f}%)
**Recommendation:** {rec}"""
    except Exception as e:
        return f"Forecast error: {str(e)}"


def researcher_answer(query: str):
    context = ""
    if vectorstore:
        docs = vectorstore.similarity_search(query, k=5)
        context = "\n\n".join([d.page_content[:700] for d in docs])

    news = ddg_search.run(f"NVIDIA {query} latest OR Blackwell OR Huawei OR DeepSeek OR World Model")[:800]

    resp = llm.invoke([
        SystemMessage(content="You are a senior NVIDIA analyst. Be insightful and balanced."),
        HumanMessage(content=f"Query: {query}\nRAG:\n{context}\nNews:\n{news}")
    ]).content
    return resp[:3800]


# ======================== DAILY ALERT ========================
def send_daily_alert():
    forecast = trader_forecast()
    bot.send_message(CHAT_ID, f"🚨 **NVIDIA DAILY ALERT** {datetime.now().strftime('%Y-%m-%d')}\n\n{forecast}")

if __name__ == "__main__":
    print("🚀 Nvidia_bot is running...")
    send_daily_alert()
    schedule.every(60).minutes.do(send_daily_alert)

    while True:
        try:
            schedule.run_pending()
            bot.polling(none_stop=True, interval=1, timeout=30)
        except Exception as e:
            print(f"Polling error: {e}")
            time.sleep(10)