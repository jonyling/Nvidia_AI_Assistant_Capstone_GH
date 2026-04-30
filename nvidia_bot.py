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

bot = telebot.TeleBot(BOT_TOKEN)
llm = ChatOpenAI(model="gpt-5.4", temperature=0.3)

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
    print(f"⚠️ RAG: {e}")

ddg_search = DuckDuckGoSearchRun()
checkpoint = joblib.load("nvidia_price_model.pkl")
prophet_model = checkpoint['prophet_model']
last_close = checkpoint['last_close']
df = pd.read_csv("nvda_2014_to_2026.csv", skiprows=[1])
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print("✅ Nvidia_bot fully loaded")

# ======================== ROUTER ========================
def get_router_decision(query: str):
    prompt = f"""Classify this query:
- trader: pure short-term price forecast
- researcher: qualitative / news / technology / competition
- hybrid: price prediction + qualitative analysis (e.g. impact of Huawei/DeepSeek on price)

Query: {query}
Reply with ONLY one word: trader, researcher or hybrid"""
    try:
        decision = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
        if "hybrid" in decision:
            return "hybrid"
        return "trader" if "trader" in decision else "researcher"
    except:
        return "researcher"

# ======================== NODES ========================
def trader_forecast() -> str:
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

        rec = "🟢 STRONG BUY" if upside > 4 else "🟡 BUY" if upside > 1.8 else "⚪ HOLD" if upside > -5 else "🔴 CAUTION"

        lines = [f"{row['ds']}: **${row['yhat']:.2f}** (range: ${row['yhat_lower']:.2f}–${row['yhat_upper']:.2f})" 
                for _, row in pred.iterrows()]

        return f"""**🚀 NVIDIA 7-DAY FORECAST**
**Current Price:** ${last_close:.2f}
**Day 7 Expected:** **${final_pred:.2f}** ({upside:+.1f}%)
**Recommendation:** {rec}

**Full Predictions:**
""" + "\n".join(lines)
    except Exception as e:
        return f"Forecast error: {str(e)}"


def researcher_answer(query: str) -> str:
    context = ""
    if vectorstore:
        docs = vectorstore.similarity_search(query, k=5)
        context = "\n\n".join([d.page_content[:700] for d in docs])

    news = ddg_search.run(f"NVIDIA {query} latest news OR Blackwell OR Huawei OR DeepSeek OR World Model")[:800]

    resp = llm.invoke([
        SystemMessage(content="You are a senior NVIDIA strategy analyst. Be insightful and balanced."),
        HumanMessage(content=f"Query: {query}\n\nRAG Context:\n{context}\n\nLatest News:\n{news}")
    ]).content
    return resp


def hybrid_answer(query: str) -> str:
    forecast = trader_forecast()
    analysis = researcher_answer(query)
    return f"{forecast}\n\n**Qualitative Analysis:**\n{analysis}"


# ======================== BOT ========================
@bot.message_handler(commands=['start', 'help'])
def welcome(message):
    bot.reply_to(message, "✅ **NVIDIA Bot is online!**\nAsk anything about price forecasts or technology/competition impact.")

@bot.message_handler(func=lambda m: True)
def handle_message(message):
    bot.send_chat_action(message.chat.id, 'typing')
    query = message.text.strip()

    route = get_router_decision(query)

    if route == "hybrid":
        reply = hybrid_answer(query)
    elif route == "trader":
        reply = trader_forecast()
    else:
        reply = researcher_answer(query)

    bot.reply_to(message, reply)


# ======================== DAILY ALERT ========================
def send_daily_alert():
    forecast = trader_forecast()
    bot.send_message(CHAT_ID, f"🚨 **NVIDIA DAILY ALERT** {datetime.now().strftime('%Y-%m-%d')}\n\n{forecast}")

if __name__ == "__main__":
    print("🚀 Nvidia_bot with Hybrid Router started...")
    send_daily_alert()
    schedule.every(60).minutes.do(send_daily_alert)

    while True:
        try:
            schedule.run_pending()
            bot.polling(none_stop=True, interval=1, timeout=30)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)