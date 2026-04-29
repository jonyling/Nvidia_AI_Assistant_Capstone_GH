import os
import time
import schedule
import telebot
import threading
import pandas as pd
import joblib
from datetime import datetime
from dotenv import load_dotenv

# LangChain & LangGraph
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

load_dotenv()

# ======================== CONFIG ========================
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID"))   # make sure it's an integer in .env

bot = telebot.TeleBot(BOT_TOKEN)

llm = ChatOpenAI(model="gpt-5.4", temperature=0.3, max_tokens=1024)

DRIVE_DB_PATH = "./chroma_db_v2"
CSV_PATH = "nvda_2014_to_2026.csv"
MODEL_PATH = "nvidia_price_model.pkl"

# ======================== LOAD COMPONENTS ========================
vectorstore = None
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=DRIVE_DB_PATH, settings=Settings(allow_reset=True))
    vectorstore = Chroma(client=client, collection_name="nvidia_annual_reports_2014_2025", embedding_function=embeddings)
    print("✅ RAG Loaded Successfully")
except Exception as e:
    print(f"⚠️ RAG load issue: {e}")

df = pd.read_csv(CSV_PATH, skiprows=[1])
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

ddg_search = DuckDuckGoSearchRun()

# ======================== LANGGRAPH STATE & NODES (identical to streamlit_app.py) ========================
class AgentState(TypedDict):
    query: str
    response: str
    debug_log: Annotated[str, operator.add]
    next_node: str


def router_node(state: AgentState) -> AgentState:
    q = state["query"].lower()
    if any(x in q for x in ["predict", "forecast", "price", "7 day", "next week", "tomorrow", "trend"]):
        state["next_node"] = "trader"
        state["debug_log"] = "🚦 Router → Trader\n"
    elif any(x in q for x in ["news", "outlook", "latest", "blackwell", "risk", "revenue", "financial", "huawei"]):
        state["next_node"] = "researcher"
        state["debug_log"] = "🚦 Router → Researcher\n"
    else:
        state["next_node"] = "general"
        state["debug_log"] = "🚦 Router → General\n"
    return state


def researcher_node(state: AgentState) -> AgentState:
    query = state["query"]
    context = ""
    if vectorstore:
        docs = vectorstore.similarity_search(query, k=5)
        context = "\n\n".join([d.page_content[:700] for d in docs])

    news = ddg_search.run(f"NVIDIA {query} latest news OR earnings")[:800]

    resp = llm.invoke([
        SystemMessage(content="You are NVIDIA expert researcher."),
        HumanMessage(content=f"Query: {query}\n\nRAG:\n{context}\n\nNews:\n{news}")
    ]).content

    state["response"] = resp
    state["debug_log"] += "🔎 Researcher completed\n"
    return state


def trader_node(state: AgentState) -> AgentState:
    debug = "📈 Trader (Prophet) activated → "
    try:
        checkpoint = joblib.load(MODEL_PATH)
        model = checkpoint['prophet_model']
        last_close = checkpoint['last_close']

        future = model.make_future_dataframe(periods=7, freq='B')
        future['MA7'] = df['Close'].rolling(7).mean().iloc[-1]
        future['MA30'] = df['Close'].rolling(30).mean().iloc[-1]
        future['Vol7'] = df['Close'].tail(7).std()
        future['Volume_MA7'] = df['Volume'].tail(7).mean()

        forecast = model.predict(future)
        pred = forecast.tail(7)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        pred['ds'] = pred['ds'].dt.strftime('%Y-%m-%d')

        final_pred = pred['yhat'].iloc[-1]
        upside = (final_pred / last_close - 1) * 100

        if upside > 4.0:
            trade_rec = "🟢 **STRONG BUY** – Trade window: next 24-48 hours"
        elif upside > 1.8:
            trade_rec = "🟡 **BUY** – Good entry in next 3 days"
        elif upside < -2.5:
            trade_rec = "🔴 **SELL / CAUTION**"
        else:
            trade_rec = "⚪ **HOLD** – Monitor closely"

        lines = [f"{row['ds']}: **${row['yhat']:.2f}** (range ${row['yhat_lower']:.2f}–${row['yhat_upper']:.2f})"
                 for _, row in pred.iterrows()]

        state["response"] = f"""**🚀 NVIDIA 7-DAY FORECAST**

**Current Price:** ${last_close:.2f} → **Day 7 Expected:** **${final_pred:.2f}** ({upside:+.1f}%)

**Predictions:**
""" + "\n".join(lines) + f"""

**Trade Recommendation:**
{trade_rec}

*Backtested MAPE: {checkpoint.get('backtest_mape', 2.23):.2f}%*"""
    except Exception as e:
        state["response"] = f"Forecast error: {str(e)}"

    state["debug_log"] += debug + "\n"
    return state


def general_node(state: AgentState) -> AgentState:
    resp = llm.invoke([
        SystemMessage(content="You are a helpful NVIDIA assistant."),
        HumanMessage(content=state["query"])
    ]).content
    state["response"] = resp
    state["debug_log"] += "💬 General Agent\n"
    return state


# ======================== BUILD GRAPH ========================
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("trader", trader_node)
workflow.add_node("general", general_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    lambda x: x["next_node"],
    {"researcher": "researcher", "trader": "trader", "general": "general"}
)
for node in ["researcher", "trader", "general"]:
    workflow.add_edge(node, END)

app = workflow.compile()

# ======================== TELEGRAM BOT ========================
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message,
        "✅ **Nvidia_bot is online!** (NTU DSAI Capstone Multi-Agent System)\n\n"
        "Just ask anything about NVIDIA.\n"
        "/forecast — 7-day price prediction + trade signal\n"
        "/news — latest news\n\n"
        "I also send daily alerts automatically!")

@bot.message_handler(commands=['forecast'])
def send_forecast(message):
    bot.send_chat_action(message.chat.id, 'typing')
    state = {"query": "Predict Nvidia share price trend for the next 7 days", "response": "", "debug_log": "", "next_node": ""}
    result = app.invoke(state)
    bot.reply_to(message, result.get("response", "Forecast unavailable"))

@bot.message_handler(commands=['news'])
def send_news(message):
    bot.send_chat_action(message.chat.id, 'typing')
    state = {"query": "latest NVIDIA news and market outlook", "response": "", "debug_log": "", "next_node": ""}
    result = app.invoke(state)
    bot.reply_to(message, result.get("response", "News unavailable"))

@bot.message_handler(func=lambda m: True)
def handle_message(message):
    bot.send_chat_action(message.chat.id, 'typing')
    query = message.text.strip()
    state = {"query": query, "response": "", "debug_log": "", "next_node": ""}
    result = app.invoke(state)
    reply = result.get("response", "Sorry, I couldn't generate an answer.")
    # Telegram has ~4096 char limit
    if len(reply) > 4000:
        reply = reply[:3990] + "\n\n... (truncated)"
    bot.reply_to(message, reply)


# ======================== DAILY ALERT ========================
def send_daily_alert():
    query = "Predict Nvidia share price trend for the next 7 days and provide latest news + trading opportunity"
    state = {"query": query, "response": "", "debug_log": "", "next_node": ""}
    result = app.invoke(state)
    alert = f"""🚨 **NVIDIA DAILY ALERT** {datetime.now().strftime('%Y-%m-%d %H:%M')}
{result.get('response', 'No response')}

🔍 Debug: {result.get('debug_log', '')}"""
    try:
        bot.send_message(CHAT_ID, alert)
        print(f"✅ Daily alert sent at {datetime.now()}")
    except Exception as e:
        print(f"Alert error: {e}")


# ... (keep all your existing code up to the scheduler part)

# ======================== HEALTH CHECK FOR RENDER (Critical) ========================
from flask import Flask
import threading

# Simple Flask app for Render health checks
health_app = Flask(__name__)

@health_app.route('/')
def health():
    return "NVIDIA Bot is running! ✅", 200

@health_app.route('/health')
def health_check():
    return "OK", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    health_app.run(host="0.0.0.0", port=port)

# ======================== SCHEDULER + BOT START ========================
if __name__ == "__main__":
    print("🚀 Nvidia_bot (Multi-Agent System) is starting on Render...")

    # Start Flask health check in background (required by Render Web Service)
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Schedule daily alerts
    schedule.every(60).minutes.do(send_daily_alert)   # Change to .day.at("09:00") if you prefer once per day
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()

    # Send immediate alert
    send_daily_alert()

    print("🤖 Starting Telegram polling...")
    bot.polling(none_stop=True, interval=1, timeout=30)