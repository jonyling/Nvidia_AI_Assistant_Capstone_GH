import os
import time
import schedule
import threading
import pandas as pd
import joblib
from datetime import datetime
from dotenv import load_dotenv

import telebot
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# ======================== FLASK FOR RENDER ========================
from flask import Flask

load_dotenv()

# CONFIG
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID"))

bot = telebot.TeleBot(BOT_TOKEN)
llm = ChatOpenAI(model="gpt-5.4", temperature=0.3, max_tokens=1024)

DRIVE_DB_PATH = "./chroma_db_v2"
CSV_PATH = "nvda_2014_to_2026.csv"
MODEL_PATH = "nvidia_price_model.pkl"

# ======================== HEALTH SERVER ========================
health_app = Flask(__name__)

@health_app.route('/')
@health_app.route('/health')
def health():
    return "✅ NVIDIA Bot is running on Render", 200

def run_health_server():
    port = int(os.environ.get("PORT", 10000))
    print(f"🌐 Starting health server on port {port}")
    health_app.run(host="0.0.0.0", port=port, debug=False)

# ======================== LOAD DATA ========================
print("Loading RAG...")
vectorstore = None
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=DRIVE_DB_PATH, settings=Settings(allow_reset=True))
    vectorstore = Chroma(client=client, collection_name="nvidia_annual_reports_2014_2025", embedding_function=embeddings)
    print("✅ RAG Loaded")
except Exception as e:
    print(f"RAG warning: {e}")

df = pd.read_csv(CSV_PATH, skiprows=[1])
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

ddg_search = DuckDuckGoSearchRun()

# ======================== OpenClaw (LangGraph) ========================
class AgentState(TypedDict):
    query: str
    response: str
    debug_log: Annotated[str, operator.add]
    next_node: str

def router_node(state: AgentState) -> AgentState:
    q = state["query"].lower()
    if any(k in q for k in ["predict", "forecast", "price", "7 day", "trend"]):
        state["next_node"] = "trader"
    elif any(k in q for k in ["news", "outlook", "latest", "blackwell", "risk", "revenue"]):
        state["next_node"] = "researcher"
    else:
        state["next_node"] = "general"
    return state

def researcher_node(state: AgentState) -> AgentState:
    # ... (same as before)
    query = state["query"]
    context = ""
    if vectorstore:
        docs = vectorstore.similarity_search(query, k=4)
        context = "\n\n".join(d.page_content[:600] for d in docs)
    news = ddg_search.run(f"NVIDIA {query} latest")[:700]
    resp = llm.invoke([SystemMessage(content="NVIDIA expert researcher"), HumanMessage(content=f"Query: {query}\nRAG:\n{context}\nNews:\n{news}")]).content
    state["response"] = resp
    return state

def trader_node(state: AgentState) -> AgentState:
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
        pred = forecast.tail(7)[['ds', 'yhat']].copy()
        pred['ds'] = pred['ds'].dt.strftime('%Y-%m-%d')
        final = pred['yhat'].iloc[-1]
        change = (final / last_close - 1) * 100
        rec = "🟢 STRONG BUY" if change > 4 else "🟡 BUY" if change > 1.8 else "🔴 SELL" if change < -2.5 else "⚪ HOLD"
        lines = [f"{row['ds']}: **${row['yhat']:.2f}**" for _, row in pred.iterrows()]
        state["response"] = f"""**NVIDIA 7-DAY FORECAST**
Current: **${last_close:.2f}** → Day 7: **${final:.2f}** ({change:+.1f}%)
**Recommendation:** {rec}

**Predictions:**
""" + "\n".join(lines)
    except Exception as e:
        state["response"] = "Forecast temporarily unavailable."
    return state

def general_node(state: AgentState) -> AgentState:
    state["response"] = llm.invoke([SystemMessage(content="Helpful NVIDIA assistant"), HumanMessage(content=state["query"])]).content
    return state

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("trader", trader_node)
workflow.add_node("general", general_node)
workflow.set_entry_point("router")
workflow.add_conditional_edges("router", lambda x: x["next_node"], {"researcher":"researcher", "trader":"trader", "general":"general"})
for n in ["researcher","trader","general"]: workflow.add_edge(n, END)
app = workflow.compile()

# ======================== DAILY ALERT ========================
def send_daily_alert():
    try:
        f = app.invoke({"query": "Predict Nvidia share price trend for the next 7 days", "response": "", "debug_log": "", "next_node": ""})["response"]
        n = app.invoke({"query": "latest NVIDIA news", "response": "", "debug_log": "", "next_node": ""})["response"]
        alert = f"""**NVIDIA DAILY ALERT** {datetime.now().strftime('%Y-%m-%d %H:%M')}

{f}

**News:**
{n[:1600]}"""
        bot.send_message(CHAT_ID, alert[:3900])
        print("✅ Daily alert sent")
    except Exception as e:
        print(f"Alert error: {e}")

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)

# ======================== MAIN ========================
if __name__ == "__main__":
    print("🚀 Starting NVIDIA Bot on Render...")

    # Start health server FIRST
    threading.Thread(target=run_health_server, daemon=True).start()
    time.sleep(3)   # Give Flask time to bind port

    # Scheduler
    schedule.every(60).minutes.do(send_daily_alert)
    threading.Thread(target=run_scheduler, daemon=True).start()

    send_daily_alert()   # immediate

    print("🤖 Telegram polling started")
    bot.polling(none_stop=True, interval=1, timeout=30)