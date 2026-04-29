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

# Flask for Render health check
from flask import Flask

load_dotenv()

# ======================== CONFIG ========================
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID"))

bot = telebot.TeleBot(BOT_TOKEN)
llm = ChatOpenAI(model="gpt-5.4", temperature=0.3, max_tokens=1024)

DRIVE_DB_PATH = "./chroma_db_v2"
CSV_PATH = "nvda_2014_to_2026.csv"
MODEL_PATH = "nvidia_price_model.pkl"

# ======================== FLASK HEALTH CHECK ========================
health_app = Flask(__name__)

@health_app.route('/')
@health_app.route('/health')
def health():
    return "✅ NVIDIA Bot is running on Render", 200

def run_health_server():
    port = int(os.environ.get("PORT", 10000))
    print(f"🌐 Health server started on port {port}")
    health_app.run(host="0.0.0.0", port=port, debug=False)

# ======================== LOAD COMPONENTS ========================
print("Loading RAG...")
vectorstore = None
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=DRIVE_DB_PATH, settings=Settings(allow_reset=True))
    vectorstore = Chroma(client=client, collection_name="nvidia_annual_reports_2014_2025", embedding_function=embeddings)
    print("✅ RAG Loaded Successfully")
except Exception as e:
    print(f"⚠️ RAG issue: {e}")

df = pd.read_csv(CSV_PATH, skiprows=[1])
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

ddg_search = DuckDuckGoSearchRun()

# ======================== OpenClaw LangGraph ========================
class AgentState(TypedDict):
    query: str
    response: str
    debug_log: Annotated[str, operator.add]
    next_node: str

def router_node(state: AgentState) -> AgentState:
    q = state["query"].lower()
    if any(k in q for k in ["predict", "forecast", "price", "7 day", "next week", "tomorrow", "trend"]):
        state["next_node"] = "trader"
    elif any(k in q for k in ["news", "outlook", "latest", "blackwell", "risk", "revenue", "financial", "huawei", "ascend"]):
        state["next_node"] = "researcher"
    else:
        state["next_node"] = "general"
    return state

def researcher_node(state: AgentState) -> AgentState:
    query = state["query"]
    context = ""
    if vectorstore:
        docs = vectorstore.similarity_search(query, k=4)   # Reduced for speed
        context = "\n\n".join([d.page_content[:650] for d in docs])
    news = ddg_search.run(f"NVIDIA {query} latest news OR earnings")[:700]
    resp = llm.invoke([
        SystemMessage(content="You are a precise NVIDIA expert researcher."),
        HumanMessage(content=f"Query: {query}\n\nRAG Context:\n{context}\n\nLatest News:\n{news}")
    ]).content
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
        final_pred = pred['yhat'].iloc[-1]
        upside = (final_pred / last_close - 1) * 100

        rec = "🟢 **STRONG BUY**" if upside > 4 else "🟡 **BUY**" if upside > 1.8 else "🔴 **SELL / CAUTION**" if upside < -2.5 else "⚪ **HOLD**"

        lines = [f"{row['ds']}: **${row['yhat']:.2f}**" for _, row in pred.iterrows()]

        state["response"] = f"""**NVIDIA 7-DAY FORECAST**
Current: **${last_close:.2f}** → Day 7: **${final_pred:.2f}** ({upside:+.1f}%)
**Recommendation:** {rec}

**Predictions:**
""" + "\n".join(lines)
    except Exception as e:
        state["response"] = f"Forecast error: {str(e)[:100]}"
    return state

def general_node(state: AgentState) -> AgentState:
    state["response"] = llm.invoke([
        SystemMessage(content="You are a helpful NVIDIA assistant."),
        HumanMessage(content=state["query"])
    ]).content
    return state

# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("trader", trader_node)
workflow.add_node("general", general_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", lambda x: x["next_node"],
    {"researcher": "researcher", "trader": "trader", "general": "general"})
for node in ["researcher", "trader", "general"]:
    workflow.add_edge(node, END)

app = workflow.compile()

# ======================== DAILY ALERT ========================
def send_daily_alert():
    try:
        f_state = {"query": "Predict Nvidia share price trend for the next 7 days", "response": "", "debug_log": "", "next_node": ""}
        forecast = app.invoke(f_state)["response"]

        n_state = {"query": "latest NVIDIA news and market outlook", "response": "", "debug_log": "", "next_node": ""}
        news = app.invoke(n_state)["response"]

        alert = f"""**NVIDIA DAILY ALERT** {datetime.now().strftime('%Y-%m-%d %H:%M')}

{forecast}

**News:**
{news[:1650]}"""
        bot.send_message(CHAT_ID, alert[:3900])
        print(f"✅ Daily alert sent at {datetime.now()}")
    except Exception as e:
        print(f"Daily alert error: {e}")

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)

# ======================== MESSAGE HANDLER (Fixed) ========================
@bot.message_handler(func=lambda m: True)
def handle_message(message):
    user_text = message.text.strip()
    if not user_text:
        return

    try:
        # Show "Thinking..." immediately
        thinking_msg = bot.reply_to(message, "🤔 Thinking... (10-20 seconds)")

        state = {"query": user_text, "response": "", "debug_log": "", "next_node": ""}
        result = app.invoke(state)
        reply = result.get("response", "Sorry, I couldn't generate an answer.")

        # Delete thinking message
        try:
            bot.delete_message(message.chat.id, thinking_msg.message_id)
        except:
            pass

        if len(reply) > 4000:
            reply = reply[:3990] + "\n\n... (truncated)"

        bot.reply_to(message, reply)

    except Exception as e:
        print(f"Handler error for '{user_text}': {e}")
        try:
            bot.delete_message(message.chat.id, thinking_msg.message_id)
        except:
            pass
        bot.reply_to(message, "⚠️ Sorry, an error occurred. Please try again.")

# ======================== START ========================
if __name__ == "__main__":
    print("🚀 Nvidia_bot (Multi-Agent System) starting on Render...")

    # Start health server first
    threading.Thread(target=run_health_server, daemon=True).start()
    time.sleep(4)   # Give Flask time to bind port

    # Scheduler
    schedule.every(60).minutes.do(send_daily_alert)
    threading.Thread(target=run_scheduler, daemon=True).start()

    send_daily_alert()   # Immediate alert

    print("🤖 Telegram polling started")
    bot.polling(none_stop=True, interval=1, timeout=30)