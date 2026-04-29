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

# Flask for Render
from flask import Flask

load_dotenv()

# ======================== CONFIG ========================
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID"))

bot = telebot.TeleBot(BOT_TOKEN)
llm = ChatOpenAI(model="gpt-5.4", temperature=0.3, max_tokens=1200)

DRIVE_DB_PATH = "./chroma_db_v2"
CSV_PATH = "nvda_2014_to_2026.csv"
MODEL_PATH = "nvidia_price_model.pkl"

# ======================== FLASK HEALTH CHECK ========================
health_app = Flask(__name__)

@health_app.route('/')
@health_app.route('/health')
def health():
    return "✅ NVIDIA Bot is running", 200

def run_health_server():
    port = int(os.environ.get("PORT", 10000))
    print(f"🌐 Health server on port {port}")
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
    print("✅ RAG Loaded")
except Exception as e:
    print(f"RAG warning: {e}")

df = pd.read_csv(CSV_PATH, skiprows=[1])
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

ddg_search = DuckDuckGoSearchRun()

# ======================== OpenClaw STATE ========================
class AgentState(TypedDict):
    query: str
    response: str
    debug_log: Annotated[str, operator.add]
    next_node: str

# ======================== NODES ========================
def router_node(state: AgentState) -> AgentState:
    q = state["query"].lower()
    if any(x in q for x in ["predict", "forecast", "price", "share price", "stock price"]):
        if any(x in q for x in ["3 months", "quarter", "90 days", "12 months", "one year", "long term", "2026", "2027"]):
            state["next_node"] = "long_term_analyst"
        else:
            state["next_node"] = "trader"
    elif any(x in q for x in ["news", "outlook", "blackwell", "risk", "revenue", "financial", "huawei", "deepseek", "ascend"]):
        state["next_node"] = "researcher"
    else:
        state["next_node"] = "general"
    return state

def long_term_analyst_node(state: AgentState) -> AgentState:
    try:
        checkpoint = joblib.load(MODEL_PATH)
        model = checkpoint['prophet_model']
        last_close = checkpoint['last_close']

        # Short-term Prophet
        future = model.make_future_dataframe(periods=30, freq='B')
        future['MA7'] = df['Close'].rolling(7).mean().iloc[-1]
        future['MA30'] = df['Close'].rolling(30).mean().iloc[-1]
        future['Vol7'] = df['Close'].tail(7).std()
        future['Volume_MA7'] = df['Volume'].tail(7).mean()

        forecast = model.predict(future)
        pred = forecast.tail(7)[['ds', 'yhat']].copy()
        pred['ds'] = pred['ds'].dt.strftime('%Y-%m-%d')
        final_30d = forecast['yhat'].iloc[-1]
        upside_30d = (final_30d / last_close - 1) * 100

        # Qualitative RAG + News
        context = ""
        if vectorstore:
            docs = vectorstore.similarity_search(state["query"], k=5)
            context = "\n\n".join([d.page_content[:700] for d in docs])

        news = ddg_search.run("NVIDIA Blackwell Huawei Ascend DeepSeek impact OR price target 2026")[:900]

        llm_response = llm.invoke([
            SystemMessage(content="You are a senior NVIDIA equity analyst. Give balanced, insightful 3-12 month outlook."),
            HumanMessage(content=f"""Query: {state["query"]}
Short-term (30d) Prophet: {upside_30d:+.1f}% to ${final_30d:.2f}
RAG: {context}
News: {news}

Provide comprehensive analysis with:
- Key drivers & risks (Blackwell, Huawei/DeepSeek, valuation, geopolitics)
- Realistic price range & recommendation""")
        ]).content

        state["response"] = f"""**🚀 NVIDIA 3-MONTH FORECAST (Hybrid ML + LLM)**

**Current Price:** ${last_close:.2f}
**Short-term (7 days):**  
""" + "\n".join([f"{row['ds']}: **${row['yhat']:.2f}**" for _, row in pred.iterrows()]) + f"""

**30-day expected move:** **{upside_30d:+.1f}%** → ~${final_30d:.2f}

{llm_response}

*Backtested MAPE: {checkpoint.get('backtest_mape', 2.23):.2f}% • Longer horizons have higher uncertainty*"""
    except Exception as e:
        state["response"] = f"Long-term analysis error: {str(e)[:150]}"
    return state

def researcher_node(state: AgentState) -> AgentState:
    query = state["query"]
    context = ""
    if vectorstore:
        docs = vectorstore.similarity_search(query, k=4)
        context = "\n\n".join([d.page_content[:650] for d in docs])
    news = ddg_search.run(f"NVIDIA {query} latest news OR earnings")[:700]
    resp = llm.invoke([SystemMessage(content="NVIDIA expert researcher."), HumanMessage(content=f"Query: {query}\nRAG:\n{context}\nNews:\n{news}")]).content
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
        rec = "🟢 **STRONG BUY**" if change > 4 else "🟡 **BUY**" if change > 1.8 else "🔴 **SELL / CAUTION**" if change < -2.5 else "⚪ **HOLD**"
        lines = [f"{row['ds']}: **${row['yhat']:.2f}**" for _, row in pred.iterrows()]
        state["response"] = f"""**NVIDIA 7-DAY FORECAST**
Current: **${last_close:.2f}** → Day 7: **${final:.2f}** ({change:+.1f}%)
**Recommendation:** {rec}

**Predictions:**
""" + "\n".join(lines)
    except Exception as e:
        state["response"] = "Short-term forecast temporarily unavailable."
    return state

def general_node(state: AgentState) -> AgentState:
    state["response"] = llm.invoke([SystemMessage(content="Helpful NVIDIA assistant."), HumanMessage(content=state["query"])]).content
    return state

# ======================== BUILD GRAPH ========================
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("trader", trader_node)
workflow.add_node("long_term_analyst", long_term_analyst_node)
workflow.add_node("general", general_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", lambda x: x["next_node"],
    {"researcher": "researcher", "trader": "trader", "long_term_analyst": "long_term_analyst", "general": "general"})

for node in ["researcher", "trader", "long_term_analyst", "general"]:
    workflow.add_edge(node, END)

app = workflow.compile()

# ======================== DAILY ALERT ========================
def send_daily_alert():
    try:
        state = {"query": "Predict Nvidia share price trend for the next 7 days and provide latest news", "response": "", "debug_log": "", "next_node": ""}
        result = app.invoke(state)
        bot.send_message(CHAT_ID, result.get("response", "Daily update unavailable"))
        print("✅ Daily alert sent")
    except Exception as e:
        print(f"Daily alert error: {e}")

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)

# ======================== TELEGRAM HANDLER ========================
@bot.message_handler(func=lambda m: True)
def handle_message(message):
    user_text = message.text.strip()
    if not user_text:
        return

    thinking_msg = None
    try:
        thinking_msg = bot.reply_to(message, "🤔 Thinking... (10-25 seconds for deep analysis)")

        state = {"query": user_text, "response": "", "debug_log": "", "next_node": ""}
        result = app.invoke(state)
        reply = result.get("response", "Sorry, I couldn't generate an answer.")

        try:
            bot.delete_message(message.chat.id, thinking_msg.message_id)
        except:
            pass

        if len(reply) > 4000:
            reply = reply[:3990] + "\n\n... (truncated)"

        bot.reply_to(message, reply)

    except Exception as e:
        print(f"Handler error: {e}")
        if thinking_msg:
            try:
                bot.delete_message(message.chat.id, thinking_msg.message_id)
            except:
                pass
        bot.reply_to(message, "⚠️ Error processing request. Please try again.")

# ======================== START ========================
if __name__ == "__main__":
    print("🚀 Nvidia_bot starting on Render...")

    threading.Thread(target=run_health_server, daemon=True).start()
    time.sleep(4)  # Give Flask time to bind port

    schedule.every(60).minutes.do(send_daily_alert)
    threading.Thread(target=run_scheduler, daemon=True).start()

    send_daily_alert()

    print("🤖 Telegram polling started")
    bot.polling(none_stop=True, interval=1, timeout=30)