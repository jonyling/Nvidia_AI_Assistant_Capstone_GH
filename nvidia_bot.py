import os
import time
import schedule
import telebot
from dotenv import load_dotenv
import joblib
import pandas as pd
from datetime import datetime
from typing import TypedDict, Annotated
import operator

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

bot = telebot.TeleBot(BOT_TOKEN)
llm = ChatOpenAI(model="gpt-5.4", temperature=0.3, max_tokens=1000)

# ======================== LOAD COMPONENTS ========================
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

print("✅ All components loaded")

# ======================== AGENT STATE ========================
class AgentState(TypedDict):
    query: str
    response: str
    debug_log: Annotated[str, operator.add]
    next_node: str  # trader, researcher, hybrid, general

# ======================== NODES ========================
def trader_node(state: AgentState) -> AgentState:
    try:
        future = prophet_model.make_future_dataframe(periods=7, freq='B')
        future['MA7'] = df['Close'].rolling(7).mean().iloc[-1]
        future['MA30'] = df['Close'].rolling(30).mean().iloc[-1]
        future['Vol7'] = df['Close'].tail(7).std()
        future['Volume_MA7'] = df['Volume'].tail(7).mean()

        forecast = prophet_model.predict(future)
        pred = forecast.tail(7)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        pred['ds'] = pred['ds'].dt.strftime('%Y-%m-%d')

        final = pred['yhat'].iloc[-1]
        upside = (final / last_close - 1) * 100
        rec = "🟢 STRONG BUY" if upside > 4 else "🟡 BUY" if upside > 1.8 else "⚪ HOLD" if upside > -5 else "🔴 CAUTION"

        lines = [f"{row['ds']}: **${row['yhat']:.2f}** (range: ${row['yhat_lower']:.2f}–${row['yhat_upper']:.2f})" 
                for _, row in pred.iterrows()]

        state["response"] = f"""**🚀 NVIDIA 7-DAY FORECAST**
**Current Price:** ${last_close:.2f}
**Day 7 Expected:** **${final:.2f}** ({upside:+.1f}%)
**Recommendation:** {rec}

**Full Predictions:**
""" + "\n".join(lines)
    except Exception as e:
        state["response"] = f"Trader error: {str(e)}"
    state["debug_log"] += "📈 Trader completed\n"
    return state


def researcher_node(state: AgentState) -> AgentState:
    query = state["query"]
    context = ""
    if vectorstore:
        docs = vectorstore.similarity_search(query, k=5)
        context = "\n\n".join([d.page_content[:700] for d in docs])

    news = ddg_search.run(f"NVIDIA {query} latest OR Blackwell OR Huawei OR DeepSeek")[:800]

    resp = llm.invoke([
        SystemMessage(content="Senior NVIDIA strategy analyst. Be insightful."),
        HumanMessage(content=f"Query: {query}\nRAG:\n{context}\nNews:\n{news}")
    ]).content

    state["response"] = resp
    state["debug_log"] += "🔎 Researcher completed\n"
    return state


def hybrid_node(state: AgentState) -> AgentState:
    trader_result = trader_node(state.copy())
    researcher_result = researcher_node(state.copy())

    state["response"] = f"""{trader_result.get("response", "")}

**Qualitative Analysis & Market Context:**
{researcher_result.get("response", "")}"""
    state["debug_log"] += "🔀 Hybrid (Trader + Researcher) completed\n"
    return state


def general_node(state: AgentState) -> AgentState:
    resp = llm.invoke([SystemMessage(content="Helpful NVIDIA assistant."), HumanMessage(content=state["query"])]).content
    state["response"] = resp
    state["debug_log"] += "💬 General completed\n"
    return state


# ======================== INTELLIGENT ROUTER ========================
def router_node(query: str) -> str:
    prompt = f"""Analyze this query and decide the best execution strategy.
You can choose:
- trader: only quantitative price forecast
- researcher: only qualitative / news / RAG analysis
- hybrid: combine both Trader (ML forecast) + Researcher (qualitative context)
- general: simple answer

Query: {query}
Reply with ONLY one word: trader, researcher, hybrid or general."""

    try:
        decision = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
        if "hybrid" in decision:
            return "hybrid"
        elif "trader" in decision:
            return "trader"
        elif "researcher" in decision:
            return "researcher"
        else:
            return "general"
    except:
        return "hybrid"  # safe default


# ======================== BOT ========================
@bot.message_handler(commands=['start', 'help'])
def welcome(message):
    bot.reply_to(message, "✅ **NVIDIA Bot is online!**\nAsk anything — I can mix forecast + analysis.")

@bot.message_handler(func=lambda m: True)
def handle_message(message):
    bot.send_chat_action(message.chat.id, 'typing')
    query = message.text.strip()

    route = router_node(query)

    state = {"query": query, "response": "", "debug_log": "", "next_node": route}

    if route == "hybrid":
        final_state = hybrid_node(state)
    elif route == "trader":
        final_state = trader_node(state)
    elif route == "researcher":
        final_state = researcher_node(state)
    else:
        final_state = general_node(state)

    bot.reply_to(message, final_state["response"])


# ======================== HOURLY ALERT ========================
def hourly_alert():
    query = "latest Nvidia news and price outlook"
    state = {"query": query, "response": "", "debug_log": "", "next_node": "hybrid"}
    final = hybrid_node(state)
    try:
        bot.send_message(CHAT_ID, f"🕒 **HOURLY NVIDIA UPDATE** {datetime.now().strftime('%H:%M')}\n\n{final['response']}")
        print("✅ Hourly alert sent")
    except:
        pass

if __name__ == "__main__":
    print("🚀 Nvidia_bot with Full Multi-Agent System (Hybrid Router) started...")
    hourly_alert()
    schedule.every(60).minutes.do(hourly_alert)

    while True:
        try:
            schedule.run_pending()
            bot.polling(none_stop=True, interval=1, timeout=30)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)