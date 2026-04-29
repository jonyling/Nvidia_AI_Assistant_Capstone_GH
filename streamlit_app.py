import os
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHROMA_NO_SERVER"] = "true"
os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

load_dotenv()

st.set_page_config(page_title="🚀 NVIDIA AI Assistant", page_icon="📈", layout="wide")

DB_PATH = "./chroma_db_v2"
CSV_PATH = "nvda_2014_to_2026.csv"
MODEL_PATH = "nvidia_price_model.pkl"

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
import chromadb
from chromadb.config import Settings

llm = ChatOpenAI(model="gpt-5.4", temperature=0.3, max_tokens=1024)
ddg_search = DuckDuckGoSearchRun()

# ======================== LOAD COMPONENTS ========================
@st.cache_resource
def load_all_components():
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2", model_kwargs={"device": "cpu"})
    client = chromadb.PersistentClient(path=DB_PATH, settings=Settings(allow_reset=True, anonymized_telemetry=False))
    vectorstore = Chroma(client=client, collection_name="nvidia_annual_reports_2014_2025", embedding_function=embeddings)

    df = pd.read_csv(CSV_PATH, skiprows=[1])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    checkpoint = joblib.load(MODEL_PATH)

    return vectorstore, df, checkpoint

vectorstore, df, model_checkpoint = load_all_components()

# ======================== STATE ========================
class AgentState(TypedDict):
    query: str
    response: str
    debug_log: Annotated[str, operator.add]
    next_node: str


# ======================== INTELLIGENT ROUTER ========================
def router_node(state: AgentState) -> AgentState:
    query = state["query"]
    prompt = f"""Classify this query into exactly one category:
- trader: any question about future share price, forecast, prediction (any time horizon)
- researcher: news, financials, strategy, competitors, risks, Blackwell, Huawei, DeepSeek, business impact
- general: everything else

Query: {query}
Answer with only one word: trader, researcher or general."""

    decision = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
    state["next_node"] = "trader" if "trader" in decision else "researcher" if "researcher" in decision else "general"
    state["debug_log"] = f"🚦 LLM Router → {state['next_node']}\n"
    return state


# ======================== HYBRID TRADER NODE (Best Version) ========================
def trader_node(state: AgentState) -> AgentState:
    if not model_checkpoint:
        state["response"] = "❌ Model not available."
        return state

    query = state["query"].lower()
    model = model_checkpoint['prophet_model']
    last_close = model_checkpoint['last_close']

    # Detect time horizon
    days = 7
    if any(x in query for x in ["month", "3 months", "quarter", "90 days"]):
        days = 90
    elif any(x in query for x in ["6 months", "half year"]):
        days = 180
    elif any(x in query for x in ["year", "12 months"]):
        days = 365

    try:
        future = model.make_future_dataframe(periods=days, freq='B')
        # Add regressors
        future['MA7'] = df['Close'].rolling(7).mean().iloc[-1]
        future['MA30'] = df['Close'].rolling(30).mean().iloc[-1]
        future['Vol7'] = df['Close'].tail(7).std()
        future['Volume_MA7'] = df['Volume'].tail(7).mean()

        forecast = model.predict(future)
        pred = forecast.tail(days)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        pred['ds'] = pred['ds'].dt.strftime('%Y-%m-%d')

        final_pred = pred['yhat'].iloc[-1]
        upside = (final_pred / last_close - 1) * 100

        # LLM Contextual Adjustment (Hybrid Intelligence)
        context = f"Current price: ${last_close:.2f}. {days}-day Prophet forecast: ${final_pred:.2f} ({upside:+.1f}%)."
        llm_adjust = llm.invoke([
            SystemMessage(content="You are NVIDIA stock analyst. Combine ML forecast with latest market context."),
            HumanMessage(content=f"{context}\nQuery: {state['query']}\nGive a balanced view.")
        ]).content

        state["response"] = f"""**🚀 NVIDIA {days}-DAY FORECAST**

**Current Price:** ${last_close:.2f}  
**Expected Price ({pred['ds'].iloc[-1]}):** **${final_pred:.2f}** ({upside:+.1f}%)

**Trade Recommendation:** {'🟢 STRONG BUY' if upside > 8 else '🟡 BUY' if upside > 3 else '⚪ HOLD' if upside > -5 else '🔴 CAUTION'}

**LLM + ML Hybrid Analysis:**
{llm_adjust}

*Note: Longer-term forecasts have higher uncertainty.*"""

    except Exception as e:
        state["response"] = f"Forecast error: {str(e)}"

    state["debug_log"] += f"📈 Hybrid Trader ({days} days) completed\n"
    return state


# (researcher_node and general_node remain the same as previous version)
def researcher_node(state: AgentState) -> AgentState:
    query = state["query"]
    context = ""
    if vectorstore:
        docs = vectorstore.similarity_search(query, k=5)
        context = "\n\n".join([d.page_content[:800] for d in docs])

    news = ddg_search.run(f"NVIDIA {query} latest news OR earnings OR Blackwell OR Huawei OR DeepSeek")[:800]

    resp = llm.invoke([
        SystemMessage(content="You are NVIDIA expert researcher. Use RAG and news. Be concise and professional."),
        HumanMessage(content=f"Query: {query}\n\nRAG:\n{context}\n\nNews:\n{news}")
    ]).content

    state["response"] = resp
    state["debug_log"] += "🔎 Researcher completed\n"
    return state


def general_node(state: AgentState) -> AgentState:
    resp = llm.invoke([SystemMessage(content="Helpful NVIDIA assistant."), HumanMessage(content=state["query"])]).content
    state["response"] = resp
    state["debug_log"] += "💬 General completed\n"
    return state


# ======================== GRAPH ========================
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

# ======================== UI ========================
st.title("🚀 NVIDIA AI Assistant – Multi-Agent System")
st.caption("NTU DSAI Capstone | Intelligent Hybrid Forecasting (ML + LLM)")

# ... (rest of UI code same as previous version)
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything about NVIDIA..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = app.invoke({"query": prompt, "response": "", "debug_log": "", "next_node": ""})
            answer = result.get("response", "Sorry, I couldn't generate an answer.")
            log = result.get("debug_log", "")

            st.markdown(answer)
            with st.expander("🔍 Debug Trace"):
                st.code(log)

    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.success("✅ Hybrid ML + LLM Forecasting")
    st.success("✅ Full RAG + DDG Ready")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()