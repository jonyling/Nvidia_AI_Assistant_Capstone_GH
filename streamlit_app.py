import os
import streamlit as st
import pandas as pd
import joblib
from datetime import timedelta
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

llm = ChatOpenAI(model="gpt-5.4", temperature=0.4, max_tokens=1200)
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

class AgentState(TypedDict):
    query: str
    response: str
    debug_log: Annotated[str, operator.add]
    next_node: str

# ======================== INTELLIGENT ROUTER ========================
def router_node(state: AgentState) -> AgentState:
    query = state["query"]
    prompt = f"""Classify into ONE category only:
- trader: any price prediction / forecast (any time horizon)
- researcher: news, financials, strategy, competitors, risks, impact analysis
- general: everything else

Query: {query}
Answer with only one word:"""

    decision = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
    state["next_node"] = "trader" if "trader" in decision else "researcher" if "researcher" in decision else "general"
    state["debug_log"] = f"🚦 Router → {state['next_node']}\n"
    return state

# ======================== IMPROVED HYBRID TRADER ========================
def trader_node(state: AgentState) -> AgentState:
    if not model_checkpoint:
        state["response"] = "❌ Model unavailable."
        return state

    query = state["query"].lower()
    model = model_checkpoint['prophet_model']
    last_close = model_checkpoint['last_close']

    # Detect horizon
    if any(x in query for x in ["3 months", "90 days", "quarter", "month"]):
        horizon_days = 90
        title = "3-MONTH FORECAST"
    elif any(x in query for x in ["6 months", "half year"]):
        horizon_days = 180
        title = "6-MONTH FORECAST"
    else:
        horizon_days = 7
        title = "7-DAY FORECAST"

    try:
        future = model.make_future_dataframe(periods=horizon_days, freq='B')
        future['MA7'] = df['Close'].rolling(7).mean().iloc[-1]
        future['MA30'] = df['Close'].rolling(30).mean().iloc[-1]
        future['Vol7'] = df['Close'].tail(7).std()
        future['Volume_MA7'] = df['Volume'].tail(7).mean()

        forecast = model.predict(future)
        pred = forecast.tail(horizon_days)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        pred['ds'] = pred['ds'].dt.strftime('%Y-%m-%d')

        final_pred = pred['yhat'].iloc[-1]
        upside = (final_pred / last_close - 1) * 100

        # Hybrid LLM Analysis
        hybrid_analysis = llm.invoke([
            SystemMessage(content="You are a professional NVIDIA stock analyst. Combine quantitative Prophet forecast with qualitative market context."),
            HumanMessage(content=f"""Current price: ${last_close:.2f}
Prophet {horizon_days}-day forecast: ${final_pred:.2f} ({upside:+.1f}%)
Query: {state['query']}

Provide balanced, insightful analysis.""")
        ]).content

        state["response"] = f"""**🚀 NVIDIA {title}**

**Current Price:** ${last_close:.2f}  
**Expected Price:** **${final_pred:.2f}** ({upside:+.1f}%)

**Hybrid Recommendation:** {'🟢 STRONG BUY' if upside > 12 else '🟡 BUY' if upside > 5 else '⚪ HOLD' if upside > -8 else '🔴 CAUTION'}

**Detailed Analysis:**
{hybrid_analysis}

*Prophet Backtested MAPE: {model_checkpoint.get('backtest_mape', 'N/A'):.2f}% • Longer horizons have higher uncertainty*"""

    except Exception as e:
        state["response"] = f"Forecast error: {str(e)}"

    state["debug_log"] += f"📈 Hybrid Trader ({horizon_days} days) completed\n"
    return state

# Researcher and General nodes (unchanged but reliable)
def researcher_node(state: AgentState) -> AgentState:
    query = state["query"]
    context = "\n\n".join([d.page_content[:800] for d in vectorstore.similarity_search(query, k=5)]) if vectorstore else ""

    news = ddg_search.run(f"NVIDIA {query} latest news OR earnings OR Blackwell OR Huawei OR DeepSeek")[:800]

    resp = llm.invoke([
        SystemMessage(content="NVIDIA expert researcher. Use RAG + news. Be concise and professional."),
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
workflow.add_conditional_edges("router", lambda x: x["next_node"], {"researcher": "researcher", "trader": "trader", "general": "general"})
for node in ["researcher", "trader", "general"]:
    workflow.add_edge(node, END)

app = workflow.compile()

# ======================== UI ========================
st.title("🚀 NVIDIA AI Assistant – Multi-Agent System")
st.caption("NTU DSAI Capstone | Intelligent Hybrid ML + LLM Forecasting")

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
    st.success("✅ Intelligent Hybrid Forecasting")
    st.success("✅ Full RAG + Prophet + DDG")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    st.caption("Running on Standard Instance (2 GB RAM)")