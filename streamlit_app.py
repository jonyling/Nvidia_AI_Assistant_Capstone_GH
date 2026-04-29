import os
import streamlit as st
import pandas as pd
import joblib
from dotenv import load_dotenv

# ==================== OPTIMIZATIONS ====================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHROMA_NO_SERVER"] = "true"
os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

load_dotenv()

st.set_page_config(page_title="🚀 NVIDIA AI Assistant", page_icon="📈", layout="wide")

# ======================== CONFIG ========================
DB_PATH = "./chroma_db_v2"
CSV_PATH = "nvda_2014_to_2026.csv"
MODEL_PATH = "nvidia_price_model.pkl"

# ======================== IMPORTS ========================
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

# ======================== LOAD COMPONENTS (2GB Safe) ========================
@st.cache_resource
def load_all_components():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-mpnet-base-v2",          # Must match your collection (768 dim)
        model_kwargs={"device": "cpu"}
    )
    client = chromadb.PersistentClient(
        path=DB_PATH,
        settings=Settings(allow_reset=True, anonymized_telemetry=False)
    )
    vectorstore = Chroma(
        client=client,
        collection_name="nvidia_annual_reports_2014_2025",
        embedding_function=embeddings
    )

    df = pd.read_csv(CSV_PATH, skiprows=[1])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    checkpoint = joblib.load(MODEL_PATH)

    return vectorstore, df, checkpoint


vectorstore, df, model_checkpoint = load_all_components()

# ======================== AGENT STATE ========================
class AgentState(TypedDict):
    query: str
    response: str
    debug_log: Annotated[str, operator.add]
    next_node: str


# ======================== INTELLIGENT LLM ROUTER ========================
def router_node(state: AgentState) -> AgentState:
    query = state["query"]

    router_prompt = f"""You are an intelligent router for the NVIDIA AI Assistant.
Available agents:
- trader: ONLY for direct stock price prediction, 7-day forecast, tomorrow's price, next week price.
- researcher: For news, financials, strategy, competitors (Huawei, DeepSeek, Ascend), Blackwell, risks, revenue, gross margin, business impact, etc.
- general: Everything else or simple questions.

Decide the SINGLE best agent for this query. Reply with ONLY one word: trader, researcher, or general.

Query: {query}
Answer:"""

    decision = llm.invoke([HumanMessage(content=router_prompt)]).content.strip().lower()

    if "trader" in decision:
        state["next_node"] = "trader"
        state["debug_log"] = "🚦 LLM Router → Trader\n"
    elif "researcher" in decision:
        state["next_node"] = "researcher"
        state["debug_log"] = "🚦 LLM Router → Researcher\n"
    else:
        state["next_node"] = "general"
        state["debug_log"] = "🚦 LLM Router → General\n"

    return state


# ======================== NODES ========================
def researcher_node(state: AgentState) -> AgentState:
    query = state["query"]
    context = ""
    if vectorstore:
        docs = vectorstore.similarity_search(query, k=5)
        context = "\n\n".join([d.page_content[:800] for d in docs])

    news = ddg_search.run(f"NVIDIA {query} latest news OR earnings OR Blackwell OR Huawei OR DeepSeek")[:800]

    resp = llm.invoke([
        SystemMessage(content="You are NVIDIA expert researcher. Use RAG context and latest news. Be concise, factual and professional."),
        HumanMessage(content=f"Query: {query}\n\nRAG Context:\n{context}\n\nLatest News:\n{news}")
    ]).content

    state["response"] = resp
    state["debug_log"] += "🔎 Researcher completed (RAG + DDG)\n"
    return state


def trader_node(state: AgentState) -> AgentState:
    if not model_checkpoint:
        state["response"] = "❌ Prophet model not available."
        return state

    try:
        model = model_checkpoint['prophet_model']
        last_close = model_checkpoint['last_close']

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

        trade_rec = (
            "🟢 **STRONG BUY** – High confidence" if upside > 4 else
            "🟡 **BUY** – Good opportunity" if upside > 1.8 else
            "🔴 **SELL / CAUTION**" if upside < -2.5 else
            "⚪ **HOLD** – Monitor closely"
        )

        lines = [f"**{row['ds']}**: ${row['yhat']:.2f} (range: ${row['yhat_lower']:.2f}–${row['yhat_upper']:.2f})" 
                 for _, row in pred.iterrows()]

        state["response"] = f"""**🚀 NVIDIA 7-DAY FORECAST**

**Current Price:** ${last_close:.2f}  
**Day 7 Expected:** **${final_pred:.2f}** ({upside:+.1f}%)

**Trade Recommendation:** {trade_rec}

**Detailed Predictions:**
""" + "\n".join(lines) + f"""

*Backtested MAPE: {model_checkpoint.get('backtest_mape', 'N/A'):.2f}%*"""

    except Exception as e:
        state["response"] = f"Forecast error: {str(e)}"

    state["debug_log"] += "📈 Trader completed\n"
    return state


def general_node(state: AgentState) -> AgentState:
    resp = llm.invoke([
        SystemMessage(content="You are a helpful NVIDIA assistant."),
        HumanMessage(content=state["query"])
    ]).content
    state["response"] = resp
    state["debug_log"] += "💬 General completed\n"
    return state


# ======================== BUILD GRAPH ========================
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
st.caption("NTU DSAI Capstone | Intelligent LLM Router + Full RAG + Prophet + DuckDuckGo | 2GB Optimized")

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
            result = app.invoke({
                "query": prompt,
                "response": "",
                "debug_log": "",
                "next_node": ""
            })
            answer = result.get("response", "Sorry, I couldn't generate an answer.")
            log = result.get("debug_log", "")

            st.markdown(answer)
            with st.expander("🔍 Debug Trace"):
                st.code(log)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# Sidebar
with st.sidebar:
    st.success("✅ Intelligent LLM Router")
    st.success("✅ Full RAG + Prophet + DDG Ready")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    st.caption("Running on Standard Instance (2 GB RAM)")