import os
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from dotenv import load_dotenv

# Memory optimization
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHROMA_NO_SERVER"] = "true"

# LangChain & LangGraph
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
import gc

load_dotenv()

# ======================== CONFIG ========================
st.set_page_config(page_title="🚀 NVIDIA AI Assistant", page_icon="📈", layout="wide")

# Paths
DB_PATH = "./chroma_db_v2"
CSV_PATH = "nvda_2014_to_2026.csv"
MODEL_PATH = "nvidia_price_model.pkl"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not found in .env")
    st.stop()

# Use lighter model for Render free/starter plan
llm = ChatOpenAI(model="gpt-5.4", temperature=0.3, max_tokens=1024)

# ======================== LOAD COMPONENTS ========================
@st.cache_resource
def load_rag():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",      # Much lighter than all-mpnet
            model_kwargs={"device": "cpu"}
        )
        import chromadb
        from chromadb.config import Settings
        
        client = chromadb.PersistentClient(
            path=DB_PATH, 
            settings=Settings(allow_reset=True)
        )
        
        vectorstore = Chroma(
            client=client,
            collection_name="nvidia_annual_reports_2014_2025",
            embedding_function=embeddings
        )
        st.sidebar.success("✅ RAG (ChromaDB) Loaded")
        return vectorstore
    except Exception as e:
        st.sidebar.error(f"❌ RAG Load Failed: {str(e)[:100]}")
        return None


@st.cache_resource
def load_model_and_data():
    try:
        df = pd.read_csv(CSV_PATH, skiprows=[1])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        checkpoint = joblib.load(MODEL_PATH)
        return df, checkpoint
    except Exception as e:
        st.sidebar.warning(f"Model/Data load issue: {e}")
        return None, None


vectorstore = load_rag()
df, model_checkpoint = load_model_and_data()
ddg_search = DuckDuckGoSearchRun()

# ======================== LANGGRAPH STATE ========================
class AgentState(TypedDict):
    query: str
    response: str
    debug_log: Annotated[str, operator.add]
    next_node: str

# ======================== NODES ========================
def router_node(state: AgentState) -> AgentState:
    q = state["query"].lower()
    if any(x in q for x in ["predict", "forecast", "price", "7 day", "next week", "tomorrow"]):
        state["next_node"] = "trader"
        state["debug_log"] = "🚦 Router → Trader\n"
    elif any(x in q for x in ["news", "outlook", "blackwell", "risk", "revenue", "financial", "huawei"]):
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
        docs = vectorstore.similarity_search(query, k=4)
        context = "\n\n".join([d.page_content[:600] for d in docs])

    news = ddg_search.run(f"NVIDIA {query} latest news OR earnings OR Blackwell")[:700]

    resp = llm.invoke([
        SystemMessage(content="You are NVIDIA expert researcher. Be concise and accurate."),
        HumanMessage(content=f"Query: {query}\n\nRAG Context:\n{context}\n\nLatest News:\n{news}")
    ]).content

    state["response"] = resp
    state["debug_log"] += "🔎 Researcher completed\n"
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

        trade_rec = "🟢 **STRONG BUY**" if upside > 4 else "🟡 **BUY**" if upside > 1.8 else "🔴 **SELL/CAUTION**" if upside < -2.5 else "⚪ **HOLD**"

        lines = [f"{row['ds']}: **${row['yhat']:.2f}** (${row['yhat_lower']:.2f}–${row['yhat_upper']:.2f})" 
                for _, row in pred.iterrows()]

        state["response"] = f"""**🚀 NVIDIA 7-DAY FORECAST**

**Current Price:** ${last_close:.2f} → **Day 7:** **${final_pred:.2f}** ({upside:+.1f}%)

**Predictions:**
""" + "\n".join(lines) + f"\n\n**Trade Recommendation:** {trade_rec}"

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
    state["debug_log"] += "💬 General Agent completed\n"
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

# ======================== STREAMLIT UI ========================
st.title("🚀 NVIDIA AI Assistant – Multi-Agent System")
st.caption("NTU DSAI Capstone | RAG + Prophet + DuckDuckGo")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything about NVIDIA (price, news, Blackwell, etc.)"):
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
    st.success("✅ Prophet ML Model Loaded")
    st.success("✅ RAG (MiniLM) Loaded")
    st.success("✅ DuckDuckGo Ready")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.caption("Memory optimized for Render Starter")