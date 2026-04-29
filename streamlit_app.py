import os
import streamlit as st
import pandas as pd
import joblib
from dotenv import load_dotenv
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHROMA_NO_SERVER"] = "true"

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

load_dotenv()

st.set_page_config(page_title="🚀 NVIDIA AI Assistant", page_icon="📈", layout="wide")

# ======================== CONFIG ========================
DB_PATH = "./chroma_db_v2"
CSV_PATH = "nvda_2014_to_2026.csv"
MODEL_PATH = "nvidia_price_model.pkl"

llm = ChatOpenAI(model="gpt-5.4", temperature=0.3, max_tokens=1024)

ddg_search = DuckDuckGoSearchRun()   # very light

# ======================== LAZY LOADERS ========================
@st.cache_resource(show_spinner=False)
def get_vectorstore():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(path=DB_PATH, settings=Settings(allow_reset=True))
        vs = Chroma(client=client, collection_name="nvidia_annual_reports_2014_2025", embedding_function=embeddings)
        return vs
    except Exception as e:
        st.sidebar.error(f"RAG failed: {str(e)[:80]}")
        return None


@st.cache_resource(show_spinner=False)
def get_model_and_df():
    try:
        df = pd.read_csv(CSV_PATH, skiprows=[1])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        checkpoint = joblib.load(MODEL_PATH)
        return df, checkpoint
    except Exception:
        return None, None


# ======================== AGENT STATE & NODES ========================
class AgentState(TypedDict):
    query: str
    response: str
    debug_log: Annotated[str, operator.add]
    next_node: str


def router_node(state: AgentState) -> AgentState:
    q = state["query"].lower()
    if any(x in q for x in ["predict", "forecast", "price", "7 day", "next week", "tomorrow"]):
        state["next_node"] = "trader"
    elif any(x in q for x in ["news", "outlook", "blackwell", "risk", "revenue", "financial", "huawei"]):
        state["next_node"] = "researcher"
    else:
        state["next_node"] = "general"
    state["debug_log"] = f"🚦 Router → {state['next_node']}\n"
    return state


def researcher_node(state: AgentState) -> AgentState:
    vectorstore = get_vectorstore()
    query = state["query"]
    context = ""
    if vectorstore:
        docs = vectorstore.similarity_search(query, k=4)
        context = "\n\n".join([d.page_content[:600] for d in docs])

    news = ddg_search.run(f"NVIDIA {query} latest")[:700]

    resp = llm.invoke([
        SystemMessage(content="You are NVIDIA expert researcher. Be concise."),
        HumanMessage(content=f"Query: {query}\nRAG:\n{context}\nNews:\n{news}")
    ]).content

    state["response"] = resp
    state["debug_log"] += "🔎 Researcher done\n"
    return state


def trader_node(state: AgentState) -> AgentState:
    df, checkpoint = get_model_and_df()
    if not checkpoint:
        state["response"] = "❌ Model not loaded."
        return state

    try:
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
        trade_rec = "🟢 **STRONG BUY**" if upside > 4 else "🟡 **BUY**" if upside > 1.8 else "🔴 **SELL/CAUTION**" if upside < -2.5 else "⚪ **HOLD**"

        lines = [f"{row['ds']}: **${row['yhat']:.2f}** (${row['yhat_lower']:.2f}–${row['yhat_upper']:.2f})" for _, row in pred.iterrows()]

        state["response"] = f"""**🚀 NVIDIA 7-DAY FORECAST**

**Current:** ${last_close:.2f} → **Day 7:** **${final_pred:.2f}** ({upside:+.1f}%)
**Recommendation:** {trade_rec}

**Predictions:**
""" + "\n".join(lines)
    except Exception as e:
        state["response"] = f"Forecast error: {str(e)}"

    state["debug_log"] += "📈 Trader done\n"
    return state


def general_node(state: AgentState) -> AgentState:
    resp = llm.invoke([SystemMessage(content="Helpful NVIDIA assistant."), HumanMessage(content=state["query"])]).content
    state["response"] = resp
    state["debug_log"] += "💬 General done\n"
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

# ======================== UI ========================
st.title("🚀 NVIDIA AI Assistant – Multi-Agent System")
st.caption("NTU DSAI Capstone | Extremely Memory-Optimized")

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
            answer = result.get("response", "Sorry, couldn't generate answer.")
            log = result.get("debug_log", "")

            st.markdown(answer)
            with st.expander("🔍 Debug Trace"):
                st.code(log)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# Sidebar
with st.sidebar:
    st.success("✅ Lazy Loading Active")
    st.success("✅ MiniLM + CPU Only")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    st.caption("Optimized for Render Starter ($7)")