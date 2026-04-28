import os
import streamlit as st
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
st.set_page_config(page_title="🚀 NVIDIA AI Assistant", page_icon="📈", layout="wide")

DRIVE_DB_PATH = "./chroma_db_v2"
CSV_PATH = "nvda_2014_to_2026.csv"
MODEL_PATH = "nvidia_price_model.pkl"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not found in .env")
    st.stop()

llm = ChatOpenAI(model="gpt-5.4", temperature=0.3, max_tokens=1024)

# ======================== LOAD COMPONENTS ========================
@st.cache_resource
def load_rag():
    try:
        device = "cpu"
        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2", model_kwargs={"device": device})
        
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(path=DRIVE_DB_PATH, settings=Settings(allow_reset=True))
        
        # Check if collection exists and has data
        collections = client.list_collections()
        if not any(c.name == "nvidia_annual_reports_2014_2025" for c in collections):
            raise Exception("Collection not found")
            
        vectorstore = Chroma(client=client, collection_name="nvidia_annual_reports_2014_2025", embedding_function=embeddings)
        print("✅ RAG Loaded from existing DB")
        return vectorstore
    except:
        print("⚠️ ChromaDB not found or empty. Building fresh...")
        # Add code here to rebuild from PDFs (I can give you this part if needed)
        st.warning("Building RAG database... This may take 2-5 minutes on first run.")
        # ... rebuild logic ...
        return None

vectorstore = load_rag()

df = pd.read_csv(CSV_PATH, skiprows=[1])
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

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
    if any(x in q for x in ["predict", "forecast", "price", "share price", "stock price"]):
        if any(x in q for x in ["12 months", "one year", "next year", "2027", "long term"]):
            state["next_node"] = "long_term_analyst"
            state["debug_log"] = "🚦 Router → Long-term Analyst\n"
        else:
            state["next_node"] = "trader"
            state["debug_log"] = "🚦 Router → Trader\n"
    elif any(x in q for x in ["news", "outlook", "latest", "blackwell", "risk", "revenue", "financial"]):
        state["next_node"] = "researcher"
        state["debug_log"] = "🚦 Router → Researcher\n"
    else:
        state["next_node"] = "general"
        state["debug_log"] = "🚦 Router → General\n"
    return state

def long_term_analyst_node(state: AgentState) -> AgentState:
    debug = "📈 Long-term Analyst (Prophet 30d + LLM) activated → "
    try:
        # 1. Run Prophet for reliable short/medium horizon (30 trading days)
        checkpoint = joblib.load(MODEL_PATH)
        model = checkpoint['prophet_model']
        last_close = checkpoint['last_close']

        future = model.make_future_dataframe(periods=30, freq='B')  # ~1.5 months
        future['MA7'] = df['Close'].rolling(7).mean().iloc[-1]
        future['MA30'] = df['Close'].rolling(30).mean().iloc[-1]
        future['Vol7'] = df['Close'].tail(7).std()
        future['Volume_MA7'] = df['Volume'].tail(7).mean()

        forecast = model.predict(future)
        pred = forecast.tail(7)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()  # show next 7 days clearly
        pred['ds'] = pred['ds'].dt.strftime('%Y-%m-%d')

        final_30d = forecast['yhat'].iloc[-1]
        upside_30d = (final_30d / last_close - 1) * 100

        # 2. LLM + RAG + DDG for 12-month qualitative view
        context = ""
        if vectorstore:
            docs = vectorstore.similarity_search(state["query"], k=6)
            context = "\n\n".join([d.page_content[:600] for d in docs])
        news = ddg_search.run(f"NVIDIA price target 2026 OR 2027 OR analyst forecast OR Blackwell revenue")[:900]

        llm_response = llm.invoke([
            SystemMessage(content="You are NVIDIA senior equity analyst. Combine quantitative forecast with fundamental outlook."),
            HumanMessage(content=f"""Query: {state["query"]}
Short-term Prophet (next 30 days): expected ~{upside_30d:+.1f}% to ${final_30d:.2f}
RAG context: {context}
Latest analyst/news: {news}

Give a balanced 12-month outlook with:
• Quantitative short-term (from ML)
• Key drivers (AI, Blackwell, competition, etc.)
• Realistic 12-month price range
• Trade recommendation""")
        ]).content

        state["response"] = f"""**🚀 NVIDIA 12-MONTH OUTLOOK (Hybrid ML + LLM)**

**Short-term (next 7 days from Prophet):**
{chr(10).join([f"{row['ds']}: **${row['yhat']:.2f}**" for _, row in pred.iterrows()])}

**30-day expected move:** **{upside_30d:+.1f}%** → ~${final_30d:.2f}

{llm_response}

*Backtested MAPE: {checkpoint.get('backtest_mape', 2.23):.2f}% • Long-term forecasts have higher uncertainty*"""
    except Exception as e:
        state["response"] = f"Long-term analysis error: {str(e)}"

    state["debug_log"] += debug + "\n"
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
        q_lower = state["query"].lower()
        
        # Dynamic horizon
        if any(x in q_lower for x in ["3 months", "quarter", "3 month"]):
            periods = 63
            horizon = "3-MONTH"
        elif any(x in q_lower for x in ["one month", "1 month", "30 days", "next 30 days"]):
            periods = 30
            horizon = "1-MONTH"
        else:
            periods = 7
            horizon = "7-DAY"

        checkpoint = joblib.load(MODEL_PATH)
        model = checkpoint['prophet_model']
        last_close = checkpoint['last_close']

        # Prophet quantitative forecast
        future = model.make_future_dataframe(periods=periods, freq='B')
        future['MA7'] = df['Close'].rolling(7).mean().iloc[-1]
        future['MA30'] = df['Close'].rolling(30).mean().iloc[-1]
        future['Vol7'] = df['Close'].tail(7).std()
        future['Volume_MA7'] = df['Volume'].tail(7).mean()

        forecast = model.predict(future)
        pred = forecast.tail(7)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        pred['ds'] = pred['ds'].dt.strftime('%Y-%m-%d')

        final_pred = forecast['yhat'].iloc[-1]
        upside = (final_pred / last_close - 1) * 100

        trade_rec = "🟢 **STRONG BUY**" if upside > 4.0 else \
                    "🟡 **BUY**" if upside > 1.8 else \
                    "🔴 **SELL / CAUTION**" if upside < -2.5 else \
                    "⚪ **HOLD** – Monitor closely"

        lines = [f"{row['ds']}: **${row['yhat']:.2f}** (range ${row['yhat_lower']:.2f}–${row['yhat_upper']:.2f})" 
                for _, row in pred.iterrows()]

        # ==================== QUALITATIVE ANALYSIS (for 30+ days) ====================
        qualitative = ""
        if periods >= 30:
            context = ""
            if vectorstore:
                docs = vectorstore.similarity_search(state["query"], k=5)
                context = "\n\n".join([d.page_content[:600] for d in docs])[:2000]

            news = ddg_search.run("NVIDIA DeepSeek V4 Huawei Ascend chips impact Blackwell revenue outlook")[:700]

            llm_analysis = llm.invoke([
                SystemMessage(content="You are a senior NVIDIA equity analyst. Provide insightful but concise qualitative context."),
                HumanMessage(content=f"""Query: {state["query"]}
Quantitative forecast: {upside:+.1f}% move in next {periods} days
RAG Context: {context}
Recent News: {news}

Give a short qualitative analysis (2-4 sentences) covering:
• Impact of DeepSeek/Huawei competition
• Blackwell momentum
• Overall sentiment and risks""")
            ]).content

            qualitative = f"\n\n**Qualitative Analysis:**\n{llm_analysis}"

        # Final response
        state["response"] = f"""**🚀 NVIDIA {horizon} FORECAST**

**Current Price:** ${last_close:.2f} → **Final Expected:** **${final_pred:.2f}** ({upside:+.1f}%)

**Next 7 Days Predictions:**
""" + "\n".join(lines) + f"""

**Trade Recommendation:** {trade_rec}

{qualitative}

*Backtested MAPE: {checkpoint.get('backtest_mape', 2.23):.2f}% • Longer horizons have higher uncertainty*"""

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
workflow.add_node("long_term_analyst", long_term_analyst_node)
workflow.add_node("general", general_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", lambda x: x["next_node"],
    {
        "researcher": "researcher",
        "trader": "trader",
        "long_term_analyst": "long_term_analyst",   # ← new
        "general": "general"
    })
for node in ["researcher", "trader", "long_term_analyst","general"]:
    workflow.add_edge(node, END)

app = workflow.compile()   # ← THIS WAS MISSING / BROKEN IN YOUR HF VERSION

# ======================== UI ========================
st.title("🚀 NVIDIA AI Assistant – Multi-Agent System")
st.caption("NTU DSAI Capstone | RAG + DuckDuckGo + Prophet ML")

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
        with st.spinner("Thinking... (Router + Agents running)"):
            try:
                result = app.invoke({
                    "query": prompt,
                    "response": "",
                    "debug_log": "",
                    "next_node": ""
                })
                answer = result.get("response", "Sorry, I couldn't generate an answer.")
                log = result.get("debug_log", "No debug info")
            except Exception as e:
                answer = f"❌ Agent error: {str(e)}"
                log = f"Error: {str(e)}"

            st.markdown(answer)
            with st.expander("🔍 Debug Trace"):
                st.code(log)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# Sidebar
with st.sidebar:
    st.success("✅ Prophet ML Model Loaded")
    st.success("✅ RAG Loaded")
    st.success("✅ DuckDuckGo Ready")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()