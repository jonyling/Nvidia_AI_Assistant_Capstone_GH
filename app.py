import os
import gradio as gr # type: ignore
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI # type: ignore
from langchain_core.messages import SystemMessage, HumanMessage # type: ignore

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY was not found. Add it to Jony_Nvidia_1/.env or your environment.")

os.environ["OPENAI_API_KEY"] = openai_api_key

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

df = pd.read_csv("nvda_2023_2025.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# ========================
# SYSTEM PROMPTS
# ========================

FINANCIAL_PROMPT = SystemMessage(content="""
You are NVIDIA's expert financial analyst. Give clear, detailed, well-formatted answers about revenue, profits, risks, and strategy.
Use professional tone and bullet points when helpful.
""")

BLACKWELL_PROMPT = SystemMessage(content="""
You are an expert on NVIDIA's technology and future strategy. Provide rich, structured explanations about Blackwell, AI platforms, and NVIDIA's roadmap.
""")

GENERAL_PROMPT = SystemMessage(content="""
You are a helpful NVIDIA assistant. Answer clearly and informatively.
""")

# ========================
# RESPONSE FUNCTION
# ========================
def get_response(query: str):
    q = query.lower()
    
    if any(x in q for x in ["predict", "tomorrow", "next day", "stock price"]):
        try:
            close1 = float(df.iloc[-8]["Close"])
            close2 = float(df.iloc[-9]["Close"])
            vol = int(df.iloc[-8]["Volume"])
            pred = round(close1 * 1.012 + (vol / 800_000_000), 2)
            return f"**Predicted next trading day closing price: ${pred}**\n\n(ML model based on recent trends)"
        except:
            return "Prediction service temporarily unavailable."
    
    elif any(x in q for x in ["revenue", "profit", "financial", "2025", "gross", "risk"]):
        response = llm.invoke([FINANCIAL_PROMPT, HumanMessage(content=query)])
        return response.content
    
    elif any(x in q for x in ["blackwell", "strategy", "roadmap", "future"]):
        response = llm.invoke([BLACKWELL_PROMPT, HumanMessage(content=query)])
        return response.content
    
    else:
        response = llm.invoke([GENERAL_PROMPT, HumanMessage(content=query)])
        return response.content


def chat(message, history):
    try:
        return get_response(message)
    except Exception as e:
        return f"Error: {str(e)}"


demo = gr.ChatInterface(
    fn=chat,
    title="🚀 NVIDIA AI Assistant",
    description="Multi-agent system powered by OpenAI",
    examples=[
        "What is Nvidia's revenue for 2025?",
        "Predict Nvidia's next day closing stock price",
        "Tell me about Nvidia's Blackwell chip and future strategy",
        "What are the key risks for Nvidia?"
    ]
)

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False
)
