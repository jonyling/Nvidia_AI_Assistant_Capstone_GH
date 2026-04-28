import os
import time
import schedule
import telebot
from dotenv import load_dotenv
import joblib
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ======================== CONFIG ========================
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

bot = telebot.TeleBot(BOT_TOKEN)
llm = ChatOpenAI(model="gpt-5.4", temperature=0.3)

# Load components (same as before)
vectorstore = None
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path="./chroma_db_v2", settings=Settings(allow_reset=True))
    vectorstore = Chroma(client=client, collection_name="nvidia_annual_reports_2014_2025", embedding_function=embeddings)
except Exception as e:
    print(f"⚠️ RAG failed: {e}")

ddg_search = DuckDuckGoSearchRun()
checkpoint = joblib.load("nvidia_price_model.pkl")
prophet_model = checkpoint['prophet_model']
last_close = checkpoint['last_close']

df = pd.read_csv("nvda_2014_to_2026.csv", skiprows=[1])
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# ======================== CHART GENERATION ========================
def generate_forecast_chart():
    try:
        future = prophet_model.make_future_dataframe(periods=30, freq='B')
        future['MA7'] = df['Close'].rolling(7).mean().iloc[-1]
        future['MA30'] = df['Close'].rolling(30).mean().iloc[-1]
        future['Vol7'] = df['Close'].tail(7).std()
        future['Volume_MA7'] = df['Volume'].tail(7).mean()

        forecast = prophet_model.predict(future)
        
        fig = prophet_model.plot(forecast)
        plt.title("NVIDIA 30-Day Price Forecast (Prophet Model)")
        plt.xlabel("Date")
        plt.ylabel("NVDA Share Price (USD)")
        plt.grid(True)
        
        chart_path = "nvidia_daily_forecast.png"
        plt.savefig(chart_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return chart_path
    except Exception as e:
        print(f"Chart generation error: {e}")
        return None


# ======================== DYNAMIC FORECAST (keep your existing one) ========================
# ... Paste your dynamic_forecast function here (from previous version) ...

# ======================== DAILY ALERT WITH CHART ========================
def send_daily_alert():
    forecast_text = dynamic_forecast("daily forecast")   # Reuse your smart function
    news = ddg_search.run("NVIDIA latest news OR Blackwell OR Huawei")[:500]

    alert_text = f"""🚨 **NVIDIA DAILY ALERT** {datetime.now().strftime('%Y-%m-%d')}

{forecast_text}

📰 **Latest News:**
{news}
"""

    chart_path = generate_forecast_chart()
    
    try:
        if chart_path and os.path.exists(chart_path):
            with open(chart_path, 'rb') as photo:
                bot.send_photo(CHAT_ID, photo, caption=alert_text[:1000])  # Caption has limit
            print("✅ Daily alert with chart sent")
        else:
            bot.send_message(CHAT_ID, alert_text)
            print("✅ Daily alert (text only) sent")
    except Exception as e:
        print(f"Daily alert error: {e}")
        bot.send_message(CHAT_ID, alert_text[:4000])


# ======================== BREAKING NEWS (keep as before) ========================
def check_breaking_news():
    # ... (your existing breaking news function) ...
    pass   # Keep the one from previous version


# ======================== BOT HANDLERS ========================
# ... (keep your existing /start, /forecast, /news, handle_message) ...


# ======================== START ========================
if __name__ == "__main__":
    print("🚀 Nvidia_bot with Daily Chart Alert + Breaking News running...")

    schedule.every(30).minutes.do(check_breaking_news)
    schedule.every().day.at("08:00").do(send_daily_alert)   # Daily at 8 AM

    send_daily_alert()  # Send one immediately on start

    while True:
        try:
            schedule.run_pending()
            bot.polling(none_stop=True, interval=1, timeout=30)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)