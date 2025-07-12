# ==============================================================
# EscalateAI – End‑to‑End Escalation Management System (v0.9.1)
# --------------------------------------------------------------
# Adds robust fallback when HuggingFace sentiment model cannot load
# (e.g., PyTorch not available or incompatible Python version).
# If HF model fails, the app silently switches to a simple rule‑based
# classifier so the UI stays operational, then logs a warning in the
# Streamlit sidebar.
# --------------------------------------------------------------
# Author: Naveen Gandham • July 2025
# ==============================================================

"""Quick‑start (terminal):

pip install streamlit pandas openpyxl python-dotenv transformers scikit-learn joblib requests apscheduler
# Optional (better accuracy):
pip install torch --index-url https://download.pytorch.org/whl/cpu

export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/…"
streamlit run escalateai_full_code.py

The code auto‑creates a SQLite DB in ./data/escalateai.db and a models
folder with a baseline LogisticRegression predictor after the first
training run.
"""

# ----------------------------
# STANDARD LIBS & THIRD PARTY
# ----------------------------
import os, re, sqlite3, warnings
from datetime import datetime
from pathlib import Path
from typing import Tuple, List

import joblib, pandas as pd, requests, streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
try:
    from transformers import pipeline as hf_pipeline
    _has_transformers = True
except ModuleNotFoundError:
    _has_transformers = False

# ----------------------------
# ENV & GLOBAL CONSTANTS
# ----------------------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
MODEL_DIR = APP_DIR / "models"
DB_PATH  = DATA_DIR / "escalateai.db"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

load_dotenv()
SLACK_WEBHOOK_URL      = os.getenv("SLACK_WEBHOOK_URL", "")
ALERT_CHANNEL_ENABLED  = bool(SLACK_WEBHOOK_URL)

# ----------------------------
# DATABASE HELPERS (SQLite)
# ----------------------------

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS escalations (
            id TEXT PRIMARY KEY,
            customer TEXT,
            issue TEXT,
            criticality TEXT,
            impact TEXT,
            sentiment TEXT,
            urgency TEXT,
            escalated INTEGER,
            date_reported TEXT,
            owner TEXT,
            status TEXT,
            action_taken TEXT,
            risk_score REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit(); conn.close()


def upsert_case(case: dict):
    conn = sqlite3.connect(DB_PATH)
    cols = ", ".join(case.keys()); placeholders = ", ".join(["?"]*len(case))
    conn.execute(f"REPLACE INTO escalations ({cols}) VALUES ({placeholders})", tuple(case.values()))
    conn.commit(); conn.close()


def fetch_cases() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql("SELECT * FROM escalations", conn)

# ----------------------------
# NLP: SENTIMENT + URGENCY
# ----------------------------
NEGATIVE_URGENCY_KEYWORDS: List[str] = [
    "urgent", "critical", "immediately", "asap", "business impact", "high priority"
]

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    """Try to load HF model; fall back to rule‑based if torch/transformers fail."""
    if not _has_transformers:
        st.sidebar.warning("Transformers library not installed – using fallback sentiment.")
        return None
    try:
        return hf_pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    except Exception as e:
        st.sidebar.warning(f"⚠️ HuggingFace model load failed: {e}\nUsing rule‑based sentiment instead.")
        return None

sentiment_model = load_sentiment_model()


def rule_based_sentiment(text: str) -> str:
    negative_words = [
        r"problem", r"delay", r"issue", r"failure", r"dissatisfaction", r"unacceptable",
        r"complaint", r"unresolved", r"unstable", r"defective", r"critical", r"risk",
    ]
    return "Negative" if any(re.search(w, text, re.I) for w in negative_words) else "Positive"


def analyze_issue(text: str) -> Tuple[str, str, bool]:
    """Return sentiment • urgency • escalation bool."""
    sent = rule_based_sentiment(text) if sentiment_model is None else (
        "Negative" if sentiment_model(text[:512])[0]["label"].lower() == "negative" else "Positive"
    )
    urgency = "High" if any(k in text.lower() for k in NEGATIVE_URGENCY_KEYWORDS) else "Low"
    return sent, urgency, sent == "Negative" and urgency == "High"

# ----------------------------
# ML RISK PREDICTOR (SKLEARN)
# ----------------------------
MODEL_FILE = MODEL_DIR / "risk_predictor.joblib"

@st.cache_resource(show_spinner=False)
def load_predictor():
    return joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None

risk_model = load_predictor()


def train_predictor(df_labelled: pd.DataFrame):
    if df_labelled.empty: return None
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(df_labelled["issue"], df_labelled["escalated"].astype(int))
    joblib.dump(pipe, MODEL_FILE)
    return pipe


def predict_risk(issue: str) -> float:
    if risk_model is None: return 0.0
    return float(round(risk_model.predict_proba([issue])[0][1], 3))

# ----------------------------
# ALERTING (SLACK)
# ----------------------------

def send_slack_alert(case: dict):
    if not ALERT_CHANNEL_ENABLED: return
    txt = (
        f":rotating_light: *New Escalation* {case['id']} | {case['customer']}\n"
        f"*Issue*: {case['issue'][:180]}…\n*Urgency*: {case['urgency']} • *Sentiment*: {case['sentiment']}"
    )
    requests.post(SLACK_WEBHOOK_URL, json={"text": txt})

# ----------------------------
# UTILITIES
# ----------------------------

def next_escalation_id() -> str:
    with sqlite3.connect(DB_PATH) as conn:
        count = conn.execute("SELECT COUNT(*) FROM escalations").fetchone()[0]
    return f"ESC-{10000+count+1}"


def standardize_columns(df: pd.DataFrame):
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    return df


def ingest_dataframe(df: pd.DataFrame):
    df = standardize_columns(df)
    for _, row in df.iterrows():
        sent, urg, escal = analyze_issue(str(row.get("brief issue", "")))
        case = {
            "id": next_escalation_id(),
            "customer":   row.get("customer", "Unknown"),
            "issue":      row.get("brief issue", "Unknown"),
            "criticality":row.get("criticalness", "Unknown"),
            "impact":     row.get("impact", "Unknown"),
            "sentiment":  sent,
            "urgency":    urg,
            "escalated":  int(escal),
            "date_reported": str(row.get("issue reported date", datetime.today().date())),
            "owner":      row.get("owner", "Unassigned"),
            "status":     row.get("status", "Open"),
            "action_taken":row.get("action taken", "None"),
            "risk_score": predict_risk(row.get("brief issue", "")),
        }
        upsert_case(case); send_slack_alert(case) if escal else None

# ----------------------------
# SCHEDULER: DAILY RETRAIN
# ----------------------------

def scheduled_daily_retrain():
    df_all = fetch_cases()
    if "escalated" in df_all.columns:
        globals()["risk_model"] = train_predictor(df_all[["issue","escalated"]])
        print("[Scheduler] Model retrained", datetime.now())

scheduler = BackgroundScheduler(); scheduler.add_job(scheduled_daily_retrain, "cron", hour=2)
scheduler.start()

# ----------------------------
# STREAMLIT UI
# ----------------------------
init_db(); st.set_page_config(page_title="EscalateAI", layout="wide")
st.title("🚨 EscalateAI – Escalation Tracking System")

# -- Sidebar (Upload & Manual) --
with st.sidebar:
    st.header("📥 Upload Escalations")
    file = st.file_uploader("Excel/CSV", type=["xlsx","csv"])
    if file and st.button("Analyze & Log"):
        df_up = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)
        ingest_dataframe(df_up); st.success("File processed!")

    st.markdown("---"); st.header("✏️ Manual Entry")
    with st.form("manual"):
        cust = st.text_input("Customer"); issue = st.text_area("Issue")
        crit  = st.selectbox("Criticality", ["Low","Medium","High"], index=1)
        imp   = st.selectbox("Impact", ["Low","Medium","High"], index=1)
        owner = st.text_input("Owner", value="Unassigned")
        if st.form_submit_button("Log") and cust and issue:
            sent, urg, escal = analyze_issue(issue)
            case = {
                "id": next_escalation_id(), "customer": cust, "issue": issue,
                "criticality": crit, "impact": imp,
                "sentiment": sent, "urgency": urg, "escalated": int(escal),
                "date_reported": str(datetime.today().date()), "owner": owner,
                "status": "Open", "action_taken": "None", "risk_score": predict_risk(issue),
            }
            upsert_case(case); send_slack_alert(case) if escal else None
            st.success(f"Escalation {case['id']} logged!")

# -- Kanban Board --
df = fetch_cases()
if df.empty:
    st.info("No cases yet.")
else:
    counts = df.status.value_counts().reindex(["Open","In Progress","Resolved"], fill_value=0)
    st.subheader(f"Kanban (Open {counts['Open']} | In Progress {counts['In Progress']} | Resolved {counts['Resolved']})")
    cols = st.columns(3); statuses = {"Open": cols
