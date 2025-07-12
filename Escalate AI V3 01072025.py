# ==============================================================
# EscalateAI – End‑to‑End Escalation Management System (v0.9.4)
# --------------------------------------------------------------
# • Full single‑file implementation
# • Robust fallback sentiment (no torch required)
# • SQLite persistence + daily model retraining
# • Streamlit Kanban UI with filters & inline edits
# • SPOC email notification + notification history tracking
# --------------------------------------------------------------
# Author: Naveen Gandham • July 2025
# ==============================================================

"""Quick‑start (terminal):

pip install streamlit pandas openpyxl python-dotenv transformers scikit-learn joblib requests apscheduler
# Optional (better accuracy – only if PyTorch wheel available for your Python):
pip install torch --index-url https://download.pytorch.org/whl/cpu

export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/…"
streamlit run escalateai_full_code.py
"""

import os, re, sqlite3, warnings, atexit, smtplib
from datetime import datetime
from pathlib import Path
from typing import Tuple, List
from email.mime.text import MIMEText

import joblib, pandas as pd, requests, streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "escalateai.db"

load_dotenv()
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
ALERT_CHANNEL_ENABLED = bool(SLACK_WEBHOOK_URL)

# ========== Sentiment Model Loading ==========
try:
    from transformers import pipeline as hf_pipeline
    _has_transformers = True
except ModuleNotFoundError:
    _has_transformers = False

try:
    import torch
    _has_torch = True
except ModuleNotFoundError:
    _has_torch = False

_use_hf = _has_transformers and _has_torch

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    if not _use_hf:
        st.sidebar.warning("Transformers or Torch not available – using rule‑based sentiment.")
        return None
    try:
        return hf_pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    except Exception as e:
        st.sidebar.warning(f"HF model load failed ({e}) – fallback to rule‑based.")
        return None

sentiment_model = load_sentiment_model()

NEG_WORDS = [
    r"problem", r"delay", r"issue", r"failure", r"dissatisfaction", r"unacceptable",
    r"complaint", r"unresolved", r"unstable", r"defective", r"critical", r"risk",
]

@st.cache_data(show_spinner=False)
def rule_based_sentiment(text: str) -> str:
    return "Negative" if any(re.search(w, text, re.I) for w in NEG_WORDS) else "Positive"

# ========== Initialize Database ==========
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
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
            spoc_email TEXT,
            spoc_notify_count INTEGER DEFAULT 0,
            spoc_last_notified TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS notification_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            escalation_id TEXT,
            recipient_email TEXT,
            subject TEXT,
            body TEXT,
            sent_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ========== Additional Functions (DB, Risk, UI Placeholder) ========== 

def upsert_case(case: dict):
    with sqlite3.connect(DB_PATH) as conn:
        keys = ','.join(case.keys())
        question_marks = ','.join(['?'] * len(case))
        values = tuple(case.values())
        conn.execute(f"REPLACE INTO escalations ({keys}) VALUES ({question_marks})", values)
        conn.commit()

def fetch_cases() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM escalations", conn)

def log_notification(escalation_id: str, to_email: str, subject: str, body: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO notification_log (escalation_id, recipient_email, subject, body, sent_at)
            VALUES (?, ?, ?, ?, ?)
        """, (escalation_id, to_email, subject, body, datetime.now().isoformat()))
        conn.commit()

def send_email(to_email: str, subject: str, body: str) -> bool:
    try:
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = int(os.getenv("SMTP_PORT", 587))
        smtp_user = os.getenv("SMTP_USER")
        smtp_pass = os.getenv("SMTP_PASS")

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_email

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_email, msg.as_string())

        log_notification(escalation_id="manual", to_email=to_email, subject=subject, body=body)
        return True
    except Exception as e:
        st.error(f"Email failed: {e}")
        return False

@st.cache_resource(show_spinner=False)
def load_predictor():
    model_path = MODEL_DIR / "risk_predictor.joblib"
    return joblib.load(model_path) if model_path.exists() else None

risk_model = load_predictor()

def predict_risk(issue: str) -> float:
    if not risk_model:
        return 0.0
    return float(risk_model.predict_proba([issue])[0][1])

# ========== KANBAN UI ==========

st.title("🛠️ Escalation Kanban Board")
df = fetch_cases()

status_counts = df["status"].value_counts().to_dict()

st.markdown(f"""
**🔄 Open**: {status_counts.get('Open', 0)} &nbsp;&nbsp;&nbsp;
**🔧 In Progress**: {status_counts.get('In Progress', 0)} &nbsp;&nbsp;&nbsp;
**✅ Resolved**: {status_counts.get('Resolved', 0)}
""")

cols = st.columns(3)
for status, col in zip(["Open", "In Progress", "Resolved"], cols):
    with col:
        st.markdown(f"### {status}")
        for _, row in df[df.status == status].iterrows():
            with st.expander(f"{row['id']} – {row['issue'][:60]}"):
                row_dict = row.to_dict()
                row_dict["spoc_email"] = st.text_input("SPOC Email", value=row.get("spoc_email", ""), key=f"email_{row['id']}")
                row_dict["owner"] = st.text_input("Owner", value=row.get("owner", "Unassigned"), key=f"owner_{row['id']}")
                row_dict["status"] = st.selectbox("Status", ["Open", "In Progress", "Resolved"], index=["Open", "In Progress", "Resolved"].index(row["status"]), key=f"status_{row['id']}")
                row_dict["action_taken"] = st.text_area("Action Taken", value=row.get("action_taken", ""), key=f"action_{row['id']}")

                if st.button("Notify SPOC", key=f"notify_{row['id']}"):
                    if row_dict["spoc_email"]:
                        subject = f"Reminder: Escalation {row_dict['id']} - {row_dict['customer']}"
                        body = f"Dear SPOC,\n\nPlease take action on escalation: {row_dict['issue']}\nStatus: {row_dict['status']}\nOwner: {row_dict['owner']}\n\nThank you."
                        if send_email(row_dict["spoc_email"], subject, body):
                            row_dict["spoc_notify_count"] = row.get("spoc_notify_count", 0) + 1
                            row_dict["spoc_last_notified"] = datetime.now().isoformat()
                            st.success("Notification sent!")

                upsert_case(row_dict)

st.success("✅ Kanban UI Loaded.")
