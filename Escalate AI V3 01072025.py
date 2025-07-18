# ==============================================================
# EscalateAI – End‑to‑End Escalation Management System (v1.1.2)
# --------------------------------------------------------------
# • Full single‑file Streamlit app
# • SQLite persistence & auto‑schema upgrade
# • Sentiment (HF or rule‑based) + risk ML model
# • Sidebar: Excel/CSV upload  & manual entry
# • Kanban board with inline edits & notifications
# • Notification History viewer
# • Robust SMTP email with retries
# • Scheduler escalates to boss after 2 SPOC emails & 24 h
# --------------------------------------------------------------
# Author: Naveen Gandham • July 2025
# ==============================================================

"""Quick‑start:

pip install streamlit pandas openpyxl python-dotenv transformers scikit-learn joblib requests apscheduler
# (Optional) better accuracy – only if PyTorch wheel available:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# .env (same folder)
SMTP_SERVER=smtp.mail.yahoo.com
SMTP_PORT=587
SMTP_USER=naveengandham@yahoo.co.in
SMTP_PASS=<YAHOO_APP_PASSWORD>
SLACK_WEBHOOK_URL=

streamlit run escalateai_app.py
"""

import os, re, sqlite3, atexit, smtplib, time
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import joblib, pandas as pd, streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ========== Paths & ENV ==========
APP_DIR   = Path(__file__).resolve().parent  # Directory of current file
MODEL_DIR = APP_DIR / "models"                # Models directory
DATA_DIR  = APP_DIR / "data"                  # Data directory
DB_PATH   = DATA_DIR / "escalateai.db"       # SQLite database file path

# Load environment variables from .env file
load_dotenv()
SMTP_SERVER = os.getenv("SMTP_SERVER")        # SMTP server host
SMTP_PORT   = int(os.getenv("SMTP_PORT", 587))# SMTP port, default 587
SMTP_USER   = os.getenv("SMTP_USER")          # SMTP login user
SMTP_PASS   = os.getenv("SMTP_PASS")          # SMTP login password/app password
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")  # Optional Slack notifications

# ========== Sentiment Analysis Setup ==========

# Try to import HuggingFace transformers pipeline for sentiment analysis
try:
    from transformers import pipeline as hf_pipeline
    import torch
    HAS_NLP = True
except Exception:
    HAS_NLP = False  # If import fails, fallback to rule-based sentiment

@st.cache_resource(show_spinner=False)
def load_sentiment():
    # Load HuggingFace sentiment analysis pipeline with CardiffNLP model if available
    if not HAS_NLP:
        return None
    try:
        return hf_pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    except Exception:
        return None

# Load sentiment analysis model once (cached)
sent_model = load_sentiment()

# List of regex patterns for common negative sentiment words
negative_words = [
    r"\b(problematic|delay|issue|failure|dissatisfaction|frustration|unacceptable|mistake|complaint|unresolved|unresponsive|unstable|broken|defective|overdue|escalation|leakage|damage|burnt|critical|risk|dispute|faulty)\b"
]
NEG_WORDS = negative_words

def rule_sent(text: str) -> str:
    # Rule-based sentiment detection by checking for negative words
    return "Negative" if any(re.search(w, text, re.I) for w in NEG_WORDS) else "Positive"

def analyze_issue(text: str) -> Tuple[str, str, bool]:
    """
    Analyze issue text to determine sentiment, urgency, and whether it's escalated
    Returns:
      sentiment: "Positive" or "Negative"
      urgency: "High" or "Low"
      escalated: True if sentiment is Negative and urgency is High, else False
    """
    if sent_model:
        # Use HF model for sentiment
        label = sent_model(text[:512])[0]["label"].lower()
        sentiment = "Negative" if label == "negative" else "Positive"
    else:
        # Fallback to rule-based
        sentiment = rule_sent(text)
    urgency = "High" if any(k in text.lower() for k in ["urgent", "immediate", "critical"]) else "Low"
    return sentiment, urgency, sentiment == "Negative" and urgency == "High"

# ========== Database Initialization & Helpers ==========

def init_db():
    """
    Initialize SQLite DB:
    - Create escalations table with schema for escalation tracking and notification fields
    - Create notification_log table to store sent emails history
    - Add missing columns on upgrades gracefully
    """
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        # Create escalations table if missing
        cur.execute("""CREATE TABLE IF NOT EXISTS escalations (
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
            spoc_boss_email TEXT,
            spoc_notify_count INTEGER DEFAULT 0,
            spoc_last_notified TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")

        # Create notification log table for audit trail
        cur.execute("""CREATE TABLE IF NOT EXISTS notification_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            escalation_id TEXT,
            recipient_email TEXT,
            subject TEXT,
            body TEXT,
            sent_at TEXT)""")
        conn.commit()

        # Upgrade schema: add missing columns if not present (safe to call multiple times)
        cur.execute("PRAGMA table_info(escalations)")
        cols = [c[1] for c in cur.fetchall()]
        need = {
            "spoc_notify_count": "INTEGER DEFAULT 0",
            "spoc_last_notified": "TEXT",
            "spoc_email": "TEXT",
            "spoc_boss_email": "TEXT"
        }
        for c, t in need.items():
            if c not in cols:
                try:
                    cur.execute(f"ALTER TABLE escalations ADD COLUMN {c} {t}")
                except Exception:
                    pass  # Ignore if cannot alter table (e.g. older SQLite version)
        conn.commit()

# Initialize DB at app start
init_db()

# Cached list of escalation table columns for upsert convenience
ESC_COLS = [c[1] for c in sqlite3.connect(DB_PATH).execute("PRAGMA table_info(escalations)").fetchall()]

def upsert_case(case: dict):
    """
    Insert or update an escalation record by REPLACE INTO primary key 'id'
    """
    data = {k: case.get(k) for k in ESC_COLS}
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"REPLACE INTO escalations ({','.join(data.keys())}) VALUES ({','.join('?'*len(data))})",
                     tuple(data.values()))
        conn.commit()

def fetch_cases() -> pd.DataFrame:
    """
    Load all escalation cases ordered by creation time descending
    """
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM escalations ORDER BY created_at DESC", conn)

def fetch_logs() -> pd.DataFrame:
    """
    Load all sent notification logs ordered by sent time descending
    """
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM notification_log ORDER BY sent_at DESC", conn)

# ========== Email Sending with Retries ==========

def send_email(to_email: str, subject: str, body: str, esc_id: str, retries: int = 3) -> bool:
    """
    Send email via SMTP with retry mechanism.
    Logs email to notification_log on success.
    Returns True if sent successfully, else False.
    """
    if not (SMTP_SERVER and SMTP_USER and SMTP_PASS):
        st.error("SMTP not configured")
        return False
    attempt = 0
    while attempt < retries:
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = f"Escalation Notification - SE Services <{SMTP_USER}>"
            msg["To"] = to_email
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
                s.starttls()
                s.login(SMTP_USER, SMTP_PASS)
                s.send_message(msg)
            # Log successful send in DB
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    "INSERT INTO notification_log (escalation_id, recipient_email, subject, body, sent_at) VALUES (?, ?, ?, ?, ?)",
                    (esc_id, to_email, subject, body, datetime.now().isoformat()))
                conn.commit()
            return True
        except Exception as e:
            attempt += 1
            time.sleep(2)  # small delay before retry
            if attempt == retries:
                st.error(f"Email failed: {e}")
                return False

# ========== ML Risk Prediction Model ==========

MODEL_FILE = MODEL_DIR / "risk_model.joblib"

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load pre-trained logistic regression risk model from disk if available
    """
    return joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None

risk_model = load_model()

def predict_risk(issue: str) -> float:
    """
    Predict risk score [0.0 - 1.0] for given issue text using the ML model
    """
    return float(risk_model.predict_proba([issue])[0][1]) if risk_model else 0.0

# ========== Scheduler to Escalate to Boss Email ==========

def boss_check():
    """
    Background job to:
    - Scan escalations
    - If SPOC has been notified >=2 times and no response for 24h
    - Send escalation email to SPOC's boss
    - Update notification count
    """
    try:
        df = fetch_cases()
        for _, r in df.iterrows():
            if r.get("spoc_notify_count", 0) >= 2 and r.get("spoc_boss_email") and r.get("spoc_last_notified"):
                last_notified = datetime.fromisoformat(r["spoc_last_notified"])
                if datetime.now() - last_notified > timedelta(hours=24):
                    subj = f"⚠️ Escalation {r['id']} unattended"
                    body = f"Dear Manager,\n\nEscalation {r['id']} requires your attention."
                    if send_email(r["spoc_boss_email"], subj, body, r["id"]):
                        upd = r.to_dict()
                        upd["spoc_notify_count"] += 1
                        upsert_case(upd)
    except Exception as e:
        # Warn in Streamlit if scheduler fails (rare)
        st.warning(f"Scheduler error: {e}")

# Initialize scheduler to run boss_check every hour (only once per session)
if "sched" not in st.session_state:
    sc = BackgroundScheduler()
    sc.add_job(boss_check, "interval", hours=1)
    sc.start()
    atexit.register(lambda: sc.shutdown(wait=False))  # shutdown scheduler on exit
    st.session_state["sched"] = True

import io

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Escalations', index=False)
    return output.getvalue()

# ========== Streamlit UI ==========
import streamlit as st
st.set_page_config(page_title="EscalateAI", layout="wide")
st.title("🚨 EscalateAI – Escalation Tracking System")

with st.sidebar:
    st.header("📥 Upload Escalations")
    upl = st.file_uploader("Excel / CSV", type=["xlsx", "csv"])
    if upl and st.button("Ingest File"):
        df_u = pd.read_excel(upl) if upl.name.endswith("xlsx") else pd.read_csv(upl)
        for _, row in df_u.iterrows():
            sentiment, urgency, esc = analyze_issue(str(row.get("issue", "")))
            case = {
                "id": row.get("id", f"ESC{int(datetime.utcnow().timestamp())}"),
                "customer": row.get("customer", "Unknown"),
                "issue": row.get("issue", ""),
                "criticality": row.get("criticality", "Medium"),
                "impact": row.get("impact", "Medium"),
                "sentiment": sentiment,
                "urgency": urgency,
                "escalated": int(esc),
                "date_reported": row.get("date_reported", datetime.today().date()),
                "owner": row.get("owner", "Unassigned"),
                "status": row.get("status", "Open"),
                "action_taken": row.get("action_taken", ""),
                "risk_score": predict_risk(row.get("issue", "")),
                "spoc_email": row.get("spoc_email", ""),
                "spoc_boss_email": row.get("spoc_boss_email", ""),
            }
            upsert_case(case)
        st.success("Uploaded cases ingested successfully.")

    st.header("✏️ Manual Entry")
    with st.form("manual_entry"):
        cname = st.text_input("Customer")
        issue = st.text_area("Issue")
        crit = st.selectbox("Criticality", ["Low", "Medium", "High"], index=1)
        imp = st.selectbox("Impact", ["Low", "Medium", "High"], index=1)
        owner = st.text_input("Owner", value="Unassigned")
        spoc = st.text_input("SPOC Email")
        boss = st.text_input("Boss Email")
        if st.form_submit_button("Log Escalation"):
            sentiment, urgency, esc = analyze_issue(issue)
            case = {
                "id": f"ESC{int(datetime.utcnow().timestamp())}",
                "customer": cname,
                "issue": issue,
                "criticality": crit,
                "impact": imp,
                "sentiment": sentiment,
                "urgency": urgency,
                "escalated": int(esc),
                "date_reported": str(datetime.today().date()),
                "owner": owner,
                "status": "Open",
                "action_taken": "",
                "risk_score": predict_risk(issue),
                "spoc_email": spoc,
                "spoc_boss_email": boss,
            }
            upsert_case(case)
            st.success(f"Escalation {case['id']} logged")
