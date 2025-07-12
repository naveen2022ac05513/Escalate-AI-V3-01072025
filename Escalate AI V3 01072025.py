# ==============================================================
# EscalateAI – End‑to‑End Escalation Management System (v0.9.3)
# --------------------------------------------------------------
# • Full single‑file implementation
# • Robust fallback sentiment (no torch required)
# • SQLite persistence + daily model retraining
# • Streamlit Kanban UI with filters & inline edits
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

import os
import re
import sqlite3
import warnings
import atexit
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, List

import joblib
import pandas as pd
import requests
import streamlit as st
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

    # Create escalations table with new columns for spoc_email, spoc_boss_email, notification tracking
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
            spoc_boss_email TEXT,
            notification_count INTEGER DEFAULT 0,
            last_notification TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
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

@st.cache_resource(show_spinner=False)
def load_predictor():
    model_path = MODEL_DIR / "risk_predictor.joblib"
    return joblib.load(model_path) if model_path.exists() else None

risk_model = load_predictor()

def predict_risk(issue: str) -> float:
    if not risk_model:
        return 0.0
    return float(risk_model.predict_proba([issue])[0][1])

def analyze_issue(text: str) -> Tuple[str, str, bool]:
    if sentiment_model is None:
        sentiment = rule_based_sentiment(text)
    else:
        sentiment = "Negative" if sentiment_model(text[:512])[0]["label"].lower() == "negative" else "Positive"
    # For simplicity urgency determined by keywords
    urgency = "High" if any(k in text.lower() for k in NEG_WORDS) else "Low"
    return sentiment, urgency, sentiment == "Negative" and urgency == "High"

# ----------------------------
# ALERTING (Slack + Email)
# ----------------------------

def send_slack_alert(case: dict):
    if not ALERT_CHANNEL_ENABLED:
        return
    msg = (
        f":rotating_light: *New Escalation* {case['id']} | {case['customer']}\n"
        f"*Issue*: {case['issue'][:180]}…\n"
        f"*Urgency*: {case['urgency']} • *Sentiment*: {case['sentiment']}"
    )
    try:
        requests.post(SLACK_WEBHOOK_URL, json={"text": msg}, timeout=5)
    except requests.exceptions.RequestException as e:
        warnings.warn(f"Slack webhook failed: {e}")

import smtplib
from email.message import EmailMessage

SMTP_SERVER = os.getenv("SMTP_SERVER", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

def send_email(to_email: str, subject: str, body: str):
    if not to_email:
        return
    try:
        msg = EmailMessage()
        msg["From"] = SMTP_USERNAME
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.set_content(body)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        warnings.warn(f"Email send failed to {to_email}: {e}")

def notify_spoc(case: dict):
    # Send notification email to SPOC
    subject = f"Escalation Alert: Case {case['id']}"
    body = (
        f"Dear SPOC,\n\n"
        f"There is an escalation case assigned to you:\n\n"
        f"ID: {case['id']}\n"
        f"Customer: {case['customer']}\n"
        f"Issue: {case['issue']}\n"
        f"Urgency: {case['urgency']}\n\n"
        f"Please respond as soon as possible.\n\n"
        "Regards,\nEscalateAI System"
    )
    send_email(case.get("spoc_email"), subject, body)

def notify_spoc_boss(case: dict):
    # Escalation email to SPOC's boss after no response
    subject = f"Escalation Reminder: Case {case['id']} - Attention Required"
    body = (
        f"Dear Manager,\n\n"
        f"The escalation case assigned to your team member has not been addressed after multiple notifications:\n\n"
        f"ID: {case['id']}\n"
        f"Customer: {case['customer']}\n"
        f"Issue: {case['issue']}\n"
        f"Urgency: {case['urgency']}\n\n"
        f"Please ensure this is resolved immediately.\n\n"
        "Regards,\nEscalateAI System"
    )
    send_email(case.get("spoc_boss_email"), subject, body)

# ----------------------------
# FILE INGESTION HELPERS
# ----------------------------

def standardize_columns(df: pd.DataFrame):
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    return df

def next_escalation_id() -> str:
    # Generates a new unique escalation ID
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM escalations")
        count = c.fetchone()[0]
    return f"ESC-{count + 1:06d}"

def ingest_dataframe(df: pd.DataFrame):
    df = standardize_columns(df)
    for _, row in df.iterrows():
        sentiment, urgency, escal = analyze_issue(str(row.get("brief issue", "")))
        case = {
            "id": next_escalation_id(),
            "customer": row.get("customer", "Unknown"),
            "issue": row.get("brief issue", "Unknown"),
            "criticality": row.get("criticalness", "Unknown"),
            "impact": row.get("impact", "Unknown"),
            "sentiment": sentiment,
            "urgency": urgency,
            "escalated": int(escal),
            "date_reported": str(row.get("issue reported date", datetime.today().date())),
            "owner": row.get("owner", "Unassigned"),
            "status": row.get("status", "Open"),
            "action_taken": row.get("action taken", "None"),
            "risk_score": predict_risk(row.get("brief issue", "")),
            "spoc_email": row.get("spoc email", ""),
            "spoc_boss_email": row.get("spoc boss email", ""),
            "notification_count": 0,
            "last_notification": None,
        }
        upsert_case(case)
        if escal:
            send_slack_alert(case)
            notify_spoc(case)

# ----------------------------
# SCHEDULER – DAILY RETRAIN
# ----------------------------

def daily_retrain():
    df_all = fetch_cases()
    if not df_all.empty and "escalated" in df_all.columns:
        globals()["risk_model"] = train_predictor(df_all[["issue", "escalated"]])
        print("[Scheduler] Risk model retrained", datetime.now())

scheduler = BackgroundScheduler()
scheduler.add_job(daily_retrain, "cron", hour=2)
scheduler.start()
atexit.register(lambda: scheduler.shutdown(wait=False))

# ----------------------------
# SCHEDULER – DAILY CHECK & ESCALATION EMAILS
# ----------------------------

def escalation_check():
    df_all = fetch_cases()
    now = datetime.now()

    for _, case in df_all.iterrows():
        # Only consider open escalations with SPOC email
        if case["status"] == "Open" and case.get("spoc_email"):
            last_notif = case.get("last_notification")
            notif_count = case.get("notification_count", 0)

            last_notif_dt = datetime.strptime(last_notif, "%Y-%m-%d %H:%M:%S") if last_notif else None
            # If never notified or last notification > 24 hrs ago
            if last_notif_dt is None or (now - last_notif_dt) > timedelta(hours=24):
                if notif_count < 2:
                    # Notify SPOC again
                    notify_spoc(case)
                    # Update DB
                    updated_case = case.to_dict()
                    updated_case["notification_count"] = notif_count + 1
                    updated_case["last_notification"] = now.strftime("%Y-%m-%d %H:%M:%S")
                    upsert_case(updated_case)
                elif notif_count >= 2:
                    # Notify SPOC's boss
                    notify_spoc_boss(case)
                    # Update DB to avoid repeat notifications until status changes
                    updated_case = case.to_dict()
                    updated_case["notification_count"] = notif_count + 1
                    updated_case["last_notification"] = now.strftime("%Y-%m-%d %H:%M:%S")
                    upsert_case(updated_case)

scheduler.add_job(escalation_check, "interval", hours=1)

# ----------------------------
# ML RISK PREDICTOR (Sklearn)
# ----------------------------

MODEL_FILE = MODEL_DIR / "risk_predictor.joblib"

@st.cache_resource(show_spinner=False)
def load_predictor():
    return joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None

risk_model = load_predictor()

def train_predictor(df_lbl: pd.DataFrame):
    if df_lbl.empty:
        return None
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(df_lbl["issue"], df_lbl["escalated"].astype(int))
    joblib.dump(pipe, MODEL_FILE)
    return pipe

def predict_risk(issue: str) -> float:
    if risk_model is None:
        return 0.0
    return float(round(risk_model.predict_proba([issue])[0][1], 3))

# ----------------------------
# STREAMLIT UI
# ----------------------------

st.set_page_config(page_title="EscalateAI", layout="wide")
st.title("🚨 EscalateAI – Escalation Tracking System")

# ---------- Sidebar (Upload & Manual Entry) ----------
with st.sidebar:
    st.header("📥 Upload Escalations")
    f = st.file_uploader("Excel / CSV", type=["xlsx", "csv"])
    if f and st.button("Analyze & Log"):
        df_up = pd.read_excel(f) if f.name.endswith("xlsx") else pd.read_csv(f)
        ingest_dataframe(df_up)
        st.success("File processed & cases logged!")

    st.markdown("---")
    st.header("✏️ Manual Entry")
    with st.form("manual"):
        cname = st.text_input("Customer")
        issue = st.text_area("Issue")
        crit = st.selectbox("Criticality", ["Low", "Medium", "High"], index=1)
        imp = st.selectbox("Impact", ["Low", "Medium", "High"], index=1)
        owner = st.text_input("Owner", value="Unassigned")
        spoc_email = st.text_input("SPOC Email")
        spoc_boss_email = st.text_input("SPOC Boss Email")
        if st.form_submit_button("Log") and cname and issue:
            sentiment, urgency, escal = analyze_issue(issue)
            case = {
                "id": next_escalation_id(),
                "customer": cname,
                "issue": issue,
                "criticality": crit,
                "impact": imp,
                "sentiment": sentiment,
                "urgency": urgency,
                "escalated": int(escal),
                "date_reported": str(datetime.today().date()),
                "owner": owner,
                "status": "Open",
                "action_taken": "None",
                "risk_score": predict_risk(issue),
                "spoc_email": spoc_email,
                "spoc_boss_email": spoc_boss_email,
                "notification_count": 0,
                "last_notification": None,
            }
            upsert_case(case)
            send_slack_alert(case)
            if escal:
                notify_spoc(case)
            st.success(f"Escalation {case['id']} logged!")

# ---------- Kanban Board ----------
df = fetch_cases()
if df.empty:
    st.info("No escalations logged yet.")
else:
    st.subheader("🗂️ Escalation Kanban Board")

    # Summary counts on top
    status_counts = df['status'].value_counts()
    col1, col2, col3 = st.columns(3)
    col1.metric("Open", status_counts.get("Open", 0))
    col2.metric("In Progress", status_counts.get("In Progress", 0))
    col3.metric("Resolved", status_counts.get("Resolved", 0))

    cols = st.columns(3)
    for status, col in zip(["Open", "In Progress", "Resolved"], cols):
        with col:
            st.markdown(f"### {status}")
            for _, row in df[df.status == status].iterrows():
                with st.expander(f"{row['id']} – {row['issue'][:60]}..."):
                    st.markdown(f"**Customer:** {row['customer']}")
                    st.markdown(f"**Sentiment / Urgency:** {row['sentiment']} / {row['urgency']}")
                    st.markdown(f"**Owner:** {row['owner']}")
                    st.markdown(f"**Risk Score:** {row['risk_score']}")

                    # Editable SPOC Email
                    new_spoc_email = st.text_input(
                        "SPOC Email", value=row.get("spoc_email", ""), key=f"spoc_email_{row['id']}"
                    )
                    # Editable SPOC Boss Email
                    new_spoc_boss_email = st.text_input(
                        "SPOC Boss Email", value=row.get("spoc_boss_email", ""), key=f"spoc_boss_email_{row['id']}"
                    )

                    # Status and Action Taken edits
                    new_status = st.selectbox(
                        "Update Status",
                        ["Open", "In Progress", "Resolved"],
                        index=["Open", "In Progress", "Resolved"].index(row["status"]),
                        key=f"status_{row['id']}",
                    )
                    new_action = st.text_input(
                        "Action Taken", value=row["action_taken"], key=f"act_{row['id']}"
                    )

                    # If any field changed, update DB and rerun
                    if (new_status != row["status"] or
                        new_action != row["action_taken"] or
                        new_spoc_email != row.get("spoc_email", "") or
                        new_spoc_boss_email != row.get("spoc_boss_email", "")):

                        updated_case = row.to_dict()
                        updated_case["status"] = new_status
                        updated_case["action_taken"] = new_action
                        updated_case["spoc_email"] = new_spoc_email
                        updated_case["spoc_boss_email"] = new_spoc_boss_email

                        upsert_case(updated_case)
                        st.experimental_rerun()

    st.download_button(
        "⬇️ Download as Excel",
        data=df.to_excel(index=False, engine="openpyxl"),
        file_name="escalations_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.success("✅ EscalateAI Core Loaded.")
