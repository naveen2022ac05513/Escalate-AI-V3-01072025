# ==============================================================
# EscalateAI – End‑to‑End Escalation Management System (v0.9.0)
# --------------------------------------------------------------
# One‑file distribution for quick cloning / Codespaces launch.
# Split into logical sections that can later be broken into
# separate modules (data_ingest.py, nlp_engine.py, etc.).
# Requires: streamlit, pandas, openpyxl, python‑dotenv, transformers,
#           scikit‑learn, joblib, sqlite3 (stdlib), requests, apscheduler
# --------------------------------------------------------------
# Author: Naveen Gandham • July 2025
# ==============================================================

"""Quick‑start (terminal):

pip install streamlit pandas openpyxl python-dotenv transformers scikit-learn joblib requests apscheduler

export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/…"
streamlit run escalateai_full_code.py

The code auto‑creates a SQLite DB in ./data/escalateai.db and a models
folder with a baseline LogisticRegression predictor after the first
training run.
"""

# ----------------------------
# STANDARD LIBS & THIRD PARTY
# ----------------------------
import os
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
import requests
import streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from transformers import pipeline as hf_pipeline

# ----------------------------
# ENV & GLOBAL CONSTANTS
# ----------------------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
MODEL_DIR = APP_DIR / "models"
DB_PATH = DATA_DIR / "escalateai.db"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

load_dotenv()
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
ALERT_CHANNEL_ENABLED = bool(SLACK_WEBHOOK_URL)

# ----------------------------
# DATABASE UTILITIES (SQLite)
# ----------------------------

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
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
    conn.commit()
    conn.close()


def upsert_case(case: dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    placeholders = ", ".join(["?" for _ in case])
    columns = ", ".join(case.keys())
    c.execute(
        f"REPLACE INTO escalations ({columns}) VALUES ({placeholders})",
        tuple(case.values()),
    )
    conn.commit()
    conn.close()


def fetch_cases() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    conn.close()
    return df

# ----------------------------
# NLP PIPELINE (Sentiment & Urgency)
# ----------------------------

# Lazy‑load Hugging Face model once
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return hf_pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

sentiment_model = load_sentiment_model()

NEGATIVE_URGENCY_KEYWORDS = [
    "urgent",
    "critical",
    "immediately",
    "asap",
    "business impact",
    "high priority",
]


def analyze_issue(text: str) -> Tuple[str, str, bool]:
    """Return sentiment ('Positive'/'Negative'), urgency ('High'/'Low'), and escalation flag."""
    sentiment_out = sentiment_model(text[:512])[0]
    sentiment = "Negative" if sentiment_out["label"] == "negative" else "Positive"
    text_lower = text.lower()
    urgency = "High" if any(k in text_lower for k in NEGATIVE_URGENCY_KEYWORDS) else "Low"
    escalated = sentiment == "Negative" and urgency == "High"
    return sentiment, urgency, escalated

# ----------------------------
# PREDICTIVE MODEL (Risk Score)
# ----------------------------
MODEL_FILE = MODEL_DIR / "risk_predictor.joblib"

@st.cache_resource(show_spinner=False)
def load_predictor() -> Pipeline | None:
    if MODEL_FILE.exists():
        return joblib.load(MODEL_FILE)
    return None

risk_model = load_predictor()


def train_predictor(df_labelled: pd.DataFrame):
    """Train / retrain model on labelled data (expects 'issue' text and 'escalated')"""
    if df_labelled.empty:
        return None
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipe.fit(df_labelled["issue"], df_labelled["escalated"].astype(int))
    joblib.dump(pipe, MODEL_FILE)
    return pipe


def predict_risk(issue: str) -> float:
    if risk_model is None:
        return 0.0
    proba = risk_model.predict_proba([issue])[0][1]
    return float(round(proba, 3))

# ----------------------------
# ALERTING
# ----------------------------

def send_slack_alert(case: dict):
    if not ALERT_CHANNEL_ENABLED:
        return
    message = (
        f":rotating_light: *New Escalation Logged* – {case['id']}\n"
        f"*Customer*: {case['customer']}\n"
        f"*Issue*: {case['issue'][:250]}…\n"
        f"*Urgency*: {case['urgency']} | *Sentiment*: {case['sentiment']}\n"
        f"<https://app-url-to-your-kanban|Open in EscalateAI>"
    )
    requests.post(SLACK_WEBHOOK_URL, json={"text": message})

# ----------------------------
# ID GENERATOR (Persistent)
# ----------------------------

def next_escalation_id() -> str:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM escalations")
    count = c.fetchone()[0]
    conn.close()
    return f"ESC-{10000 + count + 1}"

# ----------------------------
# FILE INGESTION HELPERS
# ----------------------------

def standardize_columns(df: pd.DataFrame):
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    return df


def ingest_dataframe(df: pd.DataFrame):
    df = standardize_columns(df)
    for _, row in df.iterrows():
        sentiment, urgency, escalated = analyze_issue(str(row.get("brief issue", "")))
        case = {
            "id": next_escalation_id(),
            "customer": row.get("customer", "Unknown"),
            "issue": row.get("brief issue", "Unknown"),
            "criticality": row.get("criticalness", "Unknown"),
            "impact": row.get("impact", "Unknown"),
            "sentiment": sentiment,
            "urgency": urgency,
            "escalated": int(escalated),
            "date_reported": str(row.get("issue reported date", datetime.today().date())),
            "owner": row.get("owner", "Unassigned"),
            "status": row.get("status", "Open"),
            "action_taken": row.get("action taken", "None"),
            "risk_score": predict_risk(row.get("brief issue", "")),
        }
        upsert_case(case)
        if escalated:
            send_slack_alert(case)

# ----------------------------
# CONTINUOUS LEARNING (SCHEDULER)
# ----------------------------

def scheduled_daily_retrain():
    df_all = fetch_cases()
    if "escalated" in df_all.columns:
        model = train_predictor(df_all[["issue", "escalated"]])
        global risk_model
        risk_model = model
        print("[Scheduler] Risk model retrained at", datetime.now())

scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_daily_retrain, "cron", hour=2, minute=0)
scheduler.start()

# ----------------------------
# STREAMLIT UI
# ----------------------------
init_db()
st.set_page_config(page_title="EscalateAI – Escalation Tracking", layout="wide")

st.title("🚨 EscalateAI – Escalation Tracking System")

# ---------- Sidebar: Upload & Manual Entry ----------
with st.sidebar:
    st.header("📥 Upload Escalations (Excel/CSV)")
    file = st.file_uploader("Choose a file", type=["xlsx", "csv"])
    if file:
        if file.name.endswith(".csv"):
            df_uploaded = pd.read_csv(file)
        else:
            df_uploaded = pd.read_excel(file)
        st.write("Rows detected:", len(df_uploaded))
        if st.button("Analyze & Log"):
            ingest_dataframe(df_uploaded)
            st.success("File processed and cases logged!")

    st.markdown("---")
    st.header("✏️ Manual Entry")
    with st.form("manual_entry_form"):
        col1, col2 = st.columns(2)
        with col1:
            customer_name = st.text_input("Customer Name")
            criticality = st.selectbox("Criticality", ["Low", "Medium", "High"], index=1)
            impact = st.selectbox("Impact", ["Low", "Medium", "High"], index=1)
            owner = st.text_input("Owner", value="Unassigned")
        with col2:
            date_reported = st.date_input("Date Reported", value=datetime.today())
            status = st.selectbox("Status", ["Open", "In Progress", "Resolved"], index=0)
        issue = st.text_area("Issue")
        submitted = st.form_submit_button("Log Escalation")
        if submitted and customer_name and issue:
            sentiment, urgency, escalated = analyze_issue(issue)
            case = {
                "id": next_escalation_id(),
                "customer": customer_name,
                "issue": issue,
                "criticality": criticality,
                "impact": impact,
                "sentiment": sentiment,
                "urgency": urgency,
                "escalated": int(escalated),
                "date_reported": str(date_reported),
                "owner": owner,
                "status": status,
                "action_taken": "None",
                "risk_score": predict_risk(issue),
            }
            upsert_case(case)
            if escalated:
                send_slack_alert(case)
            st.success(f"Escalation {case['id']} logged!")

# ---------- Kanban Board ----------

df_cases = fetch_cases()
if df_cases.empty:
    st.info("No escalations logged yet.")
else:
    # Filter options
    st.subheader("🔍 Filters")
    cols = st.columns(4)
    with cols[0]:
        f_customer = st.multiselect("Customer", options=sorted(df_cases.customer.unique()))
    with cols[1]:
        f_status = st.multiselect("Status", options=["Open", "In Progress", "Resolved"])
    with cols[2]:
        f_owner = st.multiselect("Owner", options=sorted(df_cases.owner.unique()))
    with cols[3]:
        show_risk = st.checkbox("Show Risk Score", value=False)

    df_view = df_cases.copy()
    if f_customer:
        df_view = df_view[df_view.customer.isin(f_customer)]
    if f_status:
        df_view = df_view[df_view.status.isin(f_status)]
    if f_owner:
        df_view = df_view[df_view.owner.isin(f_owner)]

    # Status counts for header
    counts = df_view.status.value_counts().reindex(["Open", "In Progress", "Resolved"], fill_value=0)
    st.subheader(
        f"🗂️ Escalation Kanban Board (Open: {counts['Open']} | In Progress: {counts['In Progress']} | Resolved: {counts['Resolved']})"
    )

    # Three columns
    col_open, col_prog, col_res = st.columns(3)
    buckets = {"Open": col_open, "In Progress": col_prog, "Resolved": col_res}

    for _status, col in buckets.items():
        with col:
            subset = df_view[df_view.status == _status]
            for _, row in subset.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{row['id']} – {row['issue'][:60]}…**")
                    st.write(f"**Customer**: {row['customer']}")
                    st.write(f"**Sentiment / Urgency**: {row['sentiment']} / {row['urgency']}")
                    st.write(f"**Criticality / Impact**: {row['criticality']} / {row['impact']}")
                    st.write(f"**Owner**: {row['owner']}")
                    if show_risk:
                        st.write(f"**Risk Score**: {row['risk_score']}")
                    # Inline editing controls
                    new_status = st.selectbox(
                        "Update Status",
                        ["Open", "In Progress", "Resolved"],
                        index=["Open", "In Progress", "Resolved"].index(row["status"]),
                        key=f"status_{row['id']}",
                    )
                    new_action = st.text_input(
                        "Action Taken",
                        value=row["action_taken"],
                        key=f"action_{row['id']}",
                    )
                    if new_status != row["status"] or new_action != row["action_taken"]:
                        row["status"] = new_status
                        row["action_taken"] = new_action
                        upsert_case(row.to_dict())
                        st.experimental_rerun()

# ---------- Download Button ----------
if not df_cases.empty:
    st.markdown("---")
    st.download_button(
        label="Download Escalations (Excel)",
        data=df_cases.to_excel(index=False, engine="openpyxl"),
        file_name="escalations_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# ----------------------------
# FOOTER
# ----------------------------
with st.expander("ℹ️ About EscalateAI"):
    st.write(
        "EscalateAI is an open‑source escalation management tool designed for \n"
        "teams that need real‑time visibility into customer issues, predictive \n"
        "risk analytics, and automated alerting. Fork the repo and contribute!"
    )

# Keep scheduler alive inside Streamlit (hack for Cloud Run / Codespaces)
import atexit
atexit.register(lambda: scheduler.shutdown(wait=False))

