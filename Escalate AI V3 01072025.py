# EscalateAI - Complete minimal working version (with email notification)
# Author: ChatGPT (adapted for you)

import os
import re
import sqlite3
import warnings
import atexit
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
import streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load environment variables
load_dotenv()
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")

# Paths
APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "escalateai.db"

# Initialize DB
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
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
            spoc_notify_count INTEGER DEFAULT 0,
            spoc_last_notified TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.execute("""
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

init_db()

# Helper DB functions
def upsert_case(case: dict):
    keys = ", ".join(case.keys())
    placeholders = ", ".join("?" for _ in case)
    values = tuple(case.values())
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"REPLACE INTO escalations ({keys}) VALUES ({placeholders})", values)
        conn.commit()

def fetch_cases() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM escalations ORDER BY created_at DESC", conn)

# Simple sentiment analysis stub
NEG_WORDS = ["problem", "delay", "failure", "unacceptable", "risk", "critical", "urgent"]
def rule_based_sentiment(text: str) -> str:
    return "Negative" if any(re.search(w, text, re.I) for w in NEG_WORDS) else "Positive"

def analyze_issue(text: str) -> Tuple[str,str,bool]:
    sentiment = rule_based_sentiment(text)
    urgency = "High" if any(k in text.lower() for k in ["urgent","immediate","critical"]) else "Low"
    escalated = sentiment=="Negative" and urgency=="High"
    return sentiment, urgency, escalated

# Dummy risk predictor (replace with your own ML model if you want)
def predict_risk(issue:str) -> float:
    # Example: simple heuristic, increase risk if "critical" found
    return 0.9 if "critical" in issue.lower() else 0.1

# Email sending function
def send_email(to_email:str, subject:str, body:str, esc_id:str):
    if not (SMTP_SERVER and SMTP_USER and SMTP_PASS):
        st.error("SMTP not configured. Please check your .env file.")
        return False
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = f"Escalation Notification - SE Services <{SMTP_USER}>"
        msg["To"] = to_email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        # Log notification
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO notification_log (escalation_id, recipient_email, subject, body, sent_at) VALUES (?,?,?,?,?)",
                (esc_id, to_email, subject, body, datetime.now().isoformat())
            )
            conn.commit()
        return True
    except Exception as e:
        st.error(f"Email sending failed: {e}")
        return False

# Scheduler: escalate to boss if no SPOC response after 2 notifications and 24 hours
def escalate_to_boss():
    df = fetch_cases()
    for _, row in df.iterrows():
        if row["spoc_notify_count"] >= 2 and row["spoc_boss_email"]:
            last_notified = row["spoc_last_notified"]
            if last_notified:
                last_dt = datetime.fromisoformat(last_notified)
                if datetime.now() - last_dt > timedelta(hours=24):
                    subject = f"⚠️ Escalation {row['id']} Unattended"
                    body = f"Dear Manager,\n\nEscalation {row['id']} for {row['customer']} has had no response from SPOC after 2 reminders.\nIssue: {row['issue']}\nPlease intervene."
                    if send_email(row["spoc_boss_email"], subject, body, row["id"]):
                        updated = dict(row)
                        updated["spoc_notify_count"] += 1  # Increase to avoid repeat emails
                        upsert_case(updated)

scheduler = BackgroundScheduler()
scheduler.add_job(escalate_to_boss, "interval", minutes=60)
scheduler.start()
atexit.register(lambda: scheduler.shutdown(wait=False))

# Streamlit UI
st.set_page_config(page_title="EscalateAI", layout="wide")
st.title("🚨 EscalateAI – Escalation Tracking System")

# Sidebar Upload & Manual Entry
with st.sidebar:
    st.header("📥 Upload Escalations")
    uploaded_file = st.file_uploader("Upload Excel / CSV", type=["xlsx", "csv"])
    if uploaded_file and st.button("Ingest File"):
        if uploaded_file.name.endswith("xlsx"):
            df_upload = pd.read_excel(uploaded_file)
        else:
            df_upload = pd.read_csv(uploaded_file)
        for _, row in df_upload.iterrows():
            issue_text = str(row.get("brief issue", ""))
            sent, urg, esc = analyze_issue(issue_text)
            case = {
                "id": f"ESC-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                "customer": row.get("customer", "Unknown"),
                "issue": issue_text,
                "criticality": row.get("criticalness", "Medium"),
                "impact": row.get("impact", "Medium"),
                "sentiment": sent,
                "urgency": urg,
                "escalated": int(esc),
                "date_reported": str(row.get("issue reported date", datetime.today().date())),
                "owner": row.get("owner", "Unassigned"),
                "status": row.get("status", "Open"),
                "action_taken": row.get("action taken", ""),
                "risk_score": predict_risk(issue_text),
                "spoc_email": row.get("spoc_email", ""),
                "spoc_boss_email": row.get("spoc_boss_email", ""),
                "spoc_notify_count": 0,
                "spoc_last_notified": None
            }
            upsert_case(case)
        st.success("File ingested successfully!")

    st.markdown("---")
    st.header("✏️ Manual Entry")
    with st.form("manual_entry_form"):
        cust = st.text_input("Customer")
        issue = st.text_area("Issue")
        crit = st.selectbox("Criticality", ["Low", "Medium", "High"], index=1)
        imp = st.selectbox("Impact", ["Low", "Medium", "High"], index=1)
        owner = st.text_input("Owner", value="Unassigned")
        spoc_email = st.text_input("SPOC Email")
        spoc_boss_email = st.text_input("Boss Email")
        submitted = st.form_submit_button("Log Escalation")
        if submitted:
            if cust.strip() == "" or issue.strip() == "":
                st.warning("Customer and Issue fields are required.")
            else:
                sent, urg, esc = analyze_issue(issue)
                case = {
                    "id": f"ESC-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                    "customer": cust,
                    "issue": issue,
                    "criticality": crit,
                    "impact": imp,
                    "sentiment": sent,
                    "urgency": urg,
                    "escalated": int(esc),
                    "date_reported": str(datetime.today().date()),
                    "owner": owner,
                    "status": "Open",
                    "action_taken": "",
                    "risk_score": predict_risk(issue),
                    "spoc_email": spoc_email,
                    "spoc_boss_email": spoc_boss_email,
                    "spoc_notify_count": 0,
                    "spoc_last_notified": None
                }
                upsert_case(case)
                st.success(f"Escalation logged with ID {case['id']}!")

# Fetch cases for Kanban board
df_cases = fetch_cases()
if df_cases.empty:
    st.info("No escalations logged yet.")
else:
    st.markdown(
        f"**Summary:** Open: {(df_cases.status == 'Open').sum()} | In Progress: {(df_cases.status == 'In Progress').sum()} | Resolved: {(df_cases.status == 'Resolved').sum()}"
    )
    cols = st.columns(3)
    for status, col in zip(["Open", "In Progress", "Resolved"], cols):
        with col:
            st.subheader(status)
            filtered = df_cases[df_cases.status == status]
            for idx, row in filtered.iterrows():
                with st.expander(f"{row['id']} – {row['issue'][:60]}"):
                    new_status = st.selectbox("Status", ["Open", "In Progress", "Resolved"],
                                             index=["Open", "In Progress", "Resolved"].index(row["status"]),
                                             key=f"status_{row['id']}")
                    new_action = st.text_area("Action Taken", value=row["action_taken"], key=f"action_{row['id']}")
                    new_spoc = st.text_input("SPOC Email", value=row["spoc_email"] or "", key=f"spoc_{row['id']}")
                    new_boss = st.text_input("Boss Email", value=row["spoc_boss_email"] or "", key=f"boss_{row['id']}")

                    notify_key = f"notify_{row['id']}"
                    if st.button("Notify SPOC", key=notify_key):
                        if not new_spoc:
                            st.warning("Please enter a SPOC email before sending notification.")
                        else:
                            subj = f"Escalation Notification: {row['id']}"
                            body = f"Dear SPOC,\n\nPlease review the following escalation:\n\n{row['issue']}\n\nThank you."
                            if send_email(new_spoc, subj, body, row['id']):
                                updated = dict(row)
                                updated["spoc_notify_count"] = (row["spoc_notify_count"] or 0) + 1
                                updated["spoc_last_notified"] = datetime.now().isoformat()
                                updated["spoc_email"] = new_spoc
                                updated["spoc_boss_email"] = new_boss
                                upsert_case(updated)
                                st.success("Notification sent to SPOC.")

                    # Detect if anything changed and update DB
                    if (new_status != row["status"] or
                        new_action != row["action_taken"] or
                        new_spoc != row["spoc_email"] or
                        new_boss != row["spoc_boss_email"]):
                        updated = dict(row)
                        updated["status"] = new_status
                        updated["action_taken"] = new_action
                        updated["spoc_email"] = new_spoc
                        updated["spoc_boss_email"] = new_boss
                        upsert_case(updated)
                        st.experimental_rerun()
