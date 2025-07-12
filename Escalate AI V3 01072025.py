# ==============================================================
# EscalateAI – End‑to‑End Escalation Management System (v0.9.4)
# --------------------------------------------------------------
# Adds SPOC emails + escalation email + reminders
# --------------------------------------------------------------
# Author: Naveen Gandham • July 2025
# ==============================================================

import os, re, sqlite3, warnings, atexit, smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import joblib, pandas as pd, requests, streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

load_dotenv()

# SMTP email config
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USERNAME)

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "escalateai.db"

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
ALERT_CHANNEL_ENABLED = bool(SLACK_WEBHOOK_URL)

# --------------------------
# DB initialization and schema update for new columns
# --------------------------
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
            notif_count INTEGER DEFAULT 0,
            last_notif TEXT,
            spoc_email TEXT,
            spoc_boss_email TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    # Check and add columns if missing (for backward compatibility)
    existing_cols = [row[1] for row in c.execute("PRAGMA table_info(escalations)").fetchall()]
    for col, ctype in [("notif_count", "INTEGER DEFAULT 0"),
                       ("last_notif", "TEXT"),
                       ("spoc_email", "TEXT"),
                       ("spoc_boss_email", "TEXT")]:
        if col not in existing_cols:
            c.execute(f"ALTER TABLE escalations ADD COLUMN {col} {ctype}")
            conn.commit()
    conn.close()

init_db()

# --------------------------
# Helper: Send Email function
# --------------------------
def send_email(to_email: str, subject: str, body: str):
    if not to_email:
        return
    try:
        msg = EmailMessage()
        msg["From"] = EMAIL_FROM
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.set_content(body)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        print(f"[Email] Sent to {to_email} - {subject}")
    except Exception as e:
        warnings.warn(f"Failed to send email to {to_email}: {e}")

# --------------------------
# Slack alert (optional)
# --------------------------
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

# --------------------------
# Notification logic
# --------------------------
def notify_spoc(case: dict):
    if not case.get("spoc_email"):
        return
    subject = f"Escalation Alert: {case['id']} for {case['customer']}"
    body = (
        f"Dear SPOC,\n\n"
        f"A new escalation has been logged:\n\n"
        f"ID: {case['id']}\n"
        f"Customer: {case['customer']}\n"
        f"Issue: {case['issue']}\n"
        f"Urgency: {case['urgency']}\n"
        f"Sentiment: {case['sentiment']}\n\n"
        f"Please respond to this issue as soon as possible.\n\n"
        f"Thanks,\nEscalateAI"
    )
    send_email(case["spoc_email"], subject, body)

def notify_spoc_boss(case: dict):
    if not case.get("spoc_boss_email"):
        return
    subject = f"Escalation Escalated: {case['id']} for {case['customer']} - No SPOC Response"
    body = (
        f"Dear SPOC Boss,\n\n"
        f"The escalation below has not been responded to by the SPOC after multiple reminders:\n\n"
        f"ID: {case['id']}\n"
        f"Customer: {case['customer']}\n"
        f"Issue: {case['issue']}\n"
        f"Urgency: {case['urgency']}\n"
        f"Sentiment: {case['sentiment']}\n\n"
        f"Please take urgent action.\n\n"
        f"Thanks,\nEscalateAI"
    )
    send_email(case["spoc_boss_email"], subject, body)

# --------------------------
# DB CRUD functions
# --------------------------
def upsert_case(case: dict):
    keys = ','.join(case.keys())
    question_marks = ','.join(['?'] * len(case))
    values = tuple(case.values())
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"REPLACE INTO escalations ({keys}) VALUES ({question_marks})", values)
        conn.commit()

def fetch_cases() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM escalations ORDER BY created_at DESC", conn)

# --------------------------
# Scheduled job for reminders and escalation
# --------------------------
def check_notifications():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM escalations WHERE escalated=1 AND status != 'Resolved'", conn)
        now = datetime.now()
        for _, row in df.iterrows():
            last_notif = row["last_notif"]
            notif_count = row["notif_count"] or 0
            spoc_email = row["spoc_email"]
            spoc_boss_email = row["spoc_boss_email"]

            if last_notif is None:
                continue

            last_notif_dt = datetime.fromisoformat(last_notif)
            hours_passed = (now - last_notif_dt).total_seconds() / 3600

            if notif_count < 2 and hours_passed >= 24:
                notify_spoc(row)
                notif_count += 1
                conn.execute("UPDATE escalations SET notif_count = ?, last_notif = ? WHERE id = ?",
                             (notif_count, now.isoformat(), row["id"]))
                conn.commit()
            elif notif_count >= 2 and hours_passed >= 24:
                notify_spoc_boss(row)
                # Increase notif_count to avoid repeated boss emails
                conn.execute("UPDATE escalations SET notif_count = ?, last_notif = ? WHERE id = ?",
                             (notif_count + 1, now.isoformat(), row["id"]))
                conn.commit()

scheduler = BackgroundScheduler()
scheduler.add_job(check_notifications, "interval", hours=1)
scheduler.start()
atexit.register(lambda: scheduler.shutdown(wait=False))

# --------------------------
# Your existing ML models & sentiment & UI code
# (Include your previous code for load_sentiment_model, risk_model, train_predictor, predict_risk, etc.)
# --------------------------
# For brevity, add your previous code here...

# --------------------------
# Streamlit UI updates for SPOC email and boss email
# --------------------------
st.set_page_config(page_title="EscalateAI", layout="wide")
st.title("🚨 EscalateAI – Escalation Tracking System")

with st.sidebar:
    st.header("📥 Upload Escalations")
    f = st.file_uploader("Excel / CSV", type=["xlsx", "csv"])
    if f and st.button("Analyze & Log"):
        df_up = pd.read_excel(f) if f.name.endswith("xlsx") else pd.read_csv(f)
        # Add default empty spoc emails if missing columns
        if "spoc_email" not in df_up.columns:
            df_up["spoc_email"] = ""
        if "spoc_boss_email" not in df_up.columns:
            df_up["spoc_boss_email"] = ""
        # Add your ingestion function here that includes spoc emails (adapt your existing ingest_dataframe)
        # e.g. ingest_dataframe(df_up)
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
            sentiment, urgency, escal = analyze_issue(issue)  # Use your sentiment analysis function here
            case = {
                "id": next_escalation_id(),  # Your function to generate unique IDs
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
                "risk_score": predict_risk(issue),  # Your risk prediction function
                "notif_count": 0,
                "last_notif": None,
                "spoc_email": spoc_email,
                "spoc_boss_email": spoc_boss_email,
            }
            upsert_case(case)
            if escal:
                notify_spoc(case)
            st.success(f"Escalation {case['id']} logged!")

# ---------- Kanban Board with Summary ----------
df = fetch_cases()
if df.empty:
    st.info("No escalations logged yet.")
else:
    st.markdown(
        f"### Summary: Open: {sum(df.status=='Open')} | In Progress: {sum(df.status=='In Progress')} | Resolved: {sum(df.status=='Resolved')}"
    )
    cols = st.columns(3)
    for status, col in zip(["Open", "In Progress", "Resolved"], cols):
        with col:
            st.markdown(f"### {status}")
            for _, row in df[df.status == status].iterrows():
                with st.expander(f"{row['id']} – {row['issue'][:60]}..."):
                    st.markdown(f"**Customer:** {row['customer']}")
                    st.markdown(f"**Sentiment / Urgency:** {row['sentiment']} / {row['urgency']}")
                    st.markdown(f"**Owner:** {row['owner']}")
                    st.markdown(f"**SPOC Email:** {row['spoc_email']}")
                    st.markdown(f"**Risk Score:** {row['risk_score']}")
                    new_status = st.selectbox(
                        "Update Status",
                        ["Open", "In Progress", "Resolved"],
                        index=["Open", "In Progress", "Resolved"].index(row["status"]),
                        key=f"status_{row['id']}",
                    )
                    new_action = st.text_input(
                        "Action Taken", value=row["action_taken"], key=f"act_{row['id']}"
                    )
                    if new_status != row["status"] or new_action != row["action_taken"]:
                        row_dict = row.to_dict()
                        row_dict["status"] = new_status
                        row_dict["action_taken"] = new_action
                        upsert_case(row_dict)
                        st.experimental_rerun()

    st.download_button(
        "⬇️ Download as Excel",
        data=df.to_excel(index=False, engine="openpyxl"),
        file_name="escalations_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
