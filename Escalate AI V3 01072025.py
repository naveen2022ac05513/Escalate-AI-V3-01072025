# EscalateAI v1.0.0 - Full working Streamlit app

import os, re, sqlite3, warnings, atexit, smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import joblib, pandas as pd, requests, streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Paths & config
APP_DIR = Path(__file__).parent.resolve()
MODEL_DIR = APP_DIR / "models"; MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR = APP_DIR / "data"; DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "escalateai.db"

load_dotenv()
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
ALERT_CHANNEL_ENABLED = bool(SLACK_WEBHOOK_URL)

# Sentiment detection (HF or rule-based)
try:
    from transformers import pipeline as hf_pipeline
    HAS_HF = True
except ModuleNotFoundError:
    HAS_HF = False

try:
    import torch
    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    if not (HAS_HF and HAS_TORCH):
        return None
    try:
        return hf_pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    except Exception:
        return None

sentiment_model = load_sentiment_model()

NEG_WORDS = [r"problem", r"delay", r"failure", r"unacceptable", r"risk", r"critical", r"urgent"]

def rule_based_sentiment(text: str) -> str:
    return "Negative" if any(re.search(w, text, re.I) for w in NEG_WORDS) else "Positive"

def analyze_issue(text: str) -> Tuple[str, str, bool]:
    if sentiment_model:
        label = sentiment_model(text[:512])[0]["label"].lower()
        sentiment = "Negative" if label == "negative" else "Positive"
    else:
        sentiment = rule_based_sentiment(text)
    urgency = "High" if any(k in text.lower() for k in ["urgent", "immediate", "critical"]) else "Low"
    escalated = sentiment == "Negative" and urgency == "High"
    return sentiment, urgency, escalated

# Database setup & functions

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
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
            spoc_boss_email TEXT,
            spoc_notify_count INTEGER DEFAULT 0,
            spoc_last_notified TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS notification_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            escalation_id TEXT,
            recipient_email TEXT,
            subject TEXT,
            body TEXT,
            sent_at TEXT
        )""")
        conn.commit()

def upsert_case(case: dict):
    keys = list(case.keys())
    vals = [case[k] for k in keys]
    with sqlite3.connect(DB_PATH) as conn:
        placeholders = ",".join("?" * len(keys))
        sql = f"REPLACE INTO escalations ({','.join(keys)}) VALUES ({placeholders})"
        conn.execute(sql, vals)
        conn.commit()

def fetch_cases() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM escalations ORDER BY created_at DESC", conn)
    return df

def log_notification(escalation_id, recipient, subject, body):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO notification_log (escalation_id, recipient_email, subject, body, sent_at) VALUES (?, ?, ?, ?, ?)",
            (escalation_id, recipient, subject, body, datetime.now().isoformat()),
        )
        conn.commit()

# Email sending

def send_email(to_email: str, subject: str, body: str, esc_id: str) -> bool:
    if not (SMTP_SERVER and SMTP_USER and SMTP_PASS):
        st.error("SMTP is not configured properly. Please check your .env file.")
        return False
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = f"Escalation Notification - SE Services <{SMTP_USER}>"
        msg["To"] = to_email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        log_notification(esc_id, to_email, subject, body)
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

# Risk model loading/training (basic sklearn pipeline)

MODEL_FILE = MODEL_DIR / "risk_model.joblib"

@st.cache_resource(show_spinner=False)
def load_risk_model():
    if MODEL_FILE.exists():
        return joblib.load(MODEL_FILE)
    return None

risk_model = load_risk_model()

def train_risk_model(df: pd.DataFrame):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(df["issue"], df["escalated"].astype(int))
    joblib.dump(pipe, MODEL_FILE)
    return pipe

def predict_risk(issue: str) -> float:
    if risk_model:
        return float(risk_model.predict_proba([issue])[0][1])
    return 0.0

# Initialize DB
init_db()

# Scheduler: escalate to boss if SPOC not responding after 2 notifications and 24h wait

def escalate_to_boss():
    df = fetch_cases()
    now = datetime.now()
    for _, row in df.iterrows():
        if row["spoc_notify_count"] >= 2 and row["spoc_boss_email"]:
            if row["spoc_last_notified"]:
                last_notify = datetime.fromisoformat(row["spoc_last_notified"])
                if (now - last_notify) > timedelta(hours=24):
                    subject = f"⚠️ Escalation {row['id']} unattended"
                    body = (f"Dear Manager,\n\n"
                            f"Escalation {row['id']} for customer {row['customer']} has had no SPOC response "
                            f"after 2 reminders.\n\nIssue:\n{row['issue']}\n\nPlease intervene promptly.")
                    if send_email(row["spoc_boss_email"], subject, body, row["id"]):
                        updated = row.to_dict()
                        updated["spoc_notify_count"] += 1
                        upsert_case(updated)

scheduler = BackgroundScheduler()
scheduler.add_job(escalate_to_boss, "interval", hours=1)
scheduler.start()
atexit.register(lambda: scheduler.shutdown(wait=False))

# Sidebar: Upload + Manual Entry

with st.sidebar:
    st.header("📥 Upload Escalations")
    uploaded_file = st.file_uploader("Excel or CSV file", type=["xlsx", "csv"])
    if uploaded_file and st.button("Ingest File"):
        if uploaded_file.name.endswith(".xlsx"):
            df_upload = pd.read_excel(uploaded_file)
        else:
            df_upload = pd.read_csv(uploaded_file)
        for _, row in df_upload.iterrows():
            sent, urg, esc = analyze_issue(str(row.get("brief issue", "")))
            case = {
                "id": f"ESC-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                "customer": row.get("customer", "Unknown"),
                "issue": row.get("brief issue", "Unknown"),
                "criticality": row.get("criticalness", "Medium"),
                "impact": row.get("impact", "Medium"),
                "sentiment": sent,
                "urgency": urg,
                "escalated": int(esc),
                "date_reported": str(row.get("issue reported date", datetime.today().date())),
                "owner": row.get("owner", "Unassigned"),
                "status": row.get("status", "Open"),
                "action_taken": row.get("action taken", ""),
                "risk_score": predict_risk(row.get("brief issue", "")),
                "spoc_email": row.get("spoc_email", ""),
                "spoc_boss_email": row.get("spoc_boss_email", ""),
                "spoc_notify_count": 0,
                "spoc_last_notified": None,
            }
            upsert_case(case)
        st.success("File ingested and cases logged!")

    st.markdown("---")
    st.header("✏️ Manual Entry")
    with st.form("manual_entry_form"):
        customer = st.text_input("Customer")
        issue = st.text_area("Issue")
        criticality = st.selectbox("Criticality", ["Low", "Medium", "High"], index=1)
        impact = st.selectbox("Impact", ["Low", "Medium", "High"], index=1)
        owner = st.text_input("Owner", value="Unassigned")
        spoc_email = st.text_input("SPOC Email")
        boss_email = st.text_input("Boss Email")
        submitted = st.form_submit_button("Log Escalation")
        if submitted and customer and issue:
            sent, urg, esc = analyze_issue(issue)
            case = {
                "id": f"ESC-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                "customer": customer,
                "issue": issue,
                "criticality": criticality,
                "impact": impact,
                "sentiment": sent,
                "urgency": urg,
                "escalated": int(esc),
                "date_reported": str(datetime.today().date()),
                "owner": owner,
                "status": "Open",
                "action_taken": "",
                "risk_score": predict_risk(issue),
                "spoc_email": spoc_email,
                "spoc_boss_email": boss_email,
                "spoc_notify_count": 0,
                "spoc_last_notified": None,
            }
            upsert_case(case)
            st.success("Escalation logged!")

# Main Kanban Board UI

st.title("🚨 EscalateAI – Escalation Kanban Board")

df = fetch_cases()
if df.empty:
    st.info("No escalations logged yet.")
else:
    # Summary counts
    open_count = (df.status == "Open").sum()
    inprogress_count = (df.status == "In Progress").sum()
    resolved_count = (df.status == "Resolved").sum()
    st.markdown(f"**Open:** {open_count} | **In Progress:** {inprogress_count} | **Resolved:** {resolved_count}")

    cols = st.columns(3)
    for status, col in zip(["Open", "In Progress", "Resolved"], cols):
        with col:
            st.subheader(status)
            for _, row in df[df.status == status].iterrows():
                with st.expander(f"{row['id']} – {row['issue'][:60]}..."):
                    # Editable fields
                    new_status = st.selectbox(
                        "Status",
                        options=["Open", "In Progress", "Resolved"],
                        index=["Open", "In Progress", "Resolved"].index(row["status"]),
                        key=f"status_{row['id']}"
                    )
                    new_action = st.text_area(
                        "Action Taken",
                        value=row["action_taken"] or "",
                        key=f"action_{row['id']}"
                    )
                    new_spoc = st.text_input(
                        "SPOC Email",
                        value=row["spoc_email"] or "",
                        key=f"spoc_{row['id']}"
                    )
                    new_boss = st.text_input(
                        "Boss Email",
                        value=row["spoc_boss_email"] or "",
                        key=f"boss_{row['id']}"
                    )

                    # Notify SPOC button
                    if st.button("Notify SPOC", key=f"notify_{row['id']}"):
                        if new_spoc:
                            subj = f"Escalation {row['id']} Update"
                            body = f"Dear SPOC,\n\nPlease review the following escalation:\n\n{row['issue']}\n\nThank you."
                            if send_email(new_spoc, subj, body, row["id"]):
                                updated_case = row.to_dict()
                                updated_case["spoc_notify_count"] = (updated_case.get("spoc_notify_count", 0) or 0) + 1
                                updated_case["spoc_last_notified"] = datetime.now().isoformat()
                                updated_case["spoc_email"] = new_spoc
                                updated_case["spoc_boss_email"] = new_boss
                                upsert_case(updated_case)
                                st.success("SPOC notified!")

                    # Save changes button
                    if st.button("Save Changes", key=f"save_{row['id']}"):
                        if (
                            new_status != row["status"]
                            or new_action != (row["action_taken"] or "")
                            or new_spoc != (row["spoc_email"] or "")
                            or new_boss != (row["spoc_boss_email"] or "")
                        ):
                            updated_case = row.to_dict()
                            updated_case["status"] = new_status
                            updated_case["action_taken"] = new_action
                            updated_case["spoc_email"] = new_spoc
                            updated_case["spoc_boss_email"] = new_boss
                            upsert_case(updated_case)
                            st.success("Changes saved!")
                            st.experimental_rerun()
