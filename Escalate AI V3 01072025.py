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

import os, re, sqlite3, warnings, atexit
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, List

import joblib, pandas as pd, requests, streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import smtplib
from email.message import EmailMessage

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "escalateai.db"

load_dotenv()
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
ALERT_CHANNEL_ENABLED = bool(SLACK_WEBHOOK_URL)

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")

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

NEG_URGENCY_KWS = ["urgent", "immediately", "asap", "critical", "high priority", "now"]

def analyze_issue(text: str) -> Tuple[str, str, bool]:
    if sentiment_model is None:
        sentiment = rule_based_sentiment(text)
    else:
        sentiment = "Negative" if sentiment_model(text[:512])[0]["label"].lower() == "negative" else "Positive"
    urgency = "High" if any(k in text.lower() for k in NEG_URGENCY_KWS) else "Low"
    return sentiment, urgency, sentiment == "Negative" and urgency == "High"

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
            spoc_boss_email TEXT,
            notif_count INTEGER DEFAULT 0,
            last_notif TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

def next_escalation_id() -> str:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM escalations ORDER BY created_at DESC LIMIT 1")
    last_id = c.fetchone()
    conn.close()
    if last_id and last_id[0].startswith("CESI-"):
        num = int(last_id[0].split("-")[1])
        return f"CESI-{num + 1:06d}"
    else:
        return "CESI-000001"

def upsert_case(case: dict):
    with sqlite3.connect(DB_PATH) as conn:
        keys = ','.join(case.keys())
        question_marks = ','.join(['?'] * len(case))
        values = tuple(case.values())
        conn.execute(f"REPLACE INTO escalations ({keys}) VALUES ({question_marks})", values)
        conn.commit()

def fetch_cases() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM escalations ORDER BY date_reported DESC", conn)

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
# ALERTING (Slack)
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

# ----------------------------
# EMAIL SENDING (SMTP)
# ----------------------------

def send_email(to_email: str, subject: str, body: str):
    if not SMTP_USER or not SMTP_PASS:
        st.warning("SMTP credentials not set. Email not sent.")
        return
    msg = EmailMessage()
    msg["From"] = SMTP_USER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
            smtp.starttls()
            smtp.login(SMTP_USER, SMTP_PASS)
            smtp.send_message(msg)
    except Exception as e:
        st.error(f"Error sending email: {e}")

# ----------------------------
# FILE INGESTION HELPERS
# ----------------------------

def standardize_columns(df: pd.DataFrame):
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    return df

def ingest_dataframe(df: pd.DataFrame):
    df = standardize_columns(df)
    for _, row in df.iterrows():
        sentiment, urgency, escal = analyze_issue(str(row.get("brief issue", "")))
        case = {
            "id":             next_escalation_id(),
            "customer":       row.get("customer", "Unknown"),
            "issue":          row.get("brief issue", "Unknown"),
            "criticality":    row.get("criticalness", "Unknown"),
            "impact":         row.get("impact", "Unknown"),
            "sentiment":      sentiment,
            "urgency":        urgency,
            "escalated":      int(escal),
            "date_reported":  str(row.get("issue reported date", datetime.today().date())),
            "owner":          row.get("owner", "Unassigned"),
            "status":         row.get("status", "Open"),
            "action_taken":   row.get("action taken", "None"),
            "risk_score":     predict_risk(row.get("brief issue", "")),
            "spoc_email":     row.get("spoc email", ""),
            "spoc_boss_email":row.get("spoc boss email", ""),
            "notif_count":    0,
            "last_notif":     None,
        }
        upsert_case(case)
        if escal:
            send_slack_alert(case)

# ----------------------------
# SCHEDULER – DAILY RETRAIN & ESCALATION CHECK
# ----------------------------

def daily_retrain():
    df_all = fetch_cases()
    if not df_all.empty and "escalated" in df_all.columns:
        global risk_model
        risk_model = train_predictor(df_all[["issue", "escalated"]])
        print("[Scheduler] Risk model retrained", datetime.now())

def check_and_send_escalations():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now()
    c.execute("SELECT * FROM escalations WHERE escalated=1")
    rows = c.fetchall()
    col_names = [desc[0] for desc in c.description]
    for row in rows:
        case = dict(zip(col_names, row))
        last_notif_dt = datetime.fromisoformat(case["last_notif"]) if case["last_notif"] else None
        notif_count = case["notif_count"] or 0
        spoc_email = case.get("spoc_email", "")
        spoc_boss_email = case.get("spoc_boss_email", "")
        # If no SPOC email, skip
        if not spoc_email or "@" not in spoc_email:
            continue
        # If no notifications sent yet or last notification older than 24h
        if notif_count < 2 and (not last_notif_dt or now - last_notif_dt > timedelta(hours=24)):
            # Send notification to SPOC
            try:
                send_email(
                    spoc_email,
                    f"Reminder: Escalation {case['id']} Requires Your Attention",
                    f"Dear SPOC,\n\nThis is a reminder for escalation {case['id']} regarding:\n\n{case['issue']}\n\nPlease take necessary action.\n\nRegards,\nEscalateAI"
                )
                c.execute("""
                    UPDATE escalations
                    SET notif_count = notif_count + 1,
                        last_notif = ?
                    WHERE id = ?
                """, (now.isoformat(), case['id']))
                conn.commit()
            except Exception as e:
                warnings.warn(f"Email notification failed: {e}")
        # After 2 notifications, escalate to boss if no response (status still Open or In Progress)
        elif notif_count >= 2 and case['status'] in ("Open", "In Progress") and spoc_boss_email and "@" in spoc_boss_email:
            try:
                send_email(
                    spoc_boss_email,
                    f"Escalation Alert: {case['id']} Needs Your Attention",
                    f"Dear Manager,\n\nThe escalation {case['id']} has not been addressed by the SPOC.\n\nIssue:\n{case['issue']}\n\nPlease intervene.\n\nRegards,\nEscalateAI"
                )
                # Mark escalation escalated to boss by setting notif_count to a high number (e.g. 99)
                c.execute("""
                    UPDATE escalations
                    SET notif_count = 99,
                        last_notif = ?
                    WHERE id = ?
                """, (now.isoformat(), case['id']))
                conn.commit()
            except Exception as e:
                warnings.warn(f"Email escalation to boss failed: {e}")

    conn.close()

scheduler = BackgroundScheduler()
scheduler.add_job(daily_retrain, "cron", hour=2)
scheduler.add_job(check_and_send_escalations, "interval", hours=1)
scheduler.start()
atexit.register(lambda: scheduler.shutdown(wait=False))

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
                "notif_count": 0,
                "last_notif": None,
            }
            upsert_case(case)
            if escal:
                send_slack_alert(case)
            st.success(f"Escalation {case['id']} logged!")

# ---------- Kanban Board ----------

df = fetch_cases()
if df.empty:
    st.info("No escalations logged yet.")
else:
    st.subheader("🗂️ Escalation Kanban Board")

    # Summary counts
    open_count = (df.status == "Open").sum()
    inprogress_count = (df.status == "In Progress").sum()
    resolved_count = (df.status == "Resolved").sum()

    st.markdown(f"### Summary: Open: {open_count} | In Progress: {inprogress_count} | Resolved: {resolved_count}")

    cols = st.columns(3)
    for status, col in zip(["Open", "In Progress", "Resolved"], cols):
        with col:
            st.markdown(f"### {status}")
            for _, row in df[df.status == status].iterrows():
                with st.expander(f"{row['id']} – {row['issue'][:60]}..."):
                    st.markdown(f"**Customer:** {row['customer']}")
                    st.markdown(f"**Sentiment / Urgency:** {row['sentiment']} / {row['urgency']}")
                    st.markdown(f"**Owner:** {row['owner']}")
                    st.markdown(f"**SPOC Email:** {row.get('spoc_email','')}")
                    st.markdown(f"**SPOC Boss Email:** {row.get('spoc_boss_email','')}")
                    st.markdown(f"**Risk Score:** {row['risk_score']}")

                    new_status = st.selectbox("Update Status", ["Open", "In Progress", "Resolved"],
                        index=["Open", "In Progress", "Resolved"].index(row["status"]),
                        key=f"status_{row['id']}")
                    new_action = st.text_input("Action Taken", value=row["action_taken"], key=f"act_{row['id']}")
                    spoc_email_inp = st.text_input("SPOC Email", value=row.get("spoc_email",""), key=f"spoc_email_{row['id']}")
                    spoc_boss_email_inp = st.text_input("SPOC Boss Email", value=row.get("spoc_boss_email",""), key=f"spoc_boss_email_{row['id']}")

                    if (new_status != row["status"] or new_action != row["action_taken"] or
                        spoc_email_inp != row.get("spoc_email","") or spoc_boss_email_inp != row.get("spoc_boss_email","")):
                        row["status"] = new_status
                        row["action_taken"] = new_action
                        row["spoc_email"] = spoc_email_inp
                        row["spoc_boss_email"] = spoc_boss_email_inp
                        upsert_case(row.to_dict())
                        st.experimental_rerun()

                    if st.button("Notify SPOC", key=f"notify_{row['id']}"):
                        spoc_email_val = row.get("spoc_email", "")
                        if spoc_email_val and "@" in spoc_email_val:
                            try:
                                send_email(
                                    spoc_email_val,
                                    f"Escalation {row['id']} Status Update",
                                    f"Dear SPOC,\n\nThe escalation with ID {row['id']} and issue:\n\n{row['issue']}\n\n"
                                    f"Has been updated to status: {new_status}.\n\nRegards,\nEscalateAI"
                                )
                                with sqlite3.connect(DB_PATH) as conn:
                                    c = conn.cursor()
                                    c.execute("""
                                        UPDATE escalations
                                        SET notif_count = notif_count + 1,
                                            last_notif = ?
                                        WHERE id = ?
                                    """, (datetime.now().isoformat(), row['id']))
                                    conn.commit()
                                st.success(f"Notification sent to {spoc_email_val}")
                            except Exception as e:
                                st.error(f"Failed to send email: {e}")
                        else:
                            st.warning("Please enter a valid SPOC email address.")

    # Export Button
    st.download_button(
        "⬇️ Download as Excel",
        data=df.to_excel(index=False, engine="openpyxl"),
        file_name="escalations_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
