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
from datetime import datetime
from pathlib import Path
from typing import Tuple

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
            spoc_boss_email TEXT,
            notif_count INTEGER DEFAULT 0,
            last_notif TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ========== Additional Functions (DB, Risk, UI Placeholder) ========== 

ESCALATION_COLUMNS = {
    "id", "customer", "issue", "criticality", "impact", "sentiment", "urgency",
    "escalated", "date_reported", "owner", "status", "action_taken", "risk_score",
    "spoc_email", "spoc_boss_email", "notif_count", "last_notif", "created_at"
}

ESCALATION_COLUMNS_NO_CREATED_AT = ESCALATION_COLUMNS - {"created_at"}

def upsert_case(case: dict):
    # Filter keys to those matching DB schema except created_at (auto timestamp)
    filtered_case = {k: v for k, v in case.items() if k in ESCALATION_COLUMNS_NO_CREATED_AT}
    keys = ','.join(filtered_case.keys())
    question_marks = ','.join(['?'] * len(filtered_case))
    values = tuple(filtered_case.values())
    with sqlite3.connect(DB_PATH) as conn:
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

# ----------------------------

def analyze_issue(text: str) -> Tuple[str, str, bool]:
    # Simple urgency keywords
    NEG_URGENCY_KWS = ["urgent", "immediate", "asap", "critical", "high priority", "emergency"]
    if sentiment_model is None:
        sentiment = rule_based_sentiment(text)
    else:
        sentiment = "Negative" if sentiment_model(text[:512])[0]["label"].lower() == "negative" else "Positive"
    urgency = "High" if any(k in text.lower() for k in NEG_URGENCY_KWS) else "Low"
    escalated = sentiment == "Negative" and urgency == "High"
    return sentiment, urgency, escalated

# ----------------------------
# SLACK Alert (if enabled)
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
# File ingestion
def standardize_columns(df: pd.DataFrame):
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    return df

def next_escalation_id() -> str:
    # Generate a new unique escalation ID like ESC-000001
    df = fetch_cases()
    if df.empty:
        return "ESC-000001"
    max_id = df['id'].apply(lambda x: int(x.split("-")[1]) if x and "-" in x else 0).max()
    return f"ESC-{max_id + 1:06d}"

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
            "notif_count": 0,
            "last_notif": None,
        }
        upsert_case(case)
        if escal:
            send_slack_alert(case)

# ----------------------------
# Scheduler – placeholder for retraining etc.
def daily_retrain():
    df_all = fetch_cases()
    if not df_all.empty and "escalated" in df_all.columns:
        global risk_model
        risk_model = train_predictor(df_all[["issue", "escalated"]])
        print("[Scheduler] Risk model retrained", datetime.now())

scheduler = BackgroundScheduler()
scheduler.add_job(daily_retrain, "cron", hour=2)
scheduler.start()
atexit.register(lambda: scheduler.shutdown(wait=False))

# ----------------------------
# Streamlit UI

init_db()
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

# ---------- Kanban Board Summary ----------
df = fetch_cases()

if df.empty:
    st.info("No escalations logged yet.")
else:
    # Summary counts
    open_count = df[df.status == "Open"].shape[0]
    inprogress_count = df[df.status == "In Progress"].shape[0]
    resolved_count = df[df.status == "Resolved"].shape[0]

    st.markdown(f"### Summary: Open: {open_count} | In Progress: {inprogress_count} | Resolved: {resolved_count}")

    st.subheader("🗂️ Escalation Kanban Board")
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
                    new_status = st.selectbox("Update Status", ["Open", "In Progress", "Resolved"],
                                             index=["Open", "In Progress", "Resolved"].index(row["status"]),
                                             key=f"status_{row['id']}")
                    new_action = st.text_input("Action Taken", value=row["action_taken"], key=f"act_{row['id']}")
                    if new_status != row["status"] or new_action != row["action_taken"]:
                        # Update and save changes
                        updated_case = row.to_dict()
                        updated_case["status"] = new_status
                        updated_case["action_taken"] = new_action
                        upsert_case(updated_case)
                        st.experimental_rerun()

    st.download_button(
        "⬇️ Download as Excel",
        data=df.to_excel(index=False, engine="openpyxl"),
        file_name="escalations_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
