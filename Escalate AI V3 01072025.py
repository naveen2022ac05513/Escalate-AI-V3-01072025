import io
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

# Directories and env
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "escalateai.db"

load_dotenv()
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
ALERT_CHANNEL_ENABLED = bool(SLACK_WEBHOOK_URL)

# Sentiment model loading (transformers + torch fallback)
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

NEG_URGENCY_KWS = [
    "urgent",
    "critical",
    "immediately",
    "asap",
    "business impact",
    "high priority",
]

def analyze_issue(text: str) -> Tuple[str, str, bool]:
    if sentiment_model is None:
        sentiment = rule_based_sentiment(text)
    else:
        sentiment = "Negative" if sentiment_model(text[:512])[0]["label"].lower() == "negative" else "Positive"
    urgency = "High" if any(k in text.lower() for k in NEG_URGENCY_KWS) else "Low"
    escalated = sentiment == "Negative" and urgency == "High"
    return sentiment, urgency, escalated

# Initialize DB
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
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

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

# Risk model
MODEL_FILE = MODEL_DIR / "risk_predictor.joblib"

@st.cache_resource(show_spinner=False)
def load_predictor():
    return joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None

risk_model = load_predictor()

def predict_risk(issue: str) -> float:
    if not risk_model:
        return 0.0
    return float(risk_model.predict_proba([issue])[0][1])

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

def next_escalation_id() -> str:
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM escalations")
        count = c.fetchone()[0]
    return f"ESC-{10000 + count + 1}"

def standardize_columns(df: pd.DataFrame):
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    return df

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
        }
        upsert_case(case)
        if escal:
            send_slack_alert(case)

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

# Streamlit UI
st.set_page_config(page_title="EscalateAI", layout="wide")
st.title("🚨 EscalateAI – Escalation Tracking System")

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

                    new_status = st.selectbox(
                        "Update Status",
                        ["Open", "In Progress", "Resolved"],
                        index=["Open", "In Progress", "Resolved"].index(row["status"]),
                        key=f"status_{row['id']}"
                    )
                    new_action = st.text_input(
                        "Action Taken",
                        value=row["action_taken"],
                        key=f"act_{row['id']}"
                    )

                    if st.button(f"Save changes for {row['id']}", key=f"save_{row['id']}"):
                        if new_status != row["status"] or new_action != row["action_taken"]:
                            row["status"] = new_status
                            row["action_taken"] = new_action
                            upsert_case(row.to_dict())
                            st.experimental_rerun()


    # Fix here: Excel download via BytesIO buffer
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    data = output.getvalue()

    st.download_button(
        "⬇️ Download as Excel",
        data=data,
        file_name="escalations_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
