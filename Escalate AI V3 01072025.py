# ==============================================================
# EscalateAI – End‑to‑End Escalation Management System (v1.0.0)
# --------------------------------------------------------------
# • One‑file Streamlit app
# • SQLite persistence
# • Sentiment (HF or rule‑based) + risk ML model
# • Sidebar: Excel/CSV upload  & manual entry
# • Kanban board with inline edits
# • SPOC email notification + boss escalation & history
# --------------------------------------------------------------
# Author: Naveen Gandham • July 2025
# ==============================================================

"""Quick‑start (terminal):

pip install streamlit pandas openpyxl python-dotenv transformers scikit-learn joblib requests apscheduler
# Optional (better accuracy – only if PyTorch wheel is available):
pip install torch --index-url https://download.pytorch.org/whl/cpu

export SMTP_SERVER="smtp.yourmail.com"
export SMTP_PORT="587"
export SMTP_USER="your.address@example.com"
export SMTP_PASS="your‑smtp‑password"
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/…"

streamlit run escalateai_app.py
"""

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

# ========== Paths & ENV ==========
APP_DIR   = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"; MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR  = APP_DIR / "data"  ; DATA_DIR.mkdir(exist_ok=True)
DB_PATH   = DATA_DIR / "escalateai.db"

load_dotenv()
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT   = int(os.getenv("SMTP_PORT", 587))
SMTP_USER   = os.getenv("SMTP_USER")  # mailbox/login
SMTP_PASS   = os.getenv("SMTP_PASS")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
ALERT_CHANNEL_ENABLED = bool(SLACK_WEBHOOK_URL)

# ========== Sentiment ==========
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

def analyze_issue(text: str) -> Tuple[str,str,bool]:
    sentiment = ("Negative" if sentiment_model(text[:512])[0]["label"].lower()=="negative"
                 else "Positive") if sentiment_model else rule_based_sentiment(text)
    urgency   = "High" if any(k in text.lower() for k in ["urgent","immediate","critical"]) else "Low"
    escalated = sentiment=="Negative" and urgency=="High"
    return sentiment, urgency, escalated

# ========== DB ==========

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
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
        cur.execute("""
        CREATE TABLE IF NOT EXISTS notification_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            escalation_id TEXT,
            recipient_email TEXT,
            subject TEXT,
            body TEXT,
            sent_at TEXT
        )""")
        conn.commit()

init_db()

ESC_COLS = [c[1] for c in sqlite3.connect(DB_PATH).execute("PRAGMA table_info(escalations)").fetchall()]

def upsert_case(case: dict):
    data = {k:case.get(k) for k in ESC_COLS if k!="created_at"}
    with sqlite3.connect(DB_PATH) as conn:
        placeholders = ",".join(["?" for _ in data])
        conn.execute(f"REPLACE INTO escalations ({','.join(data.keys())}) VALUES ({placeholders})", tuple(data.values()))
        conn.commit()

def fetch_cases() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM escalations ORDER BY created_at DESC", conn)

# ========== Email ==========

def send_email(to_email:str, subject:str, body:str, esc_id:str):
    if not (SMTP_SERVER and SMTP_USER and SMTP_PASS):
        st.error("SMTP not configured")
        return False
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = f"Escalation Notification - SE Services <{SMTP_USER}>"
        msg["To"] = to_email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
            s.starttls(); s.login(SMTP_USER, SMTP_PASS); s.send_message(msg)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO notification_log (escalation_id,recipient_email,subject,body,sent_at) VALUES (?,?,?,?,?)",
                         (esc_id,to_email,subject,body,datetime.now().isoformat()))
            conn.commit()
        return True
    except Exception as e:
        st.error(f"Email send failed: {e}")
        return False

# ========== ML Risk Predictor ==========
MODEL_FILE = MODEL_DIR/"risk_model.joblib"
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None

risk_model = load_model()

def train_model(df: pd.DataFrame):
    pipe = Pipeline([("tfidf",TfidfVectorizer(max_features=5000,stop_words="english")),("clf",LogisticRegression(max_iter=1000))])
    pipe.fit(df["issue"], df["escalated"].astype(int))
    joblib.dump(pipe, MODEL_FILE)
    return pipe

def predict_risk(issue:str)->float:
    if risk_model: return float(risk_model.predict_proba([issue])[0][1])
    return 0.0

# ========== Sidebar Upload & Manual Entry ==========
with st.sidebar:
    st.header("📥 Upload Escalations")
    f = st.file_uploader("Excel / CSV", type=["xlsx","csv"])
    if f and st.button("Ingest File"):
        df_up = pd.read_excel(f) if f.name.endswith("xlsx") else pd.read_csv(f)
        for _,row in df_up.iterrows():
            sent,urg,esc = analyze_issue(str(row.get("brief issue","")))
            case={"id":f"ESC-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                  "customer":row.get("customer","Unknown"),
                  "issue":row.get("brief issue","Unknown"),
                  "criticality":row.get("criticalness","Medium"),
                  "impact":row.get("impact","Medium"),
                  "sentiment":sent,"urgency":urg,"escalated":int(esc),
                  "date_reported":str(row.get("issue reported date",datetime.today().date())),
                  "owner":row.get("owner","Unassigned"),
                  "status":row.get("status","Open"),
                  "action_taken":row.get("action taken",""),
                  "risk_score":predict_risk(row.get("brief issue","")),
                  "spoc_email":row.get("spoc_email",""),
                  "spoc_boss_email":row.get("spoc_boss_email",""),
                  "spoc_notify_count":0,"spoc_last_notified":None}
            upsert_case(case)
        st.success("File ingested")

    st.markdown("---"); st.header("✏️ Manual Entry")
    with st.form("manual"):
        cust = st.text_input("Customer"); issue = st.text_area("Issue")
        crit = st.selectbox("Criticality", ["Low","Medium","High"],1)
        imp  = st.selectbox("Impact", ["Low","Medium","High"],1)
        owner= st.text_input("Owner","Unassigned")
        spoc = st.text_input("SPOC Email"); boss = st.text_input("Boss Email")
        submitted = st.form_submit_button("Log Escalation")
        if submitted and cust and issue:
            sent,urg,esc = analyze_issue(issue)
            case={"id":f"ESC-{datetime.now().strftime('%Y%m%d%H%M%S%f')}","customer":cust,"issue":issue,
                  "criticality":crit,"impact":imp,"sentiment":sent,"urgency":urg,
                  "escalated":int(esc),"date_reported":str(datetime.today().date()),
                  "owner":owner,"status":"Open","action_taken":"","risk_score":predict_risk(issue),
                  "spoc_email":spoc,"spoc_boss_email":boss,"spoc_notify_count":0,"spoc_last_notified":None}
            upsert_case(case); st.success("Logged!")

# ========== Scheduler for Escalation to Boss ==========

def boss_escalation_check():
    df = fetch_cases()
    for _,row in df.iterrows():
        if row["spoc_notify_count"]>=2 and row["spoc_boss_email"]:
            last = row["spoc_last_notified"]
            if last and datetime.now()-datetime.fromisoformat(last)>timedelta(hours=24):
                subj=f"⚠️ Escalation {row['id']} unattended"
                body=f"Dear Manager,\n\nEscalation {row['id']} for {row['customer']} has had no SPOC response after 2 reminders.\nIssue: {row['issue']}\nPlease intervene."
                if send_email(row["spoc_boss_email"],subj,body,row["id"]):
                    row_dict=row.to_dict(); row_dict["spoc_notify_count"]+=1; upsert_case(row_dict)

sched = BackgroundScheduler(); sched.add_job(boss_escalation_check,"interval",hours=1); sched.start(); atexit.register(lambda: sched.shutdown(wait=False))

# ========== Kanban Board ==========
st.title("🚨 EscalateAI Kanban Board")
df = fetch_cases()
if df.empty:
    st.info("No escalations logged yet.")
else:
    st.markdown(f"**Open:** {(df.status=='Open').sum()} | **In Progress:** {(df.status=='In Progress').sum()} | **Resolved:** {(df.status=='Resolved').sum()}")
    cols = st.columns(3)
    for status,col in zip(["Open","In Progress","Resolved"],cols):
        with col:
            st.subheader(status)
            for _,row in df[df.status==status].iterrows():
                with st.expander(f"{row['id']} – {row['issue'][:55]}"):
                    new_status = st.selectbox("Status", ["Open","In Progress","Resolved"],
                                               index=["Open","In Progress","Resolved"].index(row["status"]), key=f"stat_{row['id']}")
                    new_action = st.text_area("Action Taken", value=row["action_taken"], key=f"act_{row['id']}")
                    new_spoc   = st.text_input("SPOC Email", value=row["spoc_email"] or "", key=f"spoc_{row['id']}")
                    new_boss   = st.text_input("Boss Email", value=row["spoc_boss_email"] or "", key=f"boss_{row['id']}")
                    if st.button("Notify", key=f"notify_{row['id']}"):
                        if new_spoc:
                            subj=f"Escalation {row['id']} Update"; body=f"Dear SPOC, please review: {row['issue']}"
                            if send_email(new_spoc,subj,body,row["id"]):
                                row_dict=row.to_dict(); row_dict.update({"spoc_notify_count":row["spoc_notify_count"]+1,"spoc_last_notified":datetime.now().isoformat()})
                                upsert_case(row_dict)
                                st.success("SPOC notified")
                    if any([new_status!=row["status"], new_action!=row["action_taken"], new_spoc!=row["spoc_email"], new_boss!=row["spoc
