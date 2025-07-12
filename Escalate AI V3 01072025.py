# ==============================================================
# EscalateAI – End‑to‑End Escalation Management System (v1.2.0)
# --------------------------------------------------------------
# • Full single‑file Streamlit app
# • SQLite persistence & auto‑schema upgrade
# • Sentiment (HF or rule‑based) + risk ML model
# • Sidebar: Excel/CSV upload & manual entry
# • Kanban board with inline edits & notifications
# • Notification History viewer & Email Logs
# • Robust SMTP email with retries
# • Scheduler escalates to boss after 2 SPOC emails & 24 h
# --------------------------------------------------------------
# Author: Naveen Gandham • July 2025
# ==============================================================

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

# ========== Set Page Config ==========
st.set_page_config(page_title="EscalateAI", layout="wide")

# ========== Paths & ENV ==========
APP_DIR   = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models" ; MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR  = APP_DIR / "data"  ; DATA_DIR.mkdir(exist_ok=True)
DB_PATH   = DATA_DIR / "escalateai.db"

load_dotenv()
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT   = int(os.getenv("SMTP_PORT", 587))
SMTP_USER   = os.getenv("SMTP_USER")
SMTP_PASS   = os.getenv("SMTP_PASS")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

# ========== Sentiment ==========
try:
    from transformers import pipeline as hf_pipeline; import torch ; HAS_NLP=True
except Exception:
    HAS_NLP=False

@st.cache_resource(show_spinner=False)
def load_sentiment():
    if not HAS_NLP: return None
    try:
        return hf_pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    except Exception:
        return None

sent_model = load_sentiment()
negative_words = [
    r"\b(problematic|delay|issue|failure|dissatisfaction|frustration|unacceptable|mistake|complaint|unresolved|unresponsive|unstable|broken|defective|overdue|escalation|leakage|damage|burnt|critical|risk|dispute|faulty)\b"
]
NEG_WORDS = negative_words

def rule_sent(text:str)->str:
    return "Negative" if any(re.search(w,text,re.I) for w in NEG_WORDS) else "Positive"

def analyze_issue(text:str)->Tuple[str,str,bool]:
    if sent_model:
        label = sent_model(text[:512])[0]["label"].lower()
        sentiment = "Negative" if label=="negative" else "Positive"
    else:
        sentiment = rule_sent(text)
    urgency = "High" if any(k in text.lower() for k in ["urgent","immediate","critical"]) else "Low"
    return sentiment, urgency, sentiment=="Negative" and urgency=="High"

# ========== DB init & helpers ==========

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cur=conn.cursor()
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
        cur.execute("""CREATE TABLE IF NOT EXISTS notification_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            escalation_id TEXT,
            recipient_email TEXT,
            subject TEXT,
            body TEXT,
            sent_at TEXT)""")
        conn.commit()
        cur.execute("PRAGMA table_info(escalations)"); cols=[c[1] for c in cur.fetchall()]
        need={"spoc_notify_count":"INTEGER DEFAULT 0","spoc_last_notified":"TEXT","spoc_email":"TEXT","spoc_boss_email":"TEXT"}
        for c,t in need.items():
            if c not in cols:
                try: cur.execute(f"ALTER TABLE escalations ADD COLUMN {c} {t}")
                except Exception: pass
        conn.commit()

init_db()
ESC_COLS=[c[1] for c in sqlite3.connect(DB_PATH).execute("PRAGMA table_info(escalations)").fetchall()]

def upsert_case(case:dict):
    data={k:case.get(k) for k in ESC_COLS}
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"REPLACE INTO escalations ({','.join(data.keys())}) VALUES ({','.join('?'*len(data))})", tuple(data.values()))
        conn.commit()

def fetch_cases()->pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM escalations ORDER BY created_at DESC", conn)

def fetch_logs()->pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM notification_log ORDER BY sent_at DESC", conn)

# ========== Email (retries) ==========

def send_email(to_email:str, subject:str, body:str, esc_id:str, retries:int=3)->bool:
    if not (SMTP_SERVER and SMTP_USER and SMTP_PASS):
        st.error("SMTP not configured")
        return False
    attempt=0
    while attempt<retries:
        try:
            msg=MIMEText(body)
            msg["Subject"]=subject
            msg["From"]=f"Escalation Notification - SE Services <{SMTP_USER}>"
            msg["To"]=to_email
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
                s.starttls(); s.login(SMTP_USER, SMTP_PASS); s.send_message(msg)
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("INSERT INTO notification_log (escalation_id,recipient_email,subject,body,sent_at) VALUES (?,?,?,?,?)",
                             (esc_id,to_email,subject,body,datetime.now().isoformat())); conn.commit()
            return True
        except Exception as e:
            attempt+=1; time.sleep(2)
            if attempt==retries:
                st.error(f"Email failed: {e}")
                return False

# risk predictor stub
MODEL_FILE=MODEL_DIR/"risk_model.joblib"
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None
risk_model=load_model()

def predict_risk(issue:str)->float:
    return float(risk_model.predict_proba([issue])[0][1]) if risk_model else 0.0

# Scheduler boss check

def boss_check():
    try:
        df=fetch_cases()
        for _,r in df.iterrows():
            if r.get("spoc_notify_count",0)>=2 and r.get("spoc_boss_email") and r.get("spoc_last_notified"):
                if datetime.now()-datetime.fromisoformat(r["spoc_last_notified"])>timedelta(hours=24):
                    subj=f"⚠️ Escalation {r['id']} unattended"; body=f"Dear Manager,\n\nEscalation {r['id']} requires your attention."
                    if send_email(r["spoc_boss_email"],subj,body,r["id"]):
                        upd=r.to_dict(); upd["spoc_notify_count"]+=1; upsert_case(upd)
    except Exception as e:
        st.warning(f"Scheduler error: {e}")

if "sched" not in st.session_state:
    sc=BackgroundScheduler()
    sc.add_job(boss_check, "interval", hours=1)
    sc.start()
    atexit.register(lambda: sc.shutdown(wait=False))
    st.session_state["sched"] = True

# ========== Sidebar Upload & Manual Entry ==========
with st.sidebar:
    st.header("📥 Upload Escalations")
    f = st.file_uploader("Excel / CSV", type=["xlsx", "csv"])
    if f and st.button("Ingest File"):
        df_up = pd.read_excel(f) if f.name.endswith("xlsx") else pd.read_csv(f)
        for _, row in df_up.iterrows():
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
        st.success("File ingested")

    st.markdown("---")
    st.header("✏️ Manual Entry")
    with st.form("manual"):
        cust = st.text_input("Customer")
        issue = st.text_area("Issue")
        crit = st.selectbox("Criticality", ["Low", "Medium", "High"], 1)
        imp = st.selectbox("Impact", ["Low", "Medium", "High"], 1)
        owner = st.text_input("Owner", "Unassigned")
        spoc = st.text_input("SPOC Email")
        boss = st.text_input("Boss Email")
        submitted = st.form_submit_button("Log Escalation")
        if submitted and cust and issue:
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
                "spoc_email": spoc,
                "spoc_boss_email": boss,
                "spoc_notify_count": 0,
                "spoc_last_notified": None,
            }
            upsert_case(case)
            st.success("Logged!")


# ========== Scheduler for Escalation to Boss ==========
def boss_escalation_check():
    df = fetch_cases()
    for _, row in df.iterrows():
        # Only escalate if SPOC notified twice or more, boss email provided
        if row.get("spoc_notify_count", 0) >= 2 and row.get("spoc_boss_email"):
            last = row.get("spoc_last_notified")
            if last:
                last_dt = datetime.fromisoformat(last)
                if datetime.now() - last_dt > timedelta(hours=24):
                    subj = f"⚠️ Escalation {row['id']} unattended"
                    body = (
                        f"Dear Manager,\n\n"
                        f"Escalation {row['id']} for {row['customer']} has had no SPOC response after 2 reminders.\n"
                        f"Issue: {row['issue']}\nPlease intervene."
                    )
                    sent = send_email(row["spoc_boss_email"], subj, body, row["id"])
                    if sent:
                        # Increment notify count so we don't spam further
                        updated = row.to_dict()
                        updated["spoc_notify_count"] = (row.get("spoc_notify_count", 0) or 0) + 1
                        upsert_case(updated)


sched = BackgroundScheduler()
sched.add_job(boss_escalation_check, "interval", hours=1)
sched.start()
atexit.register(lambda: sched.shutdown(wait=False))

# ========== Kanban Board ==========
st.title("🚨 EscalateAI Kanban Board")
df = fetch_cases()

if df.empty:
    st.info("No escalations logged yet.")
else:
    # Summary counts on top
    st.markdown(
        f"**Open:** {(df.status == 'Open').sum()} | "
        f"**In Progress:** {(df.status == 'In Progress').sum()} | "
        f"**Resolved:** {(df.status == 'Resolved').sum()}"
    )
    cols = st.columns(3)
    for status, col in zip(["Open", "In Progress", "Resolved"], cols):
        with col:
            st.subheader(status)
            for _, row in df[df.status == status].iterrows():
                with st.expander(f"{row['id']} – {row['issue'][:55]}"):
                    new_status = st.selectbox(
                        "Status",
                        ["Open", "In Progress", "Resolved"],
                        index=["Open", "In Progress", "Resolved"].index(row["status"]),
                        key=f"stat_{row['id']}",
                    )
                    new_action = st.text_area("Action Taken", value=row["action_taken"], key=f"act_{row['id']}")
                    new_spoc = st.text_input("SPOC Email", value=row["spoc_email"] or "", key=f"spoc_{row['id']}")
                    new_boss = st.text_input("Boss Email", value=row["spoc_boss_email"] or "", key=f"boss_{row['id']}")

                    # Notify button to send email to SPOC
                    if st.button("Notify SPOC", key=f"notify_{row['id']}"):
                        if new_spoc:
                            subj = f"Escalation {row['id']} Update"
                            body = f"Dear SPOC, please review the escalation:\n\n{row['issue']}"
                            sent = send_email(new_spoc, subj, body, row["id"])
                            if sent:
                                updated = row.to_dict()
                                updated["spoc_notify_count"] = (row.get("spoc_notify_count", 0) or 0) + 1
                                updated["spoc_last_notified"] = datetime.now().isoformat()
                                updated["spoc_email"] = new_spoc
                                updated["spoc_boss_email"] = new_boss
                                upsert_case(updated)
                                st.success("SPOC notified!")

                    # Save updates to status, action, SPOC and Boss emails
                    if (
                        new_status != row["status"]
                        or new_action != row["action_taken"]
                        or new_spoc != (row["spoc_email"] or "")
                        or new_boss != (row["spoc_boss_email"] or "")
                    ):
                        updated = row.to_dict()
                        updated["status"] = new_status
                        updated["action_taken"] = new_action
                        updated["spoc_email"] = new_spoc
                        updated["spoc_boss_email"] = new_boss
                        upsert_case(updated)
                        st.experimental_rerun()

    # Download button to export all escalations
    towrite = pd.ExcelWriter("escalations_export.xlsx", engine="openpyxl")
    df.to_excel(towrite, index=False)
    towrite.save()
    with open("escalations_export.xlsx", "rb") as f:
        st.download_button(
            "⬇️ Download as Excel",
            data=f,
            file_name="escalations_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
# ======= Download Excel =========
df_all = fetch_cases()
if not df_all.empty:
    with st.sidebar:
        st.markdown("---")
        st.subheader("📤 Download Escalation Report")
        excel_bytes = df_all.to_excel(index=False, engine='openpyxl')
        st.download_button("⬇️ Download Excel with Status", data=excel_bytes,
                           file_name="Escalation_Status_Report.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
