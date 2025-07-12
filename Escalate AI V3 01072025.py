# ==============================================================
# EscalateAI – End‑to‑End Escalation Management System (v1.1.2)
# --------------------------------------------------------------
# • Full single‑file Streamlit app
# • SQLite persistence & auto‑schema upgrade
# • Sentiment (HF or rule‑based) + risk ML model
# • Sidebar: Excel/CSV upload & manual entry
# • Kanban board with inline edits & notifications
# • Notification History viewer
# • Robust SMTP email with retries
# • Scheduler escalates to boss after 2 SPOC emails & 24 h
# --------------------------------------------------------------
# Author: Naveen Gandham • July 2025
# ==============================================================

"""Quick‑start:

pip install streamlit pandas openpyxl python-dotenv transformers scikit-learn joblib requests apscheduler
# (Optional) better accuracy – only if PyTorch wheel available:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# .env (same folder)
SMTP_SERVER=smtp.mail.yahoo.com
SMTP_PORT=587
SMTP_USER=naveengandham@yahoo.co.in
SMTP_PASS=<YAHOO_APP_PASSWORD>
SLACK_WEBHOOK_URL=

streamlit run escalateai_app.py
"""

import os, re, sqlite3, atexit, smtplib, time, io
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
        # Ensure upgraded cols
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
                s.starttls()
                s.login(SMTP_USER, SMTP_PASS)
                s.send_message(msg)
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("INSERT INTO notification_log (escalation_id,recipient_email,subject,body,sent_at) VALUES (?,?,?,?,?)",
                             (esc_id,to_email,subject,body,datetime.now().isoformat()))
                conn.commit()
            return True
        except Exception as e:
            attempt+=1
            time.sleep(2)
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
                    subj=f"⚠️ Escalation {r['id']} unattended"
                    body=f"Dear Manager,\n\nEscalation {r['id']} requires your attention. Please follow up with SPOC."
                    if send_email(r["spoc_boss_email"],subj,body,r["id"]):
                        upd=r.to_dict()
                        upd["spoc_notify_count"]+=1
                        upsert_case(upd)
    except Exception as e:
        st.warning(f"Scheduler error: {e}")

if "sched" not in st.session_state:
    sc=BackgroundScheduler()
    sc.add_job(boss_check, "interval", hours=1)
    sc.start()
    atexit.register(lambda: sc.shutdown(wait=False))
    st.session_state["sched"] = True

# ========== UI ==========

st.set_page_config(page_title="EscalateAI", layout="wide")
st.title("🚨 EscalateAI – Escalation Tracking System")

# Sidebar: Upload + Manual Entry

with st.sidebar:
    st.header("📥 Upload Escalations")
    f = st.file_uploader("Excel / CSV", type=["xlsx", "csv"])
    if f and st.button("Analyze & Log"):
        df_up = pd.read_excel(f) if f.name.endswith("xlsx") else pd.read_csv(f)
        df_up.columns = df_up.columns.str.lower().str.strip()
        # Minimal columns expected: customer, brief issue, criticalness, impact, owner, status, action taken, issue reported date
        for _, row in df_up.iterrows():
            issue_text = row.get("brief issue") or row.get("issue") or ""
            sentiment, urgency, escal = analyze_issue(str(issue_text))
            case = {
                "id": f"ESC-{int(time.time()*1000)}", # unique id
                "customer": row.get("customer","Unknown"),
                "issue": issue_text,
                "criticality": row.get("criticalness","Unknown"),
                "impact": row.get("impact","Unknown"),
                "sentiment": sentiment,
                "urgency": urgency,
                "escalated": int(escal),
                "date_reported": str(row.get("issue reported date", datetime.today().date())),
                "owner": row.get("owner","Unassigned"),
                "status": row.get("status","Open"),
                "action_taken": row.get("action taken","None"),
                "risk_score": predict_risk(issue_text),
                "spoc_email": row.get("spoc email",""),
                "spoc_boss_email": row.get("spoc boss email",""),
                "spoc_notify_count": 0,
                "spoc_last_notified": None
            }
            upsert_case(case)
        st.success("File processed & cases logged!")

    st.markdown("---")
    st.header("✏️ Manual Entry")
    with st.form("manual_entry"):
        cname = st.text_input("Customer")
        issue = st.text_area("Issue")
        crit = st.selectbox("Criticality", ["Low", "Medium", "High"], index=1)
        imp = st.selectbox("Impact", ["Low", "Medium", "High"], index=1)
        owner = st.text_input("Owner", value="Unassigned")
        spoc_email = st.text_input("SPOC Email")
        spoc_boss_email = st.text_input("SPOC Boss Email")
        submit = st.form_submit_button("Log")
        if submit and cname and issue:
            sentiment, urgency, escal = analyze_issue(issue)
            case = {
                "id": f"ESC-{int(time.time()*1000)}",
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
                "spoc_notify_count": 0,
                "spoc_last_notified": None
            }
            upsert_case(case)
            st.success(f"Escalation {case['id']} logged!")

# Display Kanban with Summary and Notification button

df = fetch_cases()
if df.empty:
    st.info("No escalations logged yet.")
else:
    st.subheader("📊 Summary")
    st.write(f"Open: {(df['status']=='Open').sum()}  |  In Progress: {(df['status']=='In Progress').sum()}  |  Resolved: {(df['status']=='Resolved').sum()}")

    st.subheader("🗂️ Escalation Kanban Board")

    cols = st.columns(3)
    for status, col in zip(["Open", "In Progress", "Resolved"], cols):
        with col:
            st.markdown(f"### {status}")
            for idx, row in df[df.status==status].iterrows():
                with st.expander(f"{row['id']} – {row['issue'][:60]}..."):
                    st.markdown(f"**Customer:** {row['customer']}")
                    st.markdown(f"**Sentiment / Urgency:** {row['sentiment']} / {row['urgency']}")
                    st.markdown(f"**Owner:** {row['owner']}")
                    st.markdown(f"**Risk Score:** {row['risk_score']:.3f}")
                    new_status = st.selectbox("Update Status", ["Open", "In Progress", "Resolved"],
                        index=["Open", "In Progress", "Resolved"].index(row["status"]),
                        key=f"status_{row['id']}")
                    new_action = st.text_input("Action Taken", value=row["action_taken"], key=f"action_{row['id']}")
                    new_spoc = st.text_input("SPOC Email", value=row.get("spoc_email",""), key=f"spoc_{row['id']}")
                    new_boss = st.text_input("SPOC Boss Email", value=row.get("spoc_boss_email",""), key=f"boss_{row['id']}")

                    notify_button = st.button("Notify SPOC", key=f"notify_{row['id']}")

                    if notify_button:
                        if new_spoc:
                            subj = f"🚨 Escalation Notification: {row['id']}"
                            body = f"Dear SPOC,\n\nPlease attend to escalation {row['id']} reported by {row['customer']}.\nIssue: {row['issue']}\n\nRegards,\nSE Services"
                            if send_email(new_spoc, subj, body, row['id']):
                                upd = row.to_dict()
                                upd["spoc_notify_count"] = (upd.get("spoc_notify_count") or 0) + 1
                                upd["spoc_last_notified"] = datetime.now().isoformat()
                                upd["spoc_email"] = new_spoc
                                upd["spoc_boss_email"] = new_boss
                                upsert_case(upd)
                                st.success("Notification sent to SPOC.")
                                st.experimental_rerun()
                        else:
                            st.warning("Please enter SPOC email before notifying.")

                    if (new_status != row["status"] or new_action != row["action_taken"] or new_spoc != row.get("spoc_email","") or new_boss != row.get("spoc_boss_email","")):
                        upd = row.to_dict()
                        upd["status"] = new_status
                        upd["action_taken"] = new_action
                        upd["spoc_email"] = new_spoc
                        upd["spoc_boss_email"] = new_boss
                        upsert_case(upd)
                        st.experimental_rerun()

    # Excel Download
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name="Escalations")
    output.seek(0)

    st.download_button(
        label="⬇️ Download as Excel",
        data=output,
        file_name="escalations_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Notification log viewer

with st.expander("📧 Notification History"):
    logs = fetch_logs()
    if logs.empty:
        st.info("No notifications sent yet.")
    else:
        st.dataframe(logs[["escalation_id","recipient_email","subject","sent_at"]])

