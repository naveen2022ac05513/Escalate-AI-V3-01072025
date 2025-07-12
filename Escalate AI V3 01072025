import streamlit as st
import pandas as pd
import re
# Excel Based and Manual Entry of Escalations
# Set page configuration
st.set_page_config(page_title="EscalateAI - Escalation Tracking", layout="wide")

# ---------------------------------
# Sentiment Analysis for Issue Detection
# ---------------------------------
def analyze_issue(text):
    text_lower = text.lower()

    negative_words = [
        r"\b(problematic|delay|issue|failure|dissatisfaction|frustration|unacceptable|mistake|complaint|unresolved|unresponsive|unstable|broken|defective|overdue|escalation|leakage|damage|burnt|critical|risk|dispute|faulty)\b"
    ]

    sentiment = "Negative" if any(re.search(word, text_lower) for word in negative_words) else "Positive"
    urgency = "High" if any(word in text_lower for word in ["urgent", "critical", "immediately", "business impact"]) else "Low"
    escalation = sentiment == "Negative" and urgency == "High"

    return sentiment, urgency, escalation

# ---------------------------------
# Generate Sequential Escalation ID
# ---------------------------------
if "escalation_counter" not in st.session_state:
    st.session_state.escalation_counter = 10000  # Starting from a fixed number

def generate_escalation_id():
    escalation_id = f"ESC-{st.session_state.escalation_counter}"
    st.session_state.escalation_counter += 1  # Increment for the next entry
    return escalation_id
# ---------------------------------
# Log Escalation
# ---------------------------------
def log_case(row, sentiment, urgency, escalated):
    if "cases" not in st.session_state:
        st.session_state.cases = []

    escalation_id = generate_escalation_id()

    # Ensure proper key extraction
    case = {
        "Escalation ID": escalation_id,
        "Customer": row.get("Customer", row.get("customer", "Unknown")),  # Handling column variations
        "Criticality": row.get("Criticalness", row.get("criticalness", "Unknown")),
        "Issue": row.get("Brief Issue", row.get("brief issue", "Unknown")),
        "Sentiment": sentiment,
        "Urgency": urgency,
        "Escalated": escalated,
        "Date Reported": row.get("Issue reported date", row.get("issue reported date", "Unknown")),  # Ensure correct field extraction
        "Owner": row.get("Owner", row.get("owner", "Unknown")),
        "Status": row.get("Status", "Open"),
        "Action Taken": row.get("Action Taken", "None")  # New field for action taken
    }

    print(f"Logging case: {case}")  # Debugging line to verify data

    st.session_state.cases.append(case)

# ---------------------------------
# Show Kanban Board with Correct Buckets
# ---------------------------------
def show_kanban():
    if "cases" not in st.session_state or not st.session_state.cases:
        st.info("No escalations logged yet.")
        return

    # Count cases by status
    status_counts = {
        "Open": sum(1 for case in st.session_state.cases if case["Status"] == "Open"),
        "In Progress": sum(1 for case in st.session_state.cases if case["Status"] == "In Progress"),
        "Resolved": sum(1 for case in st.session_state.cases if case["Status"] == "Resolved"),
    }

    st.subheader(f"🗂️ Escalation Kanban Board (Open: {status_counts['Open']} | In Progress: {status_counts['In Progress']} | Resolved: {status_counts['Resolved']})")

    # Explicitly define three buckets
    col_open, col_progress, col_resolved = st.columns(3)
    stages = {"Open": col_open, "In Progress": col_progress, "Resolved": col_resolved}

    for case in st.session_state.cases:
        status = case.get("Status", "Open")
        if status not in stages:
            status = "Open"  # Default to Open if an invalid status is found

        with stages[status]:
            st.markdown("----")
            st.markdown(f"**🔷 Escalation ID: {case['Escalation ID']}**")
            st.markdown(f"**🧾 Issue: {case['Issue']}**")
            st.write(f"👤 **Customer**: `{case['Customer']}`")
            st.write(f"🔥 **Criticality**: `{case['Criticality']}`")
            st.write(f"📅 Reported: `{case['Date Reported']}`")  # Display issue reported date
            st.write(f"👤 **Owner**: `{case.get('Owner', 'N/A')}`")
            st.write(f"✅ Escalated: `{case['Escalated']}`")
            st.write(f"🔧 **Action Taken**: `{case['Action Taken']}`")  # Display action taken
            # Allow status updates
            new_status = st.selectbox(
                "Update Status",
                ["Open", "In Progress", "Resolved"],
                index=["Open", "In Progress", "Resolved"].index(status),
                key=f"{case['Escalation ID']}_status"
            )

            case["Status"] = new_status  # Update status in session state

            # Allow action taken updates
            new_action_taken = st.text_area(
                "Update Action Taken",
                value=case["Action Taken"],
                key=f"{case['Escalation ID']}_action_taken"
            )

            case["Action Taken"] = new_action_taken  # Update action taken in session state

# ---------------------------------
# Main App Logic
# ---------------------------------
st.title("🚨 EscalateAI - Escalation Tracking System")

# Sidebar: Excel Upload & Manual Entry
with st.sidebar:
    st.header("📥 Upload Escalation Tracker")
    file = st.file_uploader("Upload Excel File", type=["xlsx"])

    if file is not None:
        df = pd.read_excel(file)
        df.columns = df.columns.str.strip().str.lower().str.replace(" +", " ", regex=True)

        required_cols = {"customer", "brief issue", "issue reported date", "status", "owner", "criticalness"}
        missing_cols = required_cols - set(df.columns)

        if missing_cols:
            st.error(f"Excel file is missing required columns: {', '.join(missing_cols)}")
        else:
            if st.button("🔍 Analyze Issues & Log Escalations"):
                for _, row in df.iterrows():
                    sentiment, urgency, escalated = analyze_issue(row["brief issue"])
                    log_case(row, sentiment, urgency, escalated)
                st.success("Escalations auto-logged from Excel file!")

    st.header("✏️ Manual Entry")
    with st.form(key="manual_entry_form"):
        customer_name = st.text_input("Customer Name")
        issue = st.text_area("Issue")
        criticality = st.selectbox("Criticality", ["Low", "Medium", "High"])
        impact = st.selectbox("Impact", ["Low", "Medium", "High"])
        action_owner = st.text_input("Action Owner")
        date_reported = st.date_input("Date Reported")

        submit_button = st.form_submit_button("Log Escalation")

        if submit_button:
            if customer_name and issue:
                sentiment, urgency, escalated = analyze_issue(issue)
                log_case({
                    "Customer": customer_name,
                    "Brief Issue": issue,
                    "Criticalness": criticality,
                    "Impact": impact,
                    "Owner": action_owner,
                    "Issue reported date": date_reported,
                }, sentiment, urgency, escalated)
                st.success("Escalation logged successfully!")
            else:
                st.error("Please fill all fields.")

# Display Kanban Board
show_kanban()

# Option to download escalations
if "cases" in st.session_state and st.session_state.cases:
    df_cases = pd.DataFrame(st.session_state.cases)
    st.download_button(
        label="Download Escalations as Excel",
        data=df_cases.to_excel(index=False, engine="openpyxl"),
        file_name="escalations.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
