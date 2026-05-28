import streamlit as st
from graph import build_graph

st.set_page_config(page_title="Loan Processing Agent", page_icon="🏦", layout="centered")

st.title("🏦 Loan Processing Agent")
st.markdown("*Powered by LangGraph + LLaMA3 + Qdrant · Processes in 2 minutes vs 5 days*")
st.divider()

with st.form("loan_form"):
    st.subheader("Applicant Details")
    col1, col2 = st.columns(2)
    with col1:
        name         = st.text_input("Full Name", value="Sourav Prusty")
        income       = st.number_input("Monthly Income ($)", value=5000, step=500)
        loan_purpose = st.selectbox("Loan Purpose", ["Home purchase", "Business", "Education", "Vehicle", "Personal"])
    with col2:
        loan_amount    = st.number_input("Loan Amount ($)", value=200000, step=10000)
        property_value = st.number_input("Property Value ($)", value=250000, step=10000)

    submitted = st.form_submit_button("🚀 Process Loan Application", use_container_width=True)

if submitted:
    st.divider()
    st.subheader("Processing Application...")

    progress = st.progress(0)
    status   = st.empty()

    nodes = [
        "📄 OCR Extraction",
        "🪪 ID Verification",
        "📊 Credit Score Analysis",
        "🧮 DTI Calculation",
        "🏠 Property Valuation",
        "⚠️ Risk Scoring",
        "⚖️ Decision Engine",
    ]

    for i, node in enumerate(nodes):
        status.info(f"Running {node}...")
        progress.progress((i + 1) / len(nodes))

    status.empty()
    progress.empty()

    with st.spinner("Finalizing decision..."):
        app    = build_graph()
        result = app.invoke({
            "applicant_name":    name,
            "monthly_income":    income,
            "loan_amount":       loan_amount,
            "loan_purpose":      loan_purpose,
            "property_value":    property_value,
            "ocr_text":          None,
            "id_verified":       None,
            "credit_score":      None,
            "dti_ratio":         None,
            "property_valuation":None,
            "risk_score":        None,
            "risk_label":        None,
            "risk_justification":None,
            "similar_loans_count":None,
            "final_decision":    None,
            "escalate_to_human": None,
        })

    st.divider()
    st.subheader("📋 Application Results")

    # ── Row 1: core metrics ───────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Credit Score",  result.get("credit_score", "N/A"))
    col2.metric("DTI Ratio",     f"{result.get('dti_ratio', 'N/A')}%")
    col3.metric("Property",      result.get("property_valuation", "N/A"))
    col4.metric("Risk Level",    result.get("risk_label", result.get("risk_score", "N/A")))

    # ── Row 2: Qdrant RAG context ─────────────────────────────
    similar_count = result.get("similar_loans_count", 0)
    justification = result.get("risk_justification", "")

    if similar_count and similar_count > 0:
        st.caption(f"🗄️ Risk assessed using **{similar_count} similar past loan(s)** retrieved from Qdrant")
    else:
        st.caption("🗄️ No similar past loans in Qdrant yet — risk assessed from applicant data alone (cold start)")

    if justification:
        st.info(f"**Risk Rationale:** {justification}")

    st.divider()

    # ── Final decision banner ─────────────────────────────────
    decision = result.get("final_decision", "UNKNOWN")
    if "APPROVED" in decision:
        st.success(f"## ✅ {decision}")
    elif "REJECTED" in decision:
        st.error(f"## ❌ {decision}")
    else:
        st.warning(f"## ⚠️ {decision}")