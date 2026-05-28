from config import ask_llm

def risk_scoring_node(state: dict) -> dict:
    print("\n⚠️ Node 6: Risk Scoring running...")

    credit_score = state["credit_score"]
    dti_ratio = state["dti_ratio"]
    property_valuation = state["property_valuation"]
    id_verified = state["id_verified"]

    prompt = f"""
    You are a senior risk analyst at a bank.
    Combine all signals below and produce a final risk score.

    Applicant Signals:
    - ID Verified: {id_verified}
    - Credit Score: {credit_score} / 850
    - DTI Ratio: {dti_ratio}%
    - Property Valuation: {property_valuation}

    Scoring guide:
    - Credit score above 750: LOW risk
    - Credit score 650-750: MEDIUM risk
    - Credit score below 650: HIGH risk
    - DTI above 43%: increases risk by one level
    - Property INADEQUATE: increases risk by one level
    - ID not verified: automatic HIGH risk

    Give a 2-sentence risk summary then end with exactly one of:
    RISK: LOW
    RISK: MEDIUM
    RISK: HIGH
    """

    response = ask_llm(prompt, system="You are a senior bank risk analyst.")

    response_upper = response.upper()
    if "RISK: LOW" in response_upper:
        risk_score = "LOW"
    elif "RISK: HIGH" in response_upper:
        risk_score = "HIGH"
    else:
        risk_score = "MEDIUM"

    print(f"✅ Risk Score: {risk_score}")

    return {
        "risk_score": risk_score,
        "property_valuation": state["property_valuation"],
        "dti_ratio": state["dti_ratio"],
        "credit_score": state["credit_score"],
        "id_verified": state["id_verified"],
        "ocr_text": state["ocr_text"]
    }