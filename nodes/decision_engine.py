from config import ask_llm

def decision_engine_node(state: dict) -> dict:
    print("\n⚖️ Node 7: Decision Engine running...")

    risk_score = state["risk_score"]
    credit_score = state["credit_score"]
    dti_ratio = state["dti_ratio"]
    id_verified = state["id_verified"]
    property_valuation = state["property_valuation"]
    loan_amount = state["loan_amount"]
    applicant_name = state["applicant_name"]

    prompt = f"""
    You are the final decision authority at a bank.
    Make a loan decision based on all signals below.

    Applicant: {applicant_name}
    Loan Amount: ${loan_amount}
    ID Verified: {id_verified}
    Credit Score: {credit_score}
    DTI Ratio: {dti_ratio}%
    Property Valuation: {property_valuation}
    Overall Risk: {risk_score}

    Decision rules:
    - LOW risk + all checks pass → AUTO APPROVE
    - MEDIUM risk → ESCALATE TO HUMAN OFFICER
    - HIGH risk → AUTO REJECT
    - ID not verified → AUTO REJECT

    Write a formal 3-sentence bank decision letter to the applicant.
    End with exactly one of:
    DECISION: APPROVED
    DECISION: ESCALATED
    DECISION: REJECTED
    """

    response = ask_llm(prompt, system="You are a bank loan decision authority.")

    response_upper = response.upper()
    if "DECISION: APPROVED" in response_upper:
        decision = "APPROVED ✅"
        escalate = False
    elif "DECISION: REJECTED" in response_upper:
        decision = "REJECTED ❌"
        escalate = False
    else:
        decision = "ESCALATED TO HUMAN OFFICER ⚠️"
        escalate = True

    print(f"\n{'='*50}")
    print(f"FINAL DECISION: {decision}")
    print(f"{'='*50}")
    print(f"\n{response}")

    return {
        "final_decision": decision,
        "escalate_to_human": escalate,
        "risk_score": state["risk_score"],
        "property_valuation": state["property_valuation"],
        "dti_ratio": state["dti_ratio"],
        "credit_score": state["credit_score"],
        "id_verified": state["id_verified"],
        "ocr_text": state["ocr_text"]
    }