from config import ask_llm

def property_valuation_node(state: dict) -> dict:
    print("\n🏠 Node 5: Property Valuation running...")

    loan_amount = state["loan_amount"]
    property_value = state.get("property_value", 0)

    # Loan to Value ratio — key metric in mortgage lending
    ltv_ratio = round((loan_amount / property_value) * 100, 2) if property_value else 100

    prompt = f"""
    You are a property valuation specialist at a bank.

    Loan Details:
    - Loan Amount Requested: ${loan_amount}
    - Property Value: ${property_value}
    - Loan-to-Value (LTV) Ratio: {ltv_ratio}%

    LTV Thresholds:
    - Below 80%: Excellent — low risk
    - 80% to 90%: Acceptable — may need insurance
    - 90% to 95%: High risk
    - Above 95%: Very high risk — likely rejection

    Give a one paragraph assessment of the property valuation
    and whether it adequately secures the loan.

    End with exactly one of:
    VALUATION: ADEQUATE
    VALUATION: BORDERLINE
    VALUATION: INADEQUATE
    """

    response = ask_llm(prompt, system="You are a property valuation specialist.")

    # Extract valuation verdict
    response_upper = response.upper()
    if "VALUATION: ADEQUATE" in response_upper:
        valuation = "ADEQUATE"
    elif "VALUATION: INADEQUATE" in response_upper:
        valuation = "INADEQUATE"
    else:
        valuation = "BORDERLINE"

    print(f"✅ Property Valuation: {valuation} (LTV: {ltv_ratio}%)")

    return {
        "property_valuation": valuation,
        "dti_ratio": state["dti_ratio"],
        "credit_score": state["credit_score"],
        "id_verified": state["id_verified"],
        "ocr_text": state["ocr_text"]
    }