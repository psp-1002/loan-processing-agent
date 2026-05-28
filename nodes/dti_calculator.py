from config import ask_llm

def dti_calculator_node(state: dict) -> dict:
    print("\n🧮 Node 4: Debt-to-Income Ratio running...")

    monthly_income = state["monthly_income"]
    loan_amount = state["loan_amount"]
    existing_debts = 500  # from OCR

    # Estimate monthly loan payment (30 year mortgage at ~7% interest)
    monthly_loan_payment = round((loan_amount * 0.07 / 12) / (1 - (1 + 0.07/12) ** -360), 2)
    total_monthly_debt = existing_debts + monthly_loan_payment
    dti_ratio = round((total_monthly_debt / monthly_income) * 100, 2)

    prompt = f"""
    You are a loan underwriter analyzing debt-to-income ratio.

    Applicant financials:
    - Monthly Income: ${monthly_income}
    - Existing Monthly Debts: ${existing_debts}
    - Requested Loan Amount: ${loan_amount}
    - Estimated Monthly Loan Payment: ${monthly_loan_payment}
    - Total Monthly Debt: ${total_monthly_debt}
    - DTI Ratio: {dti_ratio}%

    Standard DTI thresholds:
    - Below 36%: Excellent
    - 36% to 43%: Acceptable
    - 43% to 50%: Risky
    - Above 50%: Too high — likely rejection

    Provide a one paragraph assessment of this applicant's DTI.
    """

    assessment = ask_llm(prompt, system="You are a loan underwriting specialist.")

    print(f"✅ DTI Ratio: {dti_ratio}% — Monthly payment: ${monthly_loan_payment}")

    return {
        "dti_ratio": dti_ratio,
        "credit_score": state["credit_score"],
        "id_verified": state["id_verified"],
        "ocr_text": state["ocr_text"]
    }