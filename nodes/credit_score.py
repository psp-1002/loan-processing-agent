from config import ask_llm

def credit_score_node(state: dict) -> dict:
    print("\n📊 Node 3: Credit Score Analysis running...")

    prompt = f"""
    You are a credit analyst at a bank.
    Analyze the applicant's creditworthiness based on this information:

    Applicant: {state['applicant_name']}
    Monthly Income: ${state['monthly_income']}
    Loan Amount Requested: ${state['loan_amount']}
    Loan Purpose: {state['loan_purpose']}
    Existing Monthly Debts: $500
    Employment: Salaried, 4 years

    Based on this profile, estimate a realistic credit score between 300-850.
    Consider income stability, debt levels, and loan-to-income ratio.

    Respond in exactly this format:
    CREDIT_SCORE: [number between 300-850]
    ASSESSMENT: [one sentence explanation]
    """

    response = ask_llm(prompt, system="You are a credit scoring specialist.")

    # Extract credit score from response
    credit_score = 650  # default
    for line in response.split("\n"):
        if "CREDIT_SCORE:" in line:
            try:
                credit_score = int(line.split(":")[1].strip())
            except:
                credit_score = 650

    print(f"✅ Credit Score: {credit_score}")

    return {
        "credit_score": credit_score,
        "ocr_text": state["ocr_text"],
        "id_verified": state["id_verified"]
    }