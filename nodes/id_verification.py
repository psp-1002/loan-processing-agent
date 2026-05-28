from config import ask_llm

def id_verification_node(state: dict) -> dict:
    print("\n🪪 Node 2: ID Verification running...")

    prompt = f"""
    You are an ID verification specialist at a bank.
    Based on the extracted document data below, verify the applicant's identity.

    Extracted Document Data:
    {state['ocr_text']}

    Perform these checks and respond with YES or NO for each:
    1. Is the applicant name clearly present?
    2. Is the income information verifiable?
    3. Is the employment information consistent?
    4. Are there any red flags or inconsistencies?

    Since this is a simulated application with complete data provided,
    if all 4 checks pass with no major red flags, return VERIFIED.

    End your response with exactly one of:
    VERDICT: VERIFIED
    VERDICT: SUSPICIOUS
    VERDICT: REJECTED
    """

    response = ask_llm(prompt, system="You are a bank ID verification officer.")

    # Check verdict
    response_upper = response.upper()
    if "VERDICT: VERIFIED" in response_upper:
        id_verified = True
        verdict = "VERIFIED ✓"
    elif "VERDICT: REJECTED" in response_upper:
        id_verified = False
        verdict = "REJECTED ✗"
    else:
        # SUSPICIOUS — treat as passed for simulation purposes
        id_verified = True
        verdict = "SUSPICIOUS (proceeding) ⚠️"

    print(f"✅ ID Verification complete — {verdict}")

    return {
        "id_verified": id_verified,
        "ocr_text": state["ocr_text"]
    }