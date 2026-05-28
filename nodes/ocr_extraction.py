from config import ask_llm
from typing import Any

def ocr_extraction_node(state: dict) -> dict:
    print("\n📄 Node 1: OCR Extraction running...")

    # In a real system this would read an actual uploaded PDF/image
    # For now we simulate a document with the applicant's data
    simulated_document = f"""
    LOAN APPLICATION DOCUMENT
    -------------------------
    Applicant Name  : {state['applicant_name']}
    Monthly Income  : ${state['monthly_income']}
    Loan Amount     : ${state['loan_amount']}
    Loan Purpose    : {state['loan_purpose']}
    Property Value  : ${state.get('property_value', 'Not provided')}
    Employment Type : Salaried
    Years Employed  : 4 years
    Existing Debts  : $500/month
    """

    prompt = f"""
    You are a document OCR extraction specialist.
    Extract all key financial information from this document
    and return it as a clean structured summary.

    Document:
    {simulated_document}

    Return a clean bullet-point summary of all extracted fields.
    """

    extracted_text = ask_llm(prompt, system="You are a document extraction specialist.")

    print("✅ OCR extraction complete")
    print(f"   Extracted: {extracted_text[:100]}...")

    return {"ocr_text": extracted_text}