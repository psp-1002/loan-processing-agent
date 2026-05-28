# nodes/risk_scoring.py
import os
import json
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue
)
from sentence_transformers import SentenceTransformer
from config import ask_llm

load_dotenv()

# ── Qdrant setup ──────────────────────────────────────────────
QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION     = "past_loans"
VECTOR_DIM     = 384          # matches all-MiniLM-L6-v2

client  = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
encoder = SentenceTransformer("all-MiniLM-L6-v2")


def _ensure_collection():
    """Create collection if it doesn't exist yet."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )


def _loan_to_text(state: dict) -> str:
    """Converts loan state fields into a descriptive sentence for embedding."""
    return (
        f"Applicant {state.get('applicant_name', 'Unknown')} "
        f"with credit score {state.get('credit_score', 'N/A')}, "
        f"DTI ratio {state.get('dti_ratio', 'N/A')}%, "
        f"loan amount {state.get('loan_amount', 'N/A')}, "
        f"property valuation {state.get('property_valuation', 'N/A')}, "
        f"income {state.get('monthly_income', 'N/A')}."
    )


def store_loan_in_qdrant(state: dict, risk_label: str, decision: str = None):
    """
    Call this after a loan is decided to build up the historical dataset.
    decision is optional — pass 'APPROVED'/'REJECTED' etc. when available.
    """
    _ensure_collection()
    text   = _loan_to_text(state)
    vector = encoder.encode(text).tolist()

    payload = {
        "applicant_name":    state.get("applicant_name"),
        "credit_score":      state.get("credit_score"),
        "dti_ratio":         state.get("dti_ratio"),
        "loan_amount":       state.get("loan_amount"),
        "property_valuation":state.get("property_valuation"),
        "monthly_income":    state.get("monthly_income"),
        "risk_label":        risk_label,
        "decision":          decision or "PENDING",
        "text_summary":      text,
    }

    client.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)],
    )


def _retrieve_similar_loans(state: dict, top_k: int = 3) -> list[dict]:
    """Vector search — find the top_k most similar past loans."""
    _ensure_collection()
    text   = _loan_to_text(state)
    vector = encoder.encode(text).tolist()

    results = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    return [hit.payload for hit in results.points]


# ── Main node function ────────────────────────────────────────
def risk_scoring_node(state: dict) -> dict:
    """
    Node 6: RAG-powered risk scoring.
    1. Retrieve similar past loans from Qdrant.
    2. Pass them as context to LLaMA3 for risk assessment.
    3. Store this loan in Qdrant for future retrievals.
    """
    similar_loans = _retrieve_similar_loans(state)

    # Format retrieved context for the prompt
    if similar_loans:
        context_lines = []
        for i, loan in enumerate(similar_loans, 1):
            context_lines.append(
                f"  Past Loan {i}: Credit={loan.get('credit_score')}, "
                f"DTI={loan.get('dti_ratio')}%, "
                f"Risk={loan.get('risk_label')}, "
                f"Decision={loan.get('decision')}"
            )
        rag_context = "\n".join(context_lines)
    else:
        rag_context = "  No similar past loans found — assess from applicant data alone."

    prompt = f"""You are a senior bank risk analyst.

CURRENT APPLICANT:
- Name          : {state.get('applicant_name', 'Unknown')}
- Credit Score  : {state.get('credit_score', 'N/A')}
- DTI Ratio     : {state.get('dti_ratio', 'N/A')}%
- Loan Amount   : {state.get('loan_amount', 'N/A')}
- Property Value: {state.get('property_valuation', 'N/A')}
- Monthly Income: {state.get('monthly_income', 'N/A')}

SIMILAR PAST LOANS (retrieved from database):
{rag_context}

Based on the applicant's profile AND the outcomes of similar past loans,
assign a risk level. Reply with ONLY one of: LOW, MEDIUM, HIGH
Then on the next line, give a one-sentence justification."""

    response = ask_llm(prompt).strip()
    lines    = [l.strip() for l in response.splitlines() if l.strip()]

    # Parse risk label robustly
    risk_label   = "MEDIUM"   # safe default
    justification = ""
    for line in lines:
        upper = line.upper()
        if upper in ("LOW", "MEDIUM", "HIGH"):
            risk_label = upper
        elif any(k in upper for k in ("LOW", "MEDIUM", "HIGH")):
            for k in ("HIGH", "MEDIUM", "LOW"):   # order matters — check HIGH first
                if k in upper:
                    risk_label = k
                    break
        else:
            justification = line

    # Store this loan so it enriches future retrievals
    store_loan_in_qdrant(state, risk_label)

    print(f"\n[Risk Scoring] Similar loans retrieved: {len(similar_loans)}")
    print(f"[Risk Scoring] Risk → {risk_label}")
    if justification:
        print(f"[Risk Scoring] Reason: {justification}")

    return {
        **state,
        "risk_label":          risk_label,
        "risk_justification":  justification,
        "similar_loans_count": len(similar_loans),
    }