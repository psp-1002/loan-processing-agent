from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

# This is the "state" — the loan application data
# that gets passed through all 7 nodes
class LoanState(TypedDict):
    # Applicant info
    applicant_name: str
    monthly_income: float
    loan_amount: float
    loan_purpose: str
    property_value: Optional[float]

    # Results filled in by each node
    ocr_text: Optional[str]
    id_verified: Optional[bool]
    credit_score: Optional[int]
    dti_ratio: Optional[float]
    property_valuation: Optional[str]
    risk_score: Optional[str]
    final_decision: Optional[str]
    escalate_to_human: Optional[bool]

def build_graph():
    from nodes.ocr_extraction import ocr_extraction_node
    from nodes.id_verification import id_verification_node
    from nodes.credit_score import credit_score_node
    from nodes.dti_calculator import dti_calculator_node
    from nodes.property_valuation import property_valuation_node
    from nodes.risk_scoring import risk_scoring_node
    from nodes.decision_engine import decision_engine_node

    graph = StateGraph(LoanState)

    # Add all 7 nodes
    graph.add_node("ocr_extraction", ocr_extraction_node)
    graph.add_node("id_verification", id_verification_node)
    graph.add_node("credit_score", credit_score_node)
    graph.add_node("dti_calculator", dti_calculator_node)
    graph.add_node("property_valuation", property_valuation_node)
    graph.add_node("risk_scoring", risk_scoring_node)
    graph.add_node("decision_engine", decision_engine_node)

    # Connect them in sequence
    graph.set_entry_point("ocr_extraction")
    graph.add_edge("ocr_extraction", "id_verification")
    graph.add_edge("id_verification", "credit_score")
    graph.add_edge("credit_score", "dti_calculator")
    graph.add_edge("dti_calculator", "property_valuation")
    graph.add_edge("property_valuation", "risk_scoring")
    graph.add_edge("risk_scoring", "decision_engine")
    graph.add_edge("decision_engine", END)

    return graph.compile()