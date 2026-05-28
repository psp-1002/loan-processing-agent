from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

class LoanState(TypedDict):
    applicant_name: str
    monthly_income: float
    loan_amount: float
    loan_purpose: str
    property_value: Optional[float]
    ocr_text: Optional[str]
    id_verified: Optional[bool]
    credit_score: Optional[int]
    dti_ratio: Optional[float]
    property_valuation: Optional[str]
    risk_score: Optional[str]
    risk_label: Optional[str]
    risk_justification: Optional[str]
    similar_loans_count: Optional[int]
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

    graph.add_node("ocr_extraction", ocr_extraction_node)
    graph.add_node("id_verification", id_verification_node)
    graph.add_node("credit_score", credit_score_node)
    graph.add_node("dti_calculator", dti_calculator_node)
    graph.add_node("property_valuation", property_valuation_node)
    graph.add_node("risk_scoring", risk_scoring_node)
    graph.add_node("decision_engine", decision_engine_node)

    graph.set_entry_point("ocr_extraction")
    graph.add_edge("ocr_extraction", "id_verification")
    graph.add_edge("id_verification", "credit_score")
    graph.add_edge("credit_score", "dti_calculator")
    graph.add_edge("dti_calculator", "property_valuation")
    graph.add_edge("property_valuation", "risk_scoring")
    graph.add_edge("risk_scoring", "decision_engine")
    graph.add_edge("decision_engine", END)

    return graph.compile()

if __name__ == "__main__":
    app = build_graph()

    result = app.invoke({
        "applicant_name": "Sourav Prusty",
        "monthly_income": 5000,
        "loan_amount": 200000,
        "loan_purpose": "Home purchase",
        "property_value": 250000,
        "ocr_text": None,
        "id_verified": None,
        "credit_score": None,
        "dti_ratio": None,
        "property_valuation": None,
        "risk_score": None,
        "risk_label": None,
        "risk_justification": None,
        "similar_loans_count": None,
        "final_decision": None,
        "escalate_to_human": None
    })

    print("\n" + "="*50)
    print("LANGGRAPH PIPELINE COMPLETE")
    print("="*50)
    print(f"Credit Score : {result['credit_score']}")
    print(f"DTI Ratio    : {result['dti_ratio']}%")
    print(f"Property     : {result['property_valuation']}")
    print(f"Risk         : {result['risk_label']}")
    print(f"Rationale    : {result['risk_justification']}")
    print(f"Similar Loans: {result['similar_loans_count']}")
    print(f"Decision     : {result['final_decision']}")
    print("="*50)