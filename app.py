import os, sys
from typing import Optional, Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd

# Ensure local imports work when running from different CWDs
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

import decision_engine as de  # our previously generated rules engine

app = FastAPI(title="CNP Decision Service", version="1.0.0", description="Rules-based decisioning for card-not-present transactions")

# --- Request schema ---
RiskStr = Literal["low", "medium", "high", "new_domain"]
Reputation = Literal["trusted", "recurrent", "new", "high_risk"]
ProductType = Literal["digital", "physical", "subscription"]

class Transaction(BaseModel):
    transaction_id: Optional[int] = Field(None, description="Your own ID to track the decision")
    amount_mxn: float = 0.0
    customer_txn_30d: int = 0
    geo_state: Optional[str] = None
    device_type: Optional[str] = None
    chargeback_count: int = 0
    hour: int = 12
    product_type: ProductType = "digital"
    latency_ms: int = 0
    user_reputation: Reputation = "new"
    device_fingerprint_risk: RiskStr = "low"
    ip_risk: RiskStr = "low"
    email_risk: RiskStr = "low"
    bin_country: Optional[str] = "MX"
    ip_country: Optional[str] = "MX"

class DecisionResponse(BaseModel):
    transaction_id: Optional[int]
    decision: Literal["ACCEPTED", "IN_REVIEW", "REJECTED"]
    risk_score: int
    reasons: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/config")
def get_config():
    # Expose current thresholds for transparency
    return de.DEFAULT_CONFIG

@app.post("/transaction", response_model=DecisionResponse)
def evaluate_transaction(txn: Transaction):
    # Convert the validated model to a pandas Series and score with our engine
    row = pd.Series(txn.model_dump())
    res = de.assess_row(row, de.DEFAULT_CONFIG)
    return {
        "transaction_id": txn.transaction_id,
        "decision": res["decision"],
        "risk_score": int(res["risk_score"]),
        "reasons": res.get("reasons", ""),
    }
