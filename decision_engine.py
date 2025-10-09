import argparse
import pandas as pd
from typing import Dict, Any, List, Tuple

DECISION_ACCEPTED = "ACCEPTED"
DECISION_IN_REVIEW = "IN_REVIEW"
DECISION_REJECTED = "REJECTED"

DEFAULT_CONFIG = {
    "amount_thresholds": {
        "digital": 2500,
        "physical": 6000,
        "subscription": 1500,
        "_default": 4000
    },
    "latency_ms_extreme": 2500,
    "chargeback_hard_block": 2,
    "score_weights": {
        "ip_risk": {"low": 0, "medium": 2, "high": 4},
        "email_risk": {"low": 0, "medium": 1, "high": 3, "new_domain": 2},
        "device_fingerprint_risk": {"low": 0, "medium": 2, "high": 4},
        "user_reputation": {"trusted": -2, "recurrent": -1, "new": 0, "high_risk": 4},
        "night_hour": 1,
        "geo_mismatch": 2,
        "high_amount": 2,
        "latency_extreme": 2,
        "new_user_high_amount": 2,
    },
    "score_to_decision": {
        "reject_at": 10,
        "review_at": 4
    }
}

# Optional: override thresholds via environment variables (for CI/CD / canary tuning)
try:
    import os as _os
    _rej = _os.getenv("REJECT_AT")
    _rev = _os.getenv("REVIEW_AT")
    if _rej is not None:
        DEFAULT_CONFIG["score_to_decision"]["reject_at"] = int(_rej)
    if _rev is not None:
        DEFAULT_CONFIG["score_to_decision"]["review_at"] = int(_rev)
except Exception:
    pass

def is_night(hour: int) -> bool:
    return hour >= 22 or hour <= 5

def high_amount(amount: float, product_type: str, thresholds: Dict[str, Any]) -> bool:
    t = thresholds.get(product_type, thresholds.get("_default"))
    return amount >= t

def check_hard_block(row: pd.Series, cfg: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    We check for the immediate hard-block condition:
    repeated chargebacks AND high IP risk.
    """
    if int(row.get("chargeback_count", 0)) >= cfg["chargeback_hard_block"] and \
       str(row.get("ip_risk", "low")).lower() == "high":
        reasons = ["hard_block:chargebacks>=2+ip_high"]
        return {"decision": DECISION_REJECTED, "risk_score": 100, "reasons": ";".join(reasons)}
    return None


def calculate_categorical_scores(row: pd.Series, cfg: Dict[str, Any]) -> Tuple[int, List[str]]:
    """
    We calculate the combined score and reasons for ip_risk, email_risk, and
    device_fingerprint_risk.
    """
    score = 0
    reasons: List[str] = []

    # Define fields and mappings outside the loop for clarity
    categorical_checks = [
        ("ip_risk", cfg["score_weights"]["ip_risk"]),
        ("email_risk", cfg["score_weights"]["email_risk"]),
        ("device_fingerprint_risk", cfg["score_weights"]["device_fingerprint_risk"])
    ]

    for field, mapping in categorical_checks:
        val = str(row.get(field, "low")).lower()
        add = mapping.get(val, 0)
        score += add
        if add:
            reasons.append(f"{field}:{val}(+{add})")

    return score, reasons


def score_user_reputation(row: pd.Series, cfg: Dict[str, Any]) -> Tuple[int, str]:
    """We calculate score and reason for user reputation."""
    rep = str(row.get("user_reputation", "new")).lower()
    rep_add = cfg["score_weights"]["user_reputation"].get(rep, 0)

    reason = ""
    if rep_add:
        reason = f"user_reputation:{rep}({('+' if rep_add >= 0 else '')}{rep_add})"

    return rep_add, reason

def score_night_hour(row: pd.Series, cfg: Dict[str, Any]) -> Tuple[int, str]:
    """We calculate score and reason for night hour transaction."""
    hr = int(row.get("hour", 12))

    if is_night(hr):
        add = cfg["score_weights"]["night_hour"]
        return add, f"night_hour:{hr}(+{add})"

    return 0, ""


def score_geo_mismatch(row: pd.Series, cfg: Dict[str, Any]) -> Tuple[int, str]:
    """We calculate score and reason for BIN/IP country mismatch."""
    bin_c = str(row.get("bin_country", "")).upper()
    ip_c  = str(row.get("ip_country", "")).upper()

    if bin_c and ip_c and bin_c != ip_c:
        add = cfg["score_weights"]["geo_mismatch"]
        return add, f"geo_mismatch:{bin_c}!={ip_c}(+{add})"

    return 0, ""


def score_high_amount_base(row: pd.Series, cfg: Dict[str, Any]) -> Tuple[int, str]:
    """We calculate base score for high amount for product type."""
    amount = float(row.get("amount_mxn", 0.0))
    ptype = str(row.get("product_type", "_default")).lower()

    if high_amount(amount, ptype, cfg["amount_thresholds"]):
        add = cfg["score_weights"]["high_amount"]
        return add, f"high_amount:{ptype}:{amount}(+{add})"

    return 0, ""


def score_new_user_multiplier(row: pd.Series, cfg: Dict[str, Any]) -> Tuple[int, str]:
    """
    We calculate the additional score if the transaction is a high amount
    AND the user is 'new'. (Handles the previously nested logic.)
    """
    amount = float(row.get("amount_mxn", 0.0))
    ptype = str(row.get("product_type", "_default")).lower()
    rep = str(row.get("user_reputation", "new")).lower()

    # Check both conditions that were originally nested
    if rep == "new" and high_amount(amount, ptype, cfg["amount_thresholds"]):
        add = cfg["score_weights"]["new_user_high_amount"]
        return add, f"new_user_high_amount(+{add})"

    return 0, ""


def score_extreme_latency(row: pd.Series, cfg: Dict[str, Any]) -> Tuple[int, str]:
    """We calculate score and reason for extreme latency."""
    lat = int(row.get("latency_ms", 0))

    if lat >= cfg["latency_ms_extreme"]:
        add = cfg["score_weights"]["latency_extreme"]
        return add, f"latency_extreme:{lat}ms(+{add})"

    return 0, ""


def get_final_decision(score: int, cfg: Dict[str, Any]) -> str:
    """We map the final score to a decision."""
    if score >= cfg["score_to_decision"]["reject_at"]:
        return DECISION_REJECTED
    elif score >= cfg["score_to_decision"]["review_at"]:
        return DECISION_IN_REVIEW
    else:
        return DECISION_ACCEPTED

def assess_row(row: pd.Series, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    We assess a transaction row, orchestrating risk checks via helper functions.
    The Cognitive Complexity of this main function is significantly reduced.
    """
    # 1. Hard block: immediate exit check
    hard_block_result = check_hard_block(row, cfg)
    if hard_block_result:
        return hard_block_result

    score = 0
    reasons: List[str] = []

    # 2. Categorical risks (one call for all three)
    add, new_reasons = calculate_categorical_scores(row, cfg)
    score += add
    reasons.extend(new_reasons)

    # 3. Individual risk factors (sequential calls)
    # The list makes the scoring logic declarative and easy to extend.
    scoring_functions = [
        score_user_reputation,
        score_night_hour,
        score_geo_mismatch,
        score_high_amount_base,
        score_new_user_multiplier,
        score_extreme_latency,
    ]

    for func in scoring_functions:
        add, reason = func(row, cfg)
        score += add
        if reason:
            reasons.append(reason)

    # 4. Frequency buffer for trusted/recurrent (Post-scoring adjustment)
    rep = str(row.get("user_reputation", "new")).lower()
    freq = int(row.get("customer_txn_30d", 0))
    if rep in ("recurrent", "trusted") and freq >= 3 and score > 0:
        score -= 1
        reasons.append("frequency_buffer(-1)")

    # 5. Decision mapping
    decision = get_final_decision(score, cfg)

    return {"decision": decision, "risk_score": int(score), "reasons": ";".join(reasons)}


def run(input_csv: str, output_csv: str, config: Dict[str, Any] = None) -> pd.DataFrame:
    cfg = config or DEFAULT_CONFIG
    df = pd.read_csv(input_csv)

    # 💥 Optimización: Aplicamos la función a cada fila
    results_series = df.apply(lambda row: assess_row(row, cfg), axis=1)

    # Desempaquetamos los resultados en columnas
    out = df.copy()
    out["decision"] = results_series.apply(lambda r: r["decision"])
    out["risk_score"] = results_series.apply(lambda r: r["risk_score"])
    out["reasons"] = results_series.apply(lambda r: r["reasons"])

    out.to_csv(output_csv, index=False)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=False, default="transactions_examples.csv", help="Path to input CSV")
    ap.add_argument("--output", required=False, default="decisions.csv", help="Path to output CSV")
    args = ap.parse_args()
    out = run(args.input, args.output)
    print(out.head().to_string(index=False))

if __name__ == "__main__":
    main()
