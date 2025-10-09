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


# --- AUXILIARY HELPERS ---

def is_night(hour: int) -> bool:
    """Checks if the hour falls within night hours (22 to 5)."""
    return hour >= 22 or hour <= 5


def high_amount(amount: float, product_type: str, thresholds: Dict[str, Any]) -> bool:
    """Checks if the transaction amount exceeds the threshold for the product type."""
    t = thresholds.get(product_type, thresholds.get("_default"))
    return amount >= t


# --- SCORING HELPERS ---

def check_hard_block(row: pd.Series, cfg: Dict[str, Any]) -> Dict[str, Any] | None:
    """Handles the immediate hard-block condition: chargebacks AND high IP risk."""
    if int(row.get("chargeback_count", 0)) >= cfg["chargeback_hard_block"] and \
            str(row.get("ip_risk", "low")).lower() == "high":
        reasons = ["hard_block:chargebacks>=2+ip_high"]
        return {"decision": DECISION_REJECTED, "risk_score": 100, "reasons": ";".join(reasons)}
    return None


def calculate_categorical_scores(row: pd.Series, cfg: Dict[str, Any]) -> Tuple[int, List[str]]:
    """Calculates combined score and reasons for categorical risk fields."""
    score = 0
    reasons: List[str] = []

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
    """Calculates score and reason for user reputation."""
    rep = str(row.get("user_reputation", "new")).lower()
    rep_add = cfg["score_weights"]["user_reputation"].get(rep, 0)

    reason = ""
    if rep_add:
        reason = f"user_reputation:{rep}({('+' if rep_add >= 0 else '')}{rep_add})"

    return rep_add, reason


def score_night_hour(row: pd.Series, cfg: Dict[str, Any]) -> Tuple[int, str]:
    """Calculates score and reason for night hour transaction."""
    hr = int(row.get("hour", 12))

    if is_night(hr):
        add = cfg["score_weights"]["night_hour"]
        return add, f"night_hour:{hr}(+{add})"

    return 0, ""


def score_geo_mismatch(row: pd.Series, cfg: Dict[str, Any]) -> Tuple[int, str]:
    """Calculates score and reason for BIN/IP country mismatch."""
    bin_c = str(row.get("bin_country", "")).upper()
    ip_c = str(row.get("ip_country", "")).upper()

    if bin_c and ip_c and bin_c != ip_c:
        add = cfg["score_weights"]["geo_mismatch"]
        return add, f"geo_mismatch:{bin_c}!={ip_c}(+{add})"

    return 0, ""


def score_high_amount_base(row: pd.Series, cfg: Dict[str, Any]) -> Tuple[int, str]:
    """Calculates base score for high amount for product type."""
    amount = float(row.get("amount_mxn", 0.0))
    ptype = str(row.get("product_type", "_default")).lower()

    if high_amount(amount, ptype, cfg["amount_thresholds"]):
        add = cfg["score_weights"]["high_amount"]
        return add, f"high_amount:{ptype}:{amount}(+{add})"

    return 0, ""


def score_new_user_multiplier(row: pd.Series, cfg: Dict[str, Any]) -> Tuple[int, str]:
    """
    Calculates the additional score if the transaction is a high amount
    AND the user is 'new'.
    """
    amount = float(row.get("amount_mxn", 0.0))
    ptype = str(row.get("product_type", "_default")).lower()
    rep = str(row.get("user_reputation", "new")).lower()

    if rep == "new" and high_amount(amount, ptype, cfg["amount_thresholds"]):
        add = cfg["score_weights"]["new_user_high_amount"]
        return add, f"new_user_high_amount(+{add})"

    return 0, ""


def score_extreme_latency(row: pd.Series, cfg: Dict[str, Any]) -> Tuple[int, str]:
    """Calculates score and reason for extreme latency."""
    lat = int(row.get("latency_ms", 0))

    if lat >= cfg["latency_ms_extreme"]:
        add = cfg["score_weights"]["latency_extreme"]
        return add, f"latency_extreme:{lat}ms(+{add})"

    return 0, ""


def get_final_decision(score: int, cfg: Dict[str, Any]) -> str:
    """Maps the final score to a decision."""
    if score >= cfg["score_to_decision"]["reject_at"]:
        return DECISION_REJECTED
    elif score >= cfg["score_to_decision"]["review_at"]:
        return DECISION_IN_REVIEW
    else:
        return DECISION_ACCEPTED


# --- MAIN ASSESSOR ---

def assess_row(row: pd.Series, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assesses a transaction row by orchestrating calls to small, focused helper functions.
    Low Cognitive Complexity is maintained here.
    """
    # 1. Hard block: immediate exit check
    hard_block_result = check_hard_block(row, cfg)
    if hard_block_result:
        return hard_block_result

    score = 0
    reasons: List[str] = []

    # 2. Categorical risks (one combined call)
    add, new_reasons = calculate_categorical_scores(row, cfg)
    score += add
    reasons.extend(new_reasons)

    # 3. Individual risk factors (sequential calls and accumulation)
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


# --- EXECUTION LOGIC (Optimized) ---

def run(input_csv: str, output_csv: str, config: Dict[str, Any] = None) -> pd.DataFrame:
    """Reads CSV, processes transactions using pandas.apply for efficiency, and saves results."""
    cfg = config or DEFAULT_CONFIG
    df = pd.read_csv(input_csv)

    # OPTIMIZATION: Use pandas.apply(axis=1) for faster processing than iterrows()
    results_series = df.apply(lambda row: assess_row(row, cfg), axis=1)

    # Desempaquetar los resultados de la Serie en nuevas columnas del DataFrame
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

    # Intenta buscar un archivo de ejemplo si no existe uno
    try:
        # Se asume que 'transactions_examples.csv' existe o se puede crear/descargar si es necesario
        # Para un código ejecutable, se necesitaría un archivo real aquí.
        # Por ahora, se asume que run() funcionará si el archivo existe.
        out = run(args.input, args.output)
        print("--- Decisions Head ---")
        print(out.head().to_string(index=False))
        print(f"\nResults saved to {args.output}")
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found. Please provide a valid CSV.")
    except Exception as e:
        print(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    main()