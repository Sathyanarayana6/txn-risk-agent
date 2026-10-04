from tools import get_customer_profile, check_watchlist, count_recent_txns


def review_transaction(txn):
    reasons = []

    # 1. Customer profile -> amount and country checks
    profile = get_customer_profile(txn["customer"])
    if "error" in profile:
        return {"verdict": "FLAG", "reasons": [profile["error"]]}

    try:
        amount = float(txn["amount"])
    except (ValueError, TypeError):
        return {"verdict": "FLAG", "reasons": ["invalid amount"]}

    ratio = amount / profile["avg_spend"]
    if ratio > 5:
        reasons.append(f"amount {ratio:.0f}x avg spend")

    if txn["country"] != profile["home_country"]:
        reasons.append("foreign country")

    # 2. Watchlist check
    if check_watchlist(txn["merchant"])["on_watchlist"]:
        reasons.append("merchant on watchlist")

    # 3. Velocity check
    recent = count_recent_txns(txn["customer"], 10)["count"]
    if recent >= 3:
        reasons.append(f"{recent} txns in 10 min")

    # 4. Verdict
    verdict = "FLAG" if reasons else "CLEAR"
    return {"verdict": verdict, "reasons": reasons}


if __name__ == "__main__":
    tests = [
        {"customer": "C1", "amount": 2400, "merchant": "CryptoFastCash LLC", "country": "NG"},
        {"customer": "C2", "amount": 450, "merchant": "Starbucks", "country": "US"},
        {"customer": "C99", "amount": 100, "merchant": "Starbucks", "country": "US"},
    ]
    for t in tests:
        print(t["customer"], "->", review_transaction(t))