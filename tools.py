from datetime import datetime, timedelta
from data import customers, watchlist, history

# Fixed "current time" so the fake data gives sensible results
NOW = datetime(2026, 9, 29, 10, 5)


def get_customer_profile(customer_id):
    profile = customers.get(customer_id)
    if profile is None:
        return {"error": f"customer {customer_id} not found"}
    return profile


def check_watchlist(merchant):
    normalized = merchant.strip().lower()
    hit = normalized in {m.lower() for m in watchlist}
    return {"merchant": merchant, "on_watchlist": hit}


def count_recent_txns(customer_id, minutes):
    if customer_id not in history:
        return {"error": f"customer {customer_id} not found"}

    cutoff = NOW - timedelta(minutes=minutes)
    count = 0
    for ts in history[customer_id]:
        txn_time = datetime.strptime(ts, "%Y-%m-%d %H:%M")
        if txn_time >= cutoff:
            count += 1
    return {"customer_id": customer_id, "minutes": minutes, "count": count}


def check_amount(customer_id, amount):
    profile = customers.get(customer_id)
    if profile is None:
        return {"error": f"customer {customer_id} not found"}
    try:
        amount = float(amount)
    except (ValueError, TypeError):
        return {"error": f"invalid amount {amount}"}
    ratio = amount / profile["avg_spend"]
    return {"ratio": round(ratio, 2), "over_5x": ratio > 5}


if __name__ == "__main__":
    print(get_customer_profile("C1"))
    print(get_customer_profile("C99"))
    print(check_watchlist("CryptoFastCash LLC"))
    print(check_watchlist("cryptofastcash llc"))
    print(count_recent_txns("C1", 10))
    print(check_amount("C2", 2500))
    print(check_amount("C2", "5000"))