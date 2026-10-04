import json
from collections import Counter

with open("traces.jsonl", encoding="utf-8") as f:
    traces = [json.loads(line) for line in f if line.strip()]

n = len(traces)
if n == 0:
    print("No traces yet. Run agent.py or eval.py first.")
    raise SystemExit

overrides = [t for t in traces if t["guardrail_override"]]

print(f"Runs:                 {n}")
print(f"Verdicts:             {dict(Counter(t['verdict'] for t in traces))}")
print(f"Guardrail overrides:  {len(overrides)} ({len(overrides) / n:.0%})")
print(f"Avg LLM calls/run:    {sum(t['llm_calls'] for t in traces) / n:.1f}")
print(f"Avg tokens/run:       {sum(t['input_tokens'] + t['output_tokens'] for t in traces) / n:.0f}")
print(f"Avg duration:         {sum(t['duration_ms'] for t in traces) / n:.0f} ms")

print("\nRuns where the guardrail overrode the LLM:")
for t in overrides:
    txn = t["txn"]
    reason = t["final_answer"].splitlines()[-1]
    print(f"  {t['run_id']}  {txn['customer']} {txn['country']}  "
          f"merchant={txn['merchant'][:20]}  note={'yes' if txn.get('note') else 'no'}")
    print(f"      -> {reason}")