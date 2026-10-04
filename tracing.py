import json
import time
import uuid
from datetime import datetime

TRACE_FILE = "traces.jsonl"


class Trace:
    def __init__(self, txn):
        self._start = time.perf_counter()
        self.data = {
            "run_id": uuid.uuid4().hex[:8],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "txn": txn,
            "events": [],
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    def log(self, kind, **details):
        elapsed = round((time.perf_counter() - self._start) * 1000)
        self.data["events"].append({"kind": kind, "ms": elapsed, **details})

    def add_usage(self, usage):
        self.data["llm_calls"] += 1
        if usage:
            self.data["input_tokens"] += usage.prompt_tokens
            self.data["output_tokens"] += usage.completion_tokens

    def finish(self, final_answer, model_answer, overridden):
        self.data["model_answer"] = model_answer
        self.data["final_answer"] = final_answer
        self.data["verdict"] = "FLAG" if "VERDICT: FLAG" in final_answer.upper() else "CLEAR"
        self.data["guardrail_override"] = overridden
        self.data["duration_ms"] = round((time.perf_counter() - self._start) * 1000)
        with open(TRACE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.data) + "\n")