"""How fast and light DoDiligence is.

Times the whole pipeline on every benchmark item, single-threaded, using the recorded
CourtListener responses and cached Claude verdicts, so it measures DoDiligence's own work
(finding citations, matching quotes, picking passages) without network time. The AI step's
latency comes from the benchmark's recorded live runs. Writes data/perf.json for the Accuracy tab.

Usage:
    python -m eval.perf
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.judge import ClaudeJudge  # noqa: E402
from core.pipeline import verify  # noqa: E402
from core.sources import CourtListenerSource, ReplayTransport, load_recording  # noqa: E402

DATA = ROOT / "eval" / "data"


def rss_mb() -> float | None:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        return None


def main() -> dict:
    items = json.loads((DATA / "items.json").read_text(encoding="utf-8"))
    source = CourtListenerSource(transport=ReplayTransport(load_recording()))
    judge = ClaudeJudge("cache-only", store=json.loads((DATA / "judgments.json").read_text()))
    for it in items[:5]:                       # warm up eyecite and the regex caches
        verify(it["text"], source, judge)
    times = []
    for it in items:
        t = time.perf_counter()
        verify(it["text"], source, judge)
        times.append(time.perf_counter() - t)
    times.sort()
    res = json.loads((DATA / "results.json").read_text(encoding="utf-8"))
    ai = sorted(r["seconds"] for r in res["rows"] if r["verdict"])
    k = res.get("cost") or {}
    calls = max(1, k.get("api_calls", 1))
    out = {
        "items": len(times),
        "median_ms": round(statistics.median(times) * 1000, 1),
        "p95_ms": round(times[int(0.95 * len(times))] * 1000, 1),
        "total_s": round(sum(times), 2),
        "memory_mb": round(rss_mb()) if rss_mb() else None,
        "cpu_cores_used": 1,
        "ai_checks": len(ai),
        "ai_median_s": round(ai[len(ai) // 2], 1) if ai else None,
        "ai_input_tokens": round(k.get("input_tokens", 0) / calls),
        "ai_output_tokens": round(k.get("output_tokens", 0) / calls),
        "machine": f"{os.cpu_count()}-thread CPU, Python {sys.version.split()[0]}",
    }
    (ROOT / "data" / "perf.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
