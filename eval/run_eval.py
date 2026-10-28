"""DoDiligence accuracy benchmark.

Runs the real pipeline (eyecite -> CourtListener lookup -> quote matching -> Claude judge)
over a labeled set of citations and reports how often it gets them right.

The CourtListener responses were recorded once (eval/data/recording.json.gz), so every run
sees the same opinions and the numbers are reproducible. Claude verdicts are cached in
eval/data/judgments.json; delete that file to re-run the AI step from scratch.

Usage:
    ANTHROPIC_API_KEY=sk-ant-... python -m eval.run_eval            # full run
    python -m eval.run_eval --no-ai                                  # existence + quote checks only
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.judge import ClaudeJudge, DEFAULT_MODEL  # noqa: E402
from core.pipeline import verify  # noqa: E402
from core.sources import CourtListenerSource, ReplayTransport  # noqa: E402

DATA = ROOT / "eval" / "data"
FLAGS = {"NOT_FOUND", "NAME_MISMATCH", "MISQUOTED", "CONTRADICTED", "UNSUPPORTED"}
OK = {"VERIFIED", "QUOTE_OK"}
# What a correct answer looks like for each group.
RIGHT_TYPE = {
    "fabricated": {"NOT_FOUND", "NAME_MISMATCH"},
    "false_quote": {"MISQUOTED"},
    "misrepresented": {"CONTRADICTED", "UNSUPPORTED", "MISQUOTED"},
    "control": OK,
}
GROUP_LABEL = {
    "fabricated": "Fake or wrong-case citations",
    "false_quote": "Invented quotations",
    "misrepresented": "Real case, misstated holding",
    "control": "Correct citations written by judges",
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - r) / d), min(1.0, (c + r) / d))


def load():
    items = json.loads((DATA / "items.json").read_text(encoding="utf-8"))
    with gzip.open(DATA / "recording.json.gz", "rt", encoding="utf-8") as f:
        rec = json.load(f)
    return items, rec


def run(use_ai: bool, workers: int = 6, limit: int | None = None) -> dict:
    items, rec = load()
    if limit:
        items = items[:limit]
    source = CourtListenerSource(transport=ReplayTransport(rec))
    judge = None
    store_path = DATA / "judgments.json"
    if use_ai:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            sys.exit("Set ANTHROPIC_API_KEY (or use --no-ai).")
        store = json.loads(store_path.read_text()) if store_path.exists() else {}
        judge = ClaudeJudge(key, os.environ.get("CLAUDE_MODEL", DEFAULT_MODEL), store)

    def one(it):
        t0 = time.time()
        res = verify(it["text"], source, judge)
        r = res[0] if res else None
        return {
            "id": it["id"], "group": it["group"], "split": it.get("split", ""),
            "status": r.status if r else "NO_CITATION",
            "reason": r.reason if r else "",
            "verdict": r.judgment.verdict if r and r.judgment else None,
            "found_case": r.opinion.case_name if r else "",
            "seconds": round(time.time() - t0, 2),
        }

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, items))
    if judge is not None:
        store_path.write_text(json.dumps(judge.store, indent=0))
    return summarize(items, rows, judge, use_ai)


def summarize(items, rows, judge, use_ai) -> dict:
    by_group = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    groups = {}
    for g, rs in by_group.items():
        n = len(rs)
        if g == "control":
            ok = sum(r["status"] in OK for r in rs)
            false_alarm = sum(r["status"] in FLAGS for r in rs)
            review = sum(r["status"] not in OK and r["status"] not in FLAGS for r in rs)
            held = [r for r in rs if r.get("split") == "heldout"]
            hok = sum(r["status"] in OK for r in held)
            groups[g] = {"label": GROUP_LABEL[g], "n": n, "passed": ok, "pass_rate": ok / n,
                         "heldout_n": len(held), "heldout_passed": hok,
                         "heldout_false_alarms": sum(r["status"] in FLAGS for r in held),
                         "pass_ci": wilson(ok, n), "false_alarms": false_alarm, "needs_review": review,
                         "statuses": Counter(r["status"] for r in rs)}
        else:
            caught = sum(r["status"] in FLAGS for r in rs)
            review = sum(r["status"] not in OK and r["status"] not in FLAGS for r in rs)
            right = sum(r["status"] in RIGHT_TYPE[g] for r in rs)
            groups[g] = {"label": GROUP_LABEL[g], "n": n, "caught": caught, "catch_rate": caught / n,
                         "catch_ci": wilson(caught, n), "needs_review": review,
                         "caught_or_review": (caught + review) / n,
                         "right_type": right, "missed": n - caught - review,
                         "statuses": Counter(r["status"] for r in rs)}
    bad = [r for r in rows if r["group"] != "control"]
    good = [r for r in rows if r["group"] == "control"]
    tp = sum(r["status"] in FLAGS for r in bad)
    fp = sum(r["status"] in FLAGS for r in good)
    fn = len(bad) - tp
    tn = len(good) - fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / len(bad) if bad else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + sum(r["status"] in OK for r in good)) / len(rows) if rows else 0.0
    cost = None
    if judge is not None:
        cost = {"model": judge.model, "api_calls": judge.calls, "input_tokens": judge.input_tokens,
                "output_tokens": judge.output_tokens}
    return {
        "ai_judge": use_ai, "n_items": len(rows), "groups": groups,
        "overall": {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "precision_ci": wilson(tp, tp + fp), "recall_ci": wilson(tp, len(bad))},
        "cost": cost,
        "median_seconds": sorted(r["seconds"] for r in rows)[len(rows) // 2] if rows else 0,
        "rows": rows,
    }


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def report_md(full: dict, no_ai: dict | None) -> str:
    o = full["overall"]
    full["groups"].setdefault("control", {"label": GROUP_LABEL["control"], "n": 0, "passed": 0, "pass_rate": 0.0,
                                          "false_alarms": 0, "needs_review": 0, "statuses": {}})
    L = ["# DoDiligence accuracy results", "",
         f"{full['n_items']} citations: {full['n_items'] - full['groups']['control']['n']} that US courts "
         f"found to be fake, misquoted or misrepresented in real filings, and "
         f"{full['groups']['control']['n']} correct citations written by judges (taken from the opinions in the test set).", "",
         "| | Result | 95% CI |", "|---|---|---|",
         f"| Problems caught (recall) | {pct(o['recall'])} ({o['tp']}/{o['tp'] + o['fn']}) | "
         f"{pct(o['recall_ci'][0])}–{pct(o['recall_ci'][1])} |",
         f"| Flags that were real problems (precision) | {pct(o['precision'])} ({o['tp']}/{o['tp'] + o['fp']}) | "
         f"{pct(o['precision_ci'][0])}–{pct(o['precision_ci'][1])} |",
         f"| F1 | {o['f1']:.3f} | |", "", "## By error type", "",
         "| Group | Items | Caught | Right error type | Sent to human review | Missed |",
         "|---|---|---|---|---|---|"]
    for g in ("fabricated", "false_quote", "misrepresented"):
        d = full["groups"].get(g)
        if d:
            L.append(f"| {d['label']} | {d['n']} | {pct(d['catch_rate'])} | {d['right_type']} | "
                     f"{d['needs_review']} | {d['missed']} |")
    c = full["groups"]["control"]
    L += ["", "| Correct citations | Items | Passed | False alarms | Sent to human review |",
          "|---|---|---|---|---|",
          f"| {c['label']} | {c['n']} | {pct(c['pass_rate'])} | {c['false_alarms']} | {c['needs_review']} |"]
    if c.get("heldout_n"):
        L += ["", f"Held-out check: of the {c['heldout_n']} correct citations collected after DoDiligence was last "
                  f"adjusted, {c['heldout_passed']} passed ({pct(c['heldout_passed'] / c['heldout_n'])}) and "
                  f"{c['heldout_false_alarms']} were wrongly flagged."]
    if no_ai:
        L += ["", "## What the AI step adds", "",
              "Same items, with the Claude check turned off (existence and quote checks only):", "",
              "| | With Claude | Without |", "|---|---|---|"]
        for g in ("fabricated", "false_quote", "misrepresented"):
            if g in full["groups"]:
                L.append(f"| {GROUP_LABEL[g]} caught | {pct(full['groups'][g]['catch_rate'])} | "
                         f"{pct(no_ai['groups'][g]['catch_rate'])} |")
    if full["cost"]:
        k = full["cost"]
        n = max(1, k["api_calls"])
        L += ["", "## AI step", "",
              f"{k['api_calls']} Claude calls ({k['model']}), averaging {k['input_tokens'] / n:,.0f} input and "
              f"{k['output_tokens'] / n:,.0f} output tokens per check: each check reads about five passages, "
              "not the whole opinion. Everything else runs locally on the CPU; see `python -m eval.perf`."]
    L += ["", "## Status counts", "", "| Group | " + " | ".join(
        ["VERIFIED", "QUOTE_OK", "NOT_FOUND", "NAME_MISMATCH", "MISQUOTED", "CONTRADICTED", "UNSUPPORTED", "REVIEW"]) + " |",
          "|---" * 9 + "|"]
    for g in ("fabricated", "false_quote", "misrepresented", "control"):
        if g in full["groups"]:
            s = full["groups"][g]["statuses"]
            L.append(f"| {GROUP_LABEL[g]} | " + " | ".join(str(s.get(k, 0)) for k in
                     ["VERIFIED", "QUOTE_OK", "NOT_FOUND", "NAME_MISMATCH", "MISQUOTED", "CONTRADICTED",
                      "UNSUPPORTED", "REVIEW"]) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ai", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.no_ai:
        res = run(False, a.workers, a.limit)
        (DATA / "results_no_ai.json").write_text(json.dumps(res, indent=1, default=list))
        print(report_md(res, None))
    else:
        base = run(False, a.workers, a.limit)
        res = run(True, a.workers, a.limit)
        (DATA / "results_no_ai.json").write_text(json.dumps(base, indent=1, default=list))
        (DATA / "results.json").write_text(json.dumps(res, indent=1, default=list))
        md = report_md(res, base)
        (ROOT / "eval" / "RESULTS.md").write_text(md)
        print(md)
