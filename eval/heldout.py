"""Held-out check: correct citations chosen after DoDiligence's rules were frozen.

The benchmark's 14 correct citations were looked at while the rules were tuned. This script builds a
separate set the rules have never seen, from the same pool (sentences in which judges cite a case, in the
opinions recorded for the benchmark), and measures how many pass. It keeps its own files, so the main
benchmark's data is untouched:

    eval/data/heldout_items.json          the selected citations (written once, then frozen)
    eval/data/heldout_recording.json.gz   CourtListener responses for them
    eval/data/heldout_judgments.json      support-check verdicts (cache)
    eval/data/heldout_results.json        the result

Usage:
    python -m eval.heldout select     # pick the set (refuses to overwrite an existing one)
    python -m eval.heldout record     # fetch from CourtListener (needs COURTLISTENER_TOKEN; resumable)
    python -m eval.heldout run        # run the checks (needs ANTHROPIC_API_KEY for the support check)

Keys come from the environment or .streamlit/secrets.toml.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import re
import sys
import tomllib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eyecite import get_citations  # noqa: E402

from core.extract import extract_claims  # noqa: E402
from core.judge import DEFAULT_MODEL, ClaudeJudge  # noqa: E402
from core.pipeline import verify  # noqa: E402
from core.sources import (CourtListenerSource, HttpTransport, ReplayTransport,  # noqa: E402
                          _strip_html, load_recording)
from eval.run_eval import FLAGS, OK, wilson  # noqa: E402

DATA = ROOT / "eval" / "data"
ITEMS = DATA / "heldout_items.json"
RECORDING = DATA / "heldout_recording.json.gz"
JUDGMENTS = DATA / "heldout_judgments.json"
RESULTS = DATA / "heldout_results.json"
N_ITEMS = 50
SEED = 20261029
# Same filters as the benchmark's correct citations: one full case citation, a real proposition,
# and no signal that the court is distinguishing or disagreeing with the case.
SIGNALS = re.compile(r"\b(but see|cf\.|contra|compare|see generally|overruled|abrogated)\b", re.I)
OTHER_CITES = {"FullCaseCitation", "ShortCaseCitation", "IdCitation", "SupraCitation", "ReferenceCitation"}


def secret(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    p = ROOT / ".streamlit" / "secrets.toml"
    return tomllib.loads(p.read_text(encoding="utf-8")).get(name, "") if p.exists() else ""


def sentence_start(text: str) -> str | None:
    """The claim text starting at a sentence boundary (the extractor starts right after the previous
    citation, which can be mid-sentence)."""
    if re.match(r"[\"“A-Z]", text):
        return text
    m = re.search(r"[.;:]\s+(?=[\"“A-Z])", text)
    return text[m.end():] if m else None


def candidates(rec: dict, used: set[str]) -> dict[str, list]:
    out: dict[str, list] = {}
    for url, op in rec["opinions"].items():
        text = op.get("plain_text") or _strip_html(op.get("html_with_citations") or "")
        for para in re.split(r"\n\s*\n", text):
            para = " ".join(para.split())
            if not 80 <= len(para) <= 4000:
                continue
            for c in extract_claims(para):
                claim = sentence_start(c.claim_text)
                if not claim:
                    continue
                body = claim[: len(claim) - (c.cite_span[1] - c.cite_span[0])]
                key = re.sub(r"\W+", "", claim)[:80]
                parens = re.findall(r"[()]", body)
                if not (80 <= len(claim) <= 600 and len(body.strip()) >= 50 and " v. " in c.case_name
                        and not SIGNALS.search(claim) and key not in used
                        and not (parens and parens[0] == ")")):     # starts inside the previous citation
                    continue
                if any(type(x).__name__ in OTHER_CITES for x in get_citations(body)):
                    continue                                        # one case citation per passage
                alone = extract_claims(claim)                       # must read the same on its own
                if len(alone) == 1 and alone[0].citation == c.citation:
                    out.setdefault(url, []).append((claim, c))
    return out


def select() -> None:
    if ITEMS.exists():
        sys.exit(f"{ITEMS.name} already exists; the held-out set is frozen once chosen.")
    rec = load_recording()
    items = json.loads((DATA / "items.json").read_text(encoding="utf-8"))
    used = {re.sub(r"\W+", "", i["text"])[:80] for i in items}
    # sub-opinion URL -> the case it belongs to, for describing where each sentence came from
    source = {u: cl for v in rec["lookups"].values() for cl in (v.get("clusters") or [])
              for u in (cl.get("sub_opinions") or [])}
    cands = candidates(rec, used)
    rng = random.Random(SEED)
    urls = sorted(cands)
    rng.shuffle(urls)
    picked = []
    for url in urls[:N_ITEMS]:                     # one citation per source opinion
        claim, c = rng.choice(cands[url])
        cl = source.get(url, {})
        name, date = cl.get("case_name", ""), (cl.get("date_filed") or "")
        picked.append({
            "id": f"hold-{len(picked):03d}", "group": "control", "split": "heldout",
            "cite": c.citation, "name": c.case_name, "text": claim, "expected": "ok",
            "decision": name, "date": date, "court": "",
            "court_finding": f"A correct citation, written by the court in {name} ({date[:4]})." if name
                             else "A correct citation, written by a court.",
            "document": "https://www.courtlistener.com" + cl["absolute_url"] if cl.get("absolute_url") else "",
            "source_opinion": url,
        })
    ITEMS.write_text(json.dumps(picked, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Selected {len(picked)} held-out correct citations from {len(cands)} source opinions "
          f"({sum(map(len, cands.values()))} candidates), seed {SEED} -> {ITEMS.relative_to(ROOT)}")


class Recorder:
    """Wraps the live transport and keeps every response, in the same format as the main recording."""

    def __init__(self, live, rec: dict):
        self.live, self.rec = live, rec

    def lookup(self, text: str):
        status, body = self.live.lookup(text)
        if status == 200 and isinstance(body, list):
            for item in body:
                for k in [item.get("citation", "")] + list(item.get("normalized_citations") or []):
                    self.rec["lookups"][re.sub(r"\s+", " ", k).strip()] = item
        return status, body

    def get(self, url: str):
        status, body = self.live.get(url)
        if status == 200:
            self.rec["opinions"][url] = body
        return status, body


def load_heldout_recording() -> dict:
    if RECORDING.exists():
        with gzip.open(RECORDING, "rt", encoding="utf-8") as f:
            return json.load(f)
    return {"lookups": {}, "opinions": {}}


def record() -> None:
    token = secret("COURTLISTENER_TOKEN")
    if not token:
        sys.exit("Set COURTLISTENER_TOKEN (environment or .streamlit/secrets.toml).")
    items = json.loads(ITEMS.read_text(encoding="utf-8"))
    main, rec = load_recording(), load_heldout_recording()
    todo = [i for i in items if i["cite"] not in main["lookups"] and i["cite"] not in rec["lookups"]]
    print(f"{len(items) - len(todo)} of {len(items)} already recorded; fetching {len(todo)}.")
    if todo:
        src = CourtListenerSource(transport=Recorder(HttpTransport(token), rec))
        res = src.get_many([i["cite"] for i in todo], {i["cite"]: i["name"] for i in todo})
        failed = [o for o in res.values() if o.source == "lookup_error"]
        if failed:
            print("Stopped:", failed[0].note)
        with gzip.open(RECORDING, "wt", encoding="utf-8") as f:
            json.dump(rec, f)
    missing = [i["id"] for i in items if i["cite"] not in main["lookups"] and i["cite"] not in rec["lookups"]]
    print(f"Recorded: {len(rec['lookups'])} lookups, {len(rec['opinions'])} opinions."
          + (f" Still missing: {len(missing)} (re-run later)." if missing else " Complete."))


def run() -> None:
    items = json.loads(ITEMS.read_text(encoding="utf-8"))
    main, extra = load_recording(), load_heldout_recording()
    merged = {"lookups": {**main["lookups"], **extra["lookups"]},
              "opinions": {**main["opinions"], **extra["opinions"]}}
    missing = [i["id"] for i in items if i["cite"] not in merged["lookups"]]
    if missing:
        sys.exit(f"{len(missing)} citations aren't recorded yet; run `python -m eval.heldout record` first.")
    key = secret("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("Set ANTHROPIC_API_KEY (environment or .streamlit/secrets.toml).")
    store = json.loads(JUDGMENTS.read_text()) if JUDGMENTS.exists() else {}
    judge = ClaudeJudge(key, os.environ.get("CLAUDE_MODEL", DEFAULT_MODEL), store)
    source = CourtListenerSource(transport=ReplayTransport(merged))
    rows = []
    for it in items:
        r = (verify(it["text"], source, judge) or [None])[0]
        rows.append({"id": it["id"], "status": r.status if r else "NO_CITATION", "reason": r.reason if r else "",
                     "verdict": r.judgment.verdict if r and r.judgment else None})
    JUDGMENTS.write_text(json.dumps(judge.store, indent=0))
    n = len(rows)
    passed = sum(r["status"] in OK for r in rows)
    flagged = [r for r in rows if r["status"] in FLAGS]
    review = n - passed - len(flagged)
    out = {"n": n, "passed": passed, "false_alarms": len(flagged), "needs_review": review,
           "pass_ci": wilson(passed, n), "false_alarm_ci": wilson(len(flagged), n),
           "statuses": Counter(r["status"] for r in rows), "model": judge.model, "rows": rows}
    RESULTS.write_text(json.dumps(out, indent=1, default=list), encoding="utf-8")
    print(f"Held-out correct citations: {n} | passed {passed} ({100 * passed / n:.0f}%, 95% CI "
          f"{100 * out['pass_ci'][0]:.0f}-{100 * out['pass_ci'][1]:.0f}%) | wrongly flagged {len(flagged)} "
          f"| sent to review {review}")
    print("Statuses:", dict(out["statuses"]))
    for r in rows:
        if r["status"] not in OK:
            print(f"  {r['id']} {r['status']}: {r['reason'][:150]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["select", "record", "run"])
    {"select": select, "record": record, "run": run}[ap.parse_args().step]()
