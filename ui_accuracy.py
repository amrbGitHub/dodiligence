"""The Accuracy tab: measured results from eval/run_eval.py."""
from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

DATA = Path(__file__).parent / "eval" / "data"
GROUPS = [
    ("fabricated", "Fake or wrong-case citations",
     "The citation leads to no case, or to a different case than the one named."),
    ("false_quote", "Invented quotations", "A real case, quoted saying words that aren't in it."),
    ("misrepresented", "Real case, misstated holding", "A real case cited for something it doesn't say."),
]
STATUS_LABEL = {
    "VERIFIED": "Verified", "QUOTE_OK": "Quote verified", "NOT_FOUND": "Case not found",
    "NAME_MISMATCH": "Wrong case", "MISQUOTED": "Misquoted", "CONTRADICTED": "Contradicted",
    "UNSUPPORTED": "Not supported", "REVIEW": "Needs review",
}
FLAGS = {"NOT_FOUND", "NAME_MISMATCH", "MISQUOTED", "CONTRADICTED", "UNSUPPORTED"}
# Which check produced each flag. Only the support check uses a language model.
CAUGHT_BY = {"NOT_FOUND": "case lookup", "NAME_MISMATCH": "case lookup", "MISQUOTED": "quote match",
             "CONTRADICTED": "support check", "UNSUPPORTED": "support check"}


@st.cache_data
def load():
    res = json.loads((DATA / "results.json").read_text(encoding="utf-8"))
    items = {i["id"]: i for i in json.loads((DATA / "items.json").read_text(encoding="utf-8"))}
    return res, items


@st.cache_data
def heldout() -> tuple[dict, dict] | None:
    """Correct citations chosen after the rules were frozen (eval/heldout.py), if that run exists."""
    p, q = DATA / "heldout_results.json", DATA / "heldout_items.json"
    if not (p.exists() and q.exists()):
        return None
    return (json.loads(p.read_text(encoding="utf-8")),
            {i["id"]: i for i in json.loads(q.read_text(encoding="utf-8"))})


@st.cache_data
def perf() -> dict | None:
    p = Path(__file__).parent / "data" / "perf.json"
    return json.loads(p.read_text()) if p.exists() else None


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def bar(frac: float, color: str, frac2: float = 0.0, color2: str = "#C9CFD8") -> str:
    a, b = 100 * frac, 100 * frac2
    return (f'<div class="cc-bar"><span style="width:{a:.1f}%;background:{color}"></span>'
            f'<span style="width:{b:.1f}%;background:{color2}"></span></div>')


def render() -> None:
    if not (DATA / "results.json").exists():
        st.info("Benchmark results haven't been generated yet. Run `python -m eval.run_eval`.")
        return
    res, items = load()
    o, g = res["overall"], res["groups"]
    ctrl = g.get("control") or {"n": 0, "passed": 0, "pass_rate": 0.0, "false_alarms": 0, "needs_review": 0}
    if not ctrl["n"]:
        ctrl = dict(ctrl, n=1)   # avoid dividing by zero before controls exist
    n_bad = o["tp"] + o["fn"]
    ho = heldout()
    hr, h_items = ho if ho else (None, {})
    n_good = ctrl["n"] + (hr["n"] if hr else 0)
    st.markdown(
        f'<p class="cc-lede" style="margin-bottom:.6rem">We ran DoDiligence on <b>{n_bad} citations that US courts '
        f'caught in real filings</b> (fake cases, invented quotes, real cases cited for things they don’t say), and on '
        f'<b>{n_good} correct citations written by judges</b>'
        + (f' ({hr["n"]} of them chosen only after DoDiligence’s rules were frozen)' if hr else '')
        + '. Ground truth is the courts’ own findings, not ours.</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    pf = perf()
    cards = [
        (c1, "", "of real errors flagged",
         f'{o["tp"]} of {n_bad} · 95% CI {pct(o["recall_ci"][0])}–{pct(o["recall_ci"][1])}',
         f'--to:{round(100 * o["recall"])};--suffix:\'%\''),
        (c2, f'{hr["passed"]} of {hr["n"]}', "held-out correct citations passed",
         f'{hr["false_alarms"]} wrongly flagged, {hr["needs_review"]} sent to review · never seen during development',
         "") if hr else
        (c2, f'{ctrl["passed"]} of {ctrl["n"]}', "correct citations passed",
         f'{ctrl["false_alarms"]} wrongly flagged · small sample, see note below', ""),
    ]
    if pf:
        cards.append((c3, f'{pf["median_ms"]:g} ms', "per citation for lookup and quote checks",
                      f'median on one CPU core, no GPU · support check about {pf["ai_median_s"]:g} s', ""))
    for i, (col, big, label, sub, count) in enumerate(cards):
        num = (f'<div class="big count" style="{count};--i:{i}"></div>' if count else f'<div class="big">{big}</div>')
        col.markdown(f'<div class="cc-stat" style="--i:{i}">{num}<div class="lab">{label}</div>'
                     f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown("#### Caught, by kind of error")
    caught = {key: Counter(CAUGHT_BY[r["status"]] for r in res["rows"] if r["group"] == key and r["status"] in FLAGS)
              for key, _, _ in GROUPS}
    n_flags = sum(sum(c.values()) for c in caught.values())
    n_model = sum(c["support check"] for c in caught.values())
    n_asked = sum(1 for r in res["rows"] if r.get("verdict"))
    st.caption(f"The case lookup and quote match are plain checks with no AI, and they produce {n_flags - n_model} of "
               f"the {n_flags} catches. Only the support check uses a language model, and only on citations that pass "
               f"the other two ({n_asked} of the {len(res['rows'])} test citations).")
    rows = []
    for key, label, desc in GROUPS:
        d = g.get(key)
        if not d:
            continue
        by = " · ".join(f"{name} {n}" for name, n in caught[key].most_common())
        rows.append(
            f'<div class="cc-row"><div class="name"><b>{label}</b> <span class="n">({d["n"]})</span>'
            f'<div class="desc">{desc}</div></div>'
            f'<div class="viz">{bar(d["catch_rate"], "#1C2433", d["needs_review"] / d["n"])}'
            f'<div class="nums"><b>{pct(d["catch_rate"])}</b> flagged'
            + (f' · {d["needs_review"]} sent to review' if d["needs_review"] else "")
            + f'<br>caught by {by}</div></div></div>')
    good_rows = [("Correct citations, development set", ctrl["n"], ctrl["passed"], ctrl["false_alarms"],
                  ctrl["needs_review"], "Sentences where judges cite a case, seen while the rules were tuned.")]
    if hr:
        good_rows.append(("Correct citations, held out", hr["n"], hr["passed"], hr["false_alarms"], hr["needs_review"],
                          "Chosen the same way, but only after the rules were frozen. These should pass."))
    for label, n, passed, alarms, review, desc in good_rows:
        rows.append(
            f'<div class="cc-row"><div class="name"><b>{label}</b> <span class="n">({n})</span>'
            f'<div class="desc">{desc}</div></div>'
            f'<div class="viz">{bar(passed / n, "#23663D", review / n)}'
            f'<div class="nums"><b>{pct(passed / n)}</b> passed · {alarms} false alarms · '
            f'{review} sent to review</div></div></div>')
    st.markdown("".join(rows), unsafe_allow_html=True)
    st.caption("Dark bar: flagged as a problem. Grey: “needs review” (DoDiligence couldn’t decide and asks a person to look).")
    if hr:
        lo, hi = hr["pass_ci"]
        st.markdown(f"**Held-out check.** {hr['n']} correct citations were chosen from the same opinions, with the "
                    f"same rules and a fixed random seed, only after DoDiligence’s checks were frozen; the selection "
                    f"was committed to the repository before anything was run on it. {hr['passed']} passed "
                    f"({pct(hr['passed'] / hr['n'])}, 95% CI {pct(lo)}–{pct(hi)}), {hr['false_alarms']} were wrongly "
                    f"flagged and {hr['needs_review']} were sent to review.")

    if pf:
        st.markdown(f"**Speed.** All {pf['items']} citations went through the local checks in {pf['total_s']:g} s "
                    f"on one CPU core (median {pf['median_ms']:g} ms each"
                    + (f", about {pf['memory_mb']} MB of memory" if pf.get("memory_mb") else "")
                    + f", no GPU). The support check sends the model about {pf['ai_input_tokens']:,} tokens, roughly "
                    f"five passages rather than the whole opinion, and its median answer took {pf['ai_median_s']:g} s. "
                    "Two fresh runs from scratch gave identical results. Measured with `python -m eval.perf`.")

    st.markdown("#### Every test case")
    st.caption("Including the ones DoDiligence got wrong. Pick a row group to filter.")
    df = pd.DataFrame([{
        "Group": dict((k, l) for k, l, _ in GROUPS).get(r["group"], "Correct citation"),
        "DoDiligence said": STATUS_LABEL.get(r["status"], r["status"]),
        "Right?": ("✓" if (r["status"] in FLAGS) == (r["group"] != "control") and r["status"] != "REVIEW"
                   else ("review" if r["status"] == "REVIEW" else "✗")),
        "Citation as filed": items[r["id"]]["text"],
        "What the court found": items[r["id"]].get("court_finding", ""),
        "Source": items[r["id"]].get("decision", ""),
    } for r in res["rows"]] + ([{
        "Group": "Correct citation (held out)",
        "DoDiligence said": STATUS_LABEL.get(r["status"], r["status"]),
        "Right?": "✓" if r["status"] in ("VERIFIED", "QUOTE_OK") else ("review" if r["status"] == "REVIEW" else "✗"),
        "Citation as filed": h_items[r["id"]]["text"],
        "What the court found": h_items[r["id"]].get("court_finding", ""),
        "Source": h_items[r["id"]].get("decision", ""),
    } for r in hr["rows"]] if hr else []))
    pick = st.segmented_control("Show", ["All", "Misses only", "Correct citations"], default="All",
                                key="acc_filter")
    view = df
    if pick == "Misses only":
        view = df[df["Right?"] != "✓"]
    elif pick == "Correct citations":
        view = df[df["Group"].str.startswith("Correct citation")]
    st.dataframe(view, hide_index=True, width="stretch", height=380,
                 column_config={"Citation as filed": st.column_config.TextColumn(width="large"),
                                "What the court found": st.column_config.TextColumn(width="large")})
    st.download_button("Download all results (CSV)", df.to_csv(index=False).encode(), "dodiligence_benchmark.csv",
                       "text/csv")

    with st.expander("How this was measured"):
        st.markdown(f"""
**Errors (positives).** From Damien Charlotin’s [AI Hallucination Cases database]
(https://www.damiencharlotin.com/hallucinations/) (CC BY 4.0), which records court decisions that found
AI-generated errors in filings. We took US entries with a single reporter citation (not Westlaw/Lexis numbers),
and for each one reconstructed the citation as filed: the case name and citation, plus the proposition or quotation
the court said the filer attributed to it. The court’s finding is the label.

**Correct citations (negatives).** Sentences in which judges cite another case, taken from the court opinions
downloaded for the error set (one citation per sentence, no “but see”/“cf.” signals). We assume judges cited
correctly. **The first 14 were looked at while DoDiligence’s rules were adjusted, so their pass rate is
optimistic.** A held-out set was then chosen from the same opinions with the same rules and a fixed seed, after the
rules were frozen, and committed before it was run (`eval/heldout.py`); its result is the one to trust.

**Precision.** Because the test set has far more errors than correct citations, “share of flags that were real
errors” would look near-perfect no matter what, so we report the false-alarm count on correct citations instead.

**What counts.** A problem is “caught” if DoDiligence returns Case not found, Wrong case, Misquoted, Contradicted
or Not supported. “Needs review” is counted as not caught in the headline numbers.

**Reproducible.** CourtListener responses were recorded once, so every run sees the same opinions.
Run it yourself: `python -m eval.run_eval` (needs an Anthropic key; `--no-ai` runs without one).

**Limits.** The propositions for misstated holdings were reconstructed from the courts’ descriptions with an LLM and
checked by hand, so they are close to, but not always word-for-word, what the filer wrote. Only the first
(majority) opinion of each case is checked.
""")
