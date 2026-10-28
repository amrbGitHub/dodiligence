"""DoDiligence: verify the citations in an AI-drafted legal memo."""
from __future__ import annotations

import datetime as dt
import html
import json
import os
from pathlib import Path

import streamlit as st

from core.judge import DEFAULT_MODEL, ClaudeJudge
from core.ocr import file_to_text, load_ocr, ocr_available
from core.pipeline import STATUS, verify
from core.sources import CompositeSource, CourtListenerSource, load_recording
import ui_accuracy

ROOT = Path(__file__).parent
EVAL = ROOT / "eval" / "data"
MAX_CHARS = 15000
MAX_CITES = 12


def secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, os.environ.get(name, default))
    except Exception:
        return os.environ.get(name, default)


ANTHROPIC_KEY = secret("ANTHROPIC_API_KEY")
CL_TOKEN = secret("COURTLISTENER_TOKEN")
MODEL = secret("CLAUDE_MODEL", DEFAULT_MODEL)
RUNS_PER_SESSION = int(secret("RUNS_PER_SESSION", "3"))
RUNS_PER_DAY = int(secret("RUNS_PER_DAY", "150"))

st.set_page_config(page_title="DoDiligence", page_icon="§", layout="wide")

# Highlighter colour (fill) and ink colour (text/rule) for each status.
PALETTE = {
    "VERIFIED":      ("#CDEFD6", "#23663D"),
    "QUOTE_OK":      ("#DCEBDF", "#3C6B4C"),
    "MISQUOTED":     ("#FFE680", "#7A5600"),
    "CONTRADICTED":  ("#FFC9D0", "#A3231B"),
    "UNSUPPORTED":   ("#FFD8B0", "#8F4508"),
    "NOT_FOUND":     ("#E3D7F7", "#55388F"),
    "NAME_MISMATCH": ("#E3D7F7", "#55388F"),
    "REVIEW":        ("#DEE3EA", "#475266"),
}

st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"], .stMarkdown, .stButton button, .stTabs, textarea, input {
  font-family: "Public Sans", system-ui, sans-serif;
}
.block-container { padding-top: 2.2rem; max-width: 1240px; }
h1.cc-title { font-family: "Source Serif 4", Georgia, serif; font-weight: 600; font-size: 2.6rem;
  letter-spacing: -0.01em; margin: 0 0 .35rem 0; color: #1C2433; }
h1.cc-title span.sect { color: #8A94A6; font-weight: 400; margin-right: .35rem; }
p.cc-lede { font-size: 1.06rem; line-height: 1.55; color: #3E4A5E; max-width: 68ch; margin: 0 0 1.4rem 0; }
.cc-sheet { background: #FFFFFF; border: 1px solid #D5DBE3; padding: 2.4rem 2.6rem 2.6rem;
  font-family: "Source Serif 4", Georgia, serif; font-size: 1.02rem; line-height: 1.75; color: #1C2433;
  white-space: pre-wrap; box-shadow: 0 1px 0 #D5DBE3, 0 6px 0 -3px #F7F8FA, 0 7px 0 -3px #D5DBE3; }
.cc-mark { padding: .08em .12em; border-radius: 2px; box-decoration-break: clone; -webkit-box-decoration-break: clone; }
.cc-cite { font-weight: 600; text-decoration: underline; text-decoration-thickness: 2px; text-underline-offset: 3px; }
.cc-num { font-family: "Public Sans", sans-serif; font-size: .7rem; font-weight: 700; vertical-align: super;
  margin-left: .15em; padding: 0 .3em; border-radius: 3px; color: #FFFFFF; }
.cc-summary { font-size: 1.08rem; margin: .4rem 0 1.1rem 0; color: #1C2433; }
.cc-summary b { font-weight: 700; }
.cc-note { border-left: 4px solid; background: #FFFFFF; padding: .85rem 1rem .8rem 1rem; margin-bottom: .2rem; }
.cc-note .head { display: flex; gap: .6rem; align-items: baseline; flex-wrap: wrap; }
.cc-note .n { font-weight: 700; font-size: .85rem; color: #FFFFFF; padding: 0 .4em; border-radius: 3px; }
.cc-note .status { font-weight: 700; font-size: .98rem; }
.cc-note .case { color: #3E4A5E; font-size: .92rem; }
.cc-note .case i { font-family: "Source Serif 4", serif; }
.cc-note p.reason { margin: .45rem 0 0 0; font-size: .95rem; line-height: 1.5; color: #1C2433; }
.cc-quote { font-family: "Source Serif 4", serif; font-size: .95rem; line-height: 1.6; padding: .6rem .8rem;
  background: #F7F8FA; border: 1px solid #E3E7ED; margin: .3rem 0 .6rem 0; }
.cc-legend { font-size: .85rem; color: #3E4A5E; display: flex; flex-wrap: wrap; gap: .4rem .9rem; margin: .2rem 0 1rem 0; }
.cc-legend span.sw { display: inline-block; width: .9rem; height: .9rem; border-radius: 2px; vertical-align: -2px; margin-right: .3rem; }
.cc-foot { font-size: .85rem; color: #5B6576; margin-top: 2.5rem; max-width: 80ch; line-height: 1.55; }
.cc-court { background: #F7F8FA; border: 1px dashed #B8C0CC; padding: .6rem .8rem; margin: .35rem 0 1rem 0;
  font-size: .9rem; line-height: 1.5; color: #3E4A5E; }
.cc-court b { color: #1C2433; }
.cc-stat { background: #FFFFFF; border: 1px solid #D5DBE3; padding: 1rem 1.1rem; height: 100%; }
.cc-stat .big { font-family: "Source Serif 4", Georgia, serif; font-size: 2.6rem; font-weight: 600; color: #1C2433; line-height: 1.1; }
.cc-stat .lab { font-weight: 600; color: #1C2433; margin-top: .15rem; }
.cc-stat .sub { font-size: .85rem; color: #5B6576; margin-top: .3rem; }
.cc-row { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.3fr); gap: 1.2rem; padding: .8rem 0;
  border-bottom: 1px solid #E3E7ED; align-items: center; }
.cc-row .n { color: #5B6576; font-weight: 400; }
.cc-row .desc { font-size: .88rem; color: #5B6576; margin-top: .15rem; }
.cc-row .nums { font-size: .88rem; color: #3E4A5E; margin-top: .35rem; }
.cc-bar { display: flex; height: 14px; background: #E9ECF1; border-radius: 2px; overflow: hidden; }
.cc-bar span { display: block; height: 100%; }
.cc-strip { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .8rem; margin: .4rem 0 1.2rem 0; }
.cc-strip .cc-stat .big { font-size: 2.1rem; }
/* Motion: every animation starts from a hidden state and ends on the element's normal style,
   so with reduced motion (animation: none) everything simply shows. --i staggers items. */
@property --cc-n { syntax: "<integer>"; initial-value: 0; inherits: false; }
@keyframes cc-rise { from { opacity: 0; transform: translateY(12px); } }
@keyframes cc-slide { from { opacity: 0; transform: translateX(16px); } }
@keyframes cc-swipe { from { background-size: 0% 100%; } }
@keyframes cc-pop { 0% { transform: scale(0); } 70% { transform: scale(1.3); } }
@keyframes cc-count { from { --cc-n: 0; } }
@keyframes cc-glint { from { color: #4C7DD9; } }
h1.cc-title { animation: cc-rise .6s ease-out both; }
h1.cc-title span.sect { display: inline-block; animation: cc-pop .5s .25s ease-out both, cc-glint 1.2s .25s ease-out both; }
p.cc-lede { animation: cc-rise .6s .15s ease-out both; }
.cc-stat { animation: cc-rise .5s calc(.2s + var(--i, 0) * .1s) ease-out both; }
.cc-stat .big.count { --cc-n: var(--to); counter-reset: n var(--cc-n); animation: cc-count 1.1s calc(.3s + var(--i, 0) * .1s) ease-out both; }
.cc-stat .big.count::after { content: counter(n) var(--suffix, ""); }
.cc-sheet { animation: cc-rise .5s ease-out both; }
.cc-mark { background-image: linear-gradient(var(--fill), var(--fill)); background-repeat: no-repeat; background-size: 100% 100%;
  animation: cc-swipe .6s calc(.45s + var(--i, 0) * .25s) ease-out both; }
.cc-num { display: inline-block; animation: cc-pop .35s calc(.9s + var(--i, 0) * .25s) ease-out both; }
.cc-note, .cc-court { animation: cc-slide .45s calc(.45s + var(--i, 0) * .25s) ease-out both; }
[data-testid="stColumn"] [data-testid="stExpander"] { animation: cc-rise .5s 1.2s ease-out both; }
@keyframes cc-grow { from { width: 0; } }
.cc-bar span { animation: cc-grow 1s .3s ease-out both; }
@media (max-width: 640px) {
  .cc-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .cc-row { grid-template-columns: 1fr; gap: .4rem; }
  .cc-sheet { padding: 1.2rem 1.1rem 1.4rem; font-size: .98rem; }
  h1.cc-title { font-size: 2.1rem; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
.stButton button[kind="primary"] { background: #1C2433; border-color: #1C2433; font-weight: 600; padding: .55rem 1.3rem; }
.stButton button:focus-visible { outline: 3px solid #4C7DD9; outline-offset: 2px; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------- shared state ----------------

@st.cache_resource
def judgment_store() -> dict:
    """AI verdicts shared by every visitor. Seeded with the benchmark's verdicts, so the real-cases demo is free."""
    p = EVAL / "judgments.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_resource
def recording() -> dict:
    return load_recording()


@st.cache_data
def showcase() -> tuple[str, list[dict]]:
    items = {i["id"]: i for i in json.loads((EVAL / "items.json").read_text(encoding="utf-8"))}
    picks = [items[i] for i in json.loads((ROOT / "data" / "showcase.json").read_text()) if i in items]
    return "\n\n".join(p["text"] for p in picks), picks


@st.cache_resource
def sample() -> tuple[str, list, float, dict]:
    """The sample memo, checked once per server from recorded data (no keys, no network).
    Cached so reruns don't replay the page's animations with a new timing."""
    import time
    text = (ROOT / "data" / "sample_memo.txt").read_text(encoding="utf-8")
    judge = ClaudeJudge(ANTHROPIC_KEY or "cache-only", MODEL, store=judgment_store())
    source = CompositeSource(None, recording())
    verify(text, source, judge)            # first call loads eyecite's tables; time the steady state
    t = time.perf_counter()
    results = verify(text, source, judge)
    ms = (time.perf_counter() - t) * 1000
    items = {i["cite"]: i for i in json.loads((EVAL / "items.json").read_text(encoding="utf-8"))
             if i["id"].startswith("mata")}
    return text, results, ms, items


@st.cache_resource
def daily_counter() -> dict:
    return {"day": dt.date.today().isoformat(), "runs": 0}


@st.cache_resource(show_spinner="Loading the OCR model (first use only)…")
def ocr_engine():
    return load_ocr()


def budget_left() -> tuple[bool, str]:
    c = daily_counter()
    today = dt.date.today().isoformat()
    if c["day"] != today:
        c["day"], c["runs"] = today, 0
    if st.session_state.get("runs", 0) >= RUNS_PER_SESSION:
        return False, (f"You've used this demo's {RUNS_PER_SESSION} AI checks for this session. "
                       "The real-cases demo still works, because its results are saved.")
    if c["runs"] >= RUNS_PER_DAY:
        return False, "The demo has reached today's limit for new AI checks. The real-cases demo still works."
    return True, ""


# ---------------- rendering ----------------

def render_memo(text: str, results) -> str:
    def esc(x: str) -> str:   # newlines as <br>, so Markdown doesn't split the sheet into paragraphs
        return html.escape(x).replace("\n", "<br>")

    out, pos = [], 0
    for n, r in enumerate(sorted(results, key=lambda r: r.claim.span[0])):
        a, b = r.claim.span
        ca, cb = r.claim.cite_span
        fill, ink = PALETTE[r.status]
        out.append(esc(text[pos:a]))
        out.append(
            f'<span class="cc-mark" style="--fill:{fill};--i:{n}">{esc(text[a:ca])}'
            f'<span class="cc-cite" style="text-decoration-color:{ink}">{esc(text[ca:cb])}</span></span>'
            f'<span class="cc-num" style="background:{ink};--i:{n}">{r.claim.idx}</span>'
        )
        pos = b
    out.append(esc(text[pos:]))
    return f'<div class="cc-sheet">{"".join(out)}</div>'


def render_court(item: dict | None, i: int = 0) -> None:
    if not item:
        return
    if item["group"] == "control":
        head = "Ground truth: a correct citation, written by a judge"
    else:
        head = "Ground truth: what the court found"
    when = (item.get("date") or "")[:10]
    src = f'{html.escape(item.get("decision", ""))}' + (f' ({html.escape(item.get("court", ""))}{", " + when if when else ""})' if item.get("court") else "")
    link = item.get("document") or ""
    st.markdown(
        f'<div class="cc-court" style="--i:{i}"><b>{head}.</b> {html.escape(item.get("court_finding", ""))}<br>'
        f'<span style="font-size:.82rem">Source: {src}'
        + (f' · <a href="{html.escape(link)}" target="_blank">document</a>' if link.startswith("http") else "")
        + '</span></div>', unsafe_allow_html=True)


def render_note(r, item: dict | None = None) -> None:
    fill, ink = PALETTE[r.status]
    label = STATUS[r.status][0]
    db_name = r.opinion.case_name if r.opinion.found else ""
    case = f"<i>{html.escape(r.claim.case_name)}</i>, {html.escape(r.claim.citation)}"
    st.markdown(
        f'<div class="cc-note" style="border-color:{ink};--i:{r.claim.idx - 1}">'
        f'<div class="head"><span class="n" style="background:{ink}">{r.claim.idx}</span>'
        f'<span class="status" style="color:{ink}">{label}</span><span class="case">{case}</span></div>'
        f'<p class="reason">{html.escape(r.reason)}</p></div>',
        unsafe_allow_html=True,
    )
    render_court(item, r.claim.idx - 1)
    if not r.opinion.found and r.status == "NOT_FOUND":
        return
    with st.expander("Compare with the opinion"):
        if db_name:
            st.markdown(f"**Opinion:** *{html.escape(db_name)}* ({r.opinion.year})")
        for q in r.quotes:
            st.markdown("Memo quotes:")
            st.markdown(f'<div class="cc-quote">“{html.escape(q.quote)}”</div>', unsafe_allow_html=True)
            st.markdown(f"Opinion says ({q.score:.0f}% match):")
            st.markdown(f'<div class="cc-quote">…{html.escape(q.context)}…</div>', unsafe_allow_html=True)
        if r.judgment and r.excerpts:
            k = r.judgment.key_excerpt
            if 1 <= k <= len(r.excerpts):
                st.markdown("Key passage the check relied on:")
                st.markdown(f'<div class="cc-quote">{html.escape(r.excerpts[k - 1])}</div>', unsafe_allow_html=True)
        if r.opinion.url:
            st.markdown(f"[Read the full opinion on CourtListener]({r.opinion.url})")


def legend() -> None:
    items = [("VERIFIED", "Verified"), ("MISQUOTED", "Misquoted"), ("CONTRADICTED", "Contradicted"),
             ("UNSUPPORTED", "Not supported"), ("NOT_FOUND", "Case not found or wrong case"),
             ("REVIEW", "Needs review")]
    spans = "".join(
        f'<span><span class="sw" style="background:{PALETTE[k][0]};border:1px solid {PALETTE[k][1]}"></span>{t}</span>'
        for k, t in items)
    st.markdown(f'<div class="cc-legend">{spans}</div>', unsafe_allow_html=True)


def stat(i: int, big: str, label: str, sub: str = "", count: int | None = None, suffix: str = "") -> str:
    """A stat card. With `count`, the number counts up from 0 (CSS only; browsers without @property show it at once)."""
    num = (f'<div class="big count" style="--to:{count};--suffix:\'{suffix}\';--i:{i}"></div>' if count is not None
           else f'<div class="big">{big}</div>')
    return (f'<div class="cc-stat" style="--i:{i}">{num}<div class="lab">{label}</div>'
            + (f'<div class="sub">{sub}</div>' if sub else "") + '</div>')


def show_results(text: str, results, picks=lambda r: None) -> None:
    good = sum(r.status in ("VERIFIED", "QUOTE_OK") for r in results)
    bad = len(results) - good
    word = "citation" if len(results) == 1 else "citations"
    st.markdown(
        f'<p class="cc-summary"><b>{len(results)} {word} checked.</b> {good} held up; '
        f'{bad} {"needs" if bad == 1 else "need"} attention.</p>', unsafe_allow_html=True)
    legend()
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown(render_memo(text, results), unsafe_allow_html=True)
    with right:
        for r in results:
            render_note(r, picks(r))


# ---------------- page ----------------

st.markdown('<h1 class="cc-title"><span class="sect">§</span>DoDiligence</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="cc-lede">AI legal research tools can cite cases that don’t exist, misquote real ones, or claim a case '
    'held something it didn’t. DoDiligence reads a memo, finds every case citation, pulls the actual opinion, and '
    'checks each one: does the case exist, is the quote accurate, and does the opinion really support the claim?</p>',
    unsafe_allow_html=True,
)

tab_sample, tab_real, tab_acc, tab_paste, tab_upload = st.tabs(
    ["Sample memo", "Real cases", "Accuracy", "Paste a memo", "Upload a file"])
memo_text, mode = None, None

with tab_sample:
    s_text, s_results, s_ms, s_items = sample()
    s_flagged = sum(r.status not in ("VERIFIED", "QUOTE_OK", "REVIEW") for r in s_results)
    st.caption("An AI-drafted research memo on the issue in *Mata v. Avianca* (S.D.N.Y. 2023). Its six case "
               "citations are the fake ones the lawyers actually filed in that case, the filing that got them "
               "sanctioned. The surrounding text is a sample written for this demo.")
    st.markdown(
        '<div class="cc-strip">'
        + stat(0, "", "citations found", count=len(s_results))
        + stat(1, "", "flagged as fake or wrong", count=s_flagged)
        + stat(2, "", "to check the whole memo", count=max(1, round(s_ms)), suffix=" ms")
        + stat(3, "0", "GPUs needed", "runs on an ordinary CPU")
        + '</div>', unsafe_allow_html=True)
    show_results(s_text, s_results, lambda r: s_items.get(r.claim.citation))

with tab_real:
    sc_text, sc_items = showcase()
    st.caption("Every citation below was filed in a real US court case. Judges later found most of them fake, "
               "misquoted or misrepresented; the last ones were written correctly by judges. DoDiligence doesn't "
               "see the courts' findings. Run it, then compare its notes with what the court said.")
    with st.expander("Read the citations first"):
        st.markdown(f'<div class="cc-sheet">{html.escape(sc_text)}</div>', unsafe_allow_html=True)
    if st.button("Check these citations", type="primary", key="run_real"):
        memo_text, mode = sc_text, "showcase"

with tab_acc:
    ui_accuracy.render()


with tab_paste:
    pasted = st.text_area("Memo text", height=260, max_chars=MAX_CHARS,
                          placeholder="Paste a legal memo or brief that cites U.S. cases…")
    if not CL_TOKEN:
        st.caption("Live case lookups aren't configured on this deployment, so only cases from the benchmark can be checked.")
    else:
        st.caption("Case lookups use CourtListener's free API, which allows only a few requests a minute, "
                   "so a memo with many citations can take a minute or two.")
    if st.button("Check citations", type="primary", key="run_paste", disabled=not pasted.strip()):
        memo_text, mode = pasted, "paste"

with tab_upload:
    up = st.file_uploader("PDF or image of a memo or brief", type=["pdf", "png", "jpg", "jpeg"])
    st.caption("Digital PDFs are read directly. Scanned pages and images are read with PP-OCRv6, "
               "running on this server, so documents aren't sent to an outside OCR service. "
               "First 5 pages only." + ("" if ocr_available() else " (OCR isn't installed on this deployment.)"))
    if up is not None and st.button("Read and check citations", type="primary", key="run_upload"):
        try:
            with st.spinner("Reading the document…"):
                text, how = file_to_text(up.name, up.getvalue(), ocr_engine)
            st.session_state["upload_note"] = how
            memo_text, mode = text[:MAX_CHARS], "upload"
        except Exception as e:
            st.error(str(e))

OUT = {"showcase": tab_real.container(), "paste": tab_paste.container(), "upload": tab_upload.container()}

# ---------------- run ----------------

if memo_text:
    st.session_state["last_mode"] = mode
run_mode = mode or st.session_state.get("last_mode") or st.session_state.get("last", (None, None, None))[2]
with (OUT.get(run_mode) if run_mode else st.container()):
    if memo_text:
        judge = None
        if mode == "showcase":
            # Every showcase verdict is in the store, so this needs no key; a cache miss fails into "Needs review".
            judge = ClaudeJudge(ANTHROPIC_KEY or "cache-only", MODEL, store=judgment_store())
        elif ANTHROPIC_KEY:
            allowed, why = budget_left()
            if allowed:
                judge = ClaudeJudge(ANTHROPIC_KEY, MODEL, store=judgment_store())
            else:
                st.info(why)
        source = CompositeSource(CourtListenerSource(CL_TOKEN) if CL_TOKEN else None, recording())
        bar = st.progress(0.0, text="Finding citations…")

        def progress(n, total, claim):
            bar.progress(n / total, text=f"Checking {n} of {total}: {claim.case_name}")

        try:
            # Cap before lookup: each citation uses CourtListener requests (50 an hour on the free tier).
            results = verify(memo_text, source, judge, on_progress=progress,
                             max_cites=None if mode == "showcase" else MAX_CITES)
        except Exception as e:
            bar.empty()
            st.error(f"The check stopped: {e}")
            results = None
        bar.empty()
        if results is not None:
            if results and judge is not None and judge.api_calls > 0:
                st.session_state["runs"] = st.session_state.get("runs", 0) + 1
                daily_counter()["runs"] += 1
            st.session_state["last"] = (memo_text, results, mode)

    if "last" in st.session_state:
        memo_text, results, mode = st.session_state["last"]
        if not results:
            st.warning("No case citations were found. DoDiligence looks for full citations like "
                       "“Kontrick v. Ryan, 540 U.S. 443 (2004)”.")
        else:
            if len(results) == MAX_CITES and mode != "showcase":
                st.caption(f"The demo checks at most the first {MAX_CITES} citations in a document.")
            if mode == "upload" and st.session_state.get("upload_note"):
                st.caption(st.session_state["upload_note"])
            if not ANTHROPIC_KEY and mode != "showcase":
                st.caption("AI support-checking is off on this deployment, so only existence and quotes were checked.")
            picks = showcase()[1] if mode == "showcase" else []
            show_results(memo_text, results,
                         lambda r: picks[r.claim.idx - 1] if r.claim.idx <= len(picks) else None)

st.markdown(
    '<div class="cc-foot"><b>How it works.</b> Citations are found with eyecite (Free Law Project). '
    'Opinions come from CourtListener. Quotes are matched against the opinion text with fuzzy matching, '
    'so small transcription differences don’t trigger false alarms. '
    'Finding citations, matching quotes and picking passages take a few milliseconds per citation on an '
    'ordinary CPU, with no GPU. '
    + (f'Whether the opinion supports the claim is judged by {html.escape(MODEL)}, which reads only about five '
       'passages from the opinion, not the whole thing, and answers in about a second. ' if ANTHROPIC_KEY else '')
    + 'Real-case data: Damien Charlotin, <a href="https://www.damiencharlotin.com/hallucinations/" target="_blank">'
      'AI Hallucination Cases database</a> (CC BY 4.0). '
    + 'This is a demo, not legal advice; always read the opinion.</div>',
    unsafe_allow_html=True,
)
