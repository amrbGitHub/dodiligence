"""Quote matching and passage retrieval over an opinion's text."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from rapidfuzz import fuzz

_STAR_PAGE = re.compile(r"\*\s?\d{1,4}\s?")          # Harvard star pagination: *447
_TRANSLATE = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "–": "-"})


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Lowercase, straighten quotes, drop brackets/star-pages, collapse spaces.

    Returns the normalized string and, for every normalized char, the index of
    the original char it came from (so matches can be shown in the original).
    """
    text = text.translate(_TRANSLATE)
    out: list[str] = []
    idx: list[int] = []
    stars = [(m.start(), m.end()) for m in _STAR_PAGE.finditer(text)]
    s_i = 0
    prev_space = True
    for i, ch in enumerate(text):
        while s_i < len(stars) and stars[s_i][1] <= i:
            s_i += 1
        if s_i < len(stars) and stars[s_i][0] <= i < stars[s_i][1]:
            continue
        if ch in "[]":
            continue
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            idx.append(i)
            prev_space = True
            continue
        out.append(ch.lower())
        idx.append(i)
        prev_space = False
    return "".join(out), idx


def normalize(text: str) -> str:
    return _normalize_with_map(text)[0]


@dataclass
class QuoteResult:
    quote: str
    score: float             # 0-100
    status: str              # "exact" | "close" | "not_found"
    matched_text: str        # what the opinion actually says at the best match
    context: str = ""        # the same, with a few words either side


def check_quote(quote: str, opinion_text: str) -> QuoteResult:
    norm_op, op_map = _normalize_with_map(opinion_text)
    # Split on ellipses so "a . . . b" is checked as two fragments.
    # Nested quotation marks (" `...' ") belong to the memo, not the opinion.
    frags = [f.strip(" `'\"") for f in re.split(r"\.\s?\.\s?\.|…", quote) if len(f.strip(" `'\"")) >= 8]
    if not frags:
        frags = [quote]
    scores, spans = [], []
    for frag in frags:
        nq = normalize(frag)
        al = fuzz.partial_ratio_alignment(nq, norm_op, score_cutoff=0)
        scores.append(al.score)
        spans.append((al.dest_start, al.dest_end))
    score = min(scores)
    a = min(s for s, _ in spans)
    b = max(e for _, e in spans)
    a_orig = op_map[a] if a < len(op_map) else 0
    b_orig = op_map[b - 1] + 1 if 0 < b <= len(op_map) else a_orig
    def clip(a: int, b: int, pad: int) -> str:
        a = 0 if a - pad <= 0 else opinion_text.rfind(" ", 0, a - pad) + 1
        e = opinion_text.find(" ", min(len(opinion_text), b + pad))
        b = len(opinion_text) if e == -1 else e
        return re.sub(r"\s+", " ", opinion_text[a:b]).strip()

    matched = clip(a_orig, b_orig, 0).lstrip(" “\"")
    # The alignment window is as long as the memo's quote, so it can run past the
    # end of the real quotation. Stop at a closing quote or the last full stop.
    close = [i for i in (matched.find("”", 20), matched.find('"', 20)) if i != -1]
    if close:
        matched = matched[: min(close)]
    else:
        stop = matched.rfind(". ")
        if stop > len(matched) * 0.6:
            matched = matched[: stop + 1]
    matched = matched.strip()
    status = "exact" if score >= 96 else "close" if score >= 75 else "not_found"
    return QuoteResult(quote=quote, score=round(score, 1), status=status,
                       matched_text=matched, context=clip(a_orig, b_orig, 60))


# ---------------- passage retrieval (small BM25, no extra deps) ----------------

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set(
    "the a an of to and or in on for that is are was were be by with as at it its this "
    "which from not but has have had any may such their they we our court courts".split()
)


def _tokens(s: str) -> list[str]:
    return [w for w in _WORD.findall(normalize(s)) if w not in _STOP and len(w) > 2]


def split_passages(text: str, max_chars: int = 1400, min_chars: int = 600) -> list[str]:
    # Some opinion texts put a blank line after every printed line; join short paragraphs until a
    # passage is long enough to carry a whole thought.
    paras: list[str] = []
    for p in (" ".join(p.split()) for p in re.split(r"\n\s*\n", text)):
        if not p:
            continue
        if paras and len(paras[-1]) < min_chars and not p.startswith("[0"):   # "[010lead]" etc. start a new opinion
            paras[-1] += " " + p
        else:
            paras.append(p)
    paras = [p for p in paras if len(p) > 40]
    out: list[str] = []
    for p in paras:
        while len(p) > max_chars:
            cut = p.rfind(". ", 0, max_chars)
            cut = max_chars if cut < 200 else cut + 1
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            out.append(p)
    return out


def top_passages(claim: str, passages: list[str], k: int = 5) -> list[int]:
    docs = [_tokens(p) for p in passages]
    q = _tokens(claim)
    if not docs or not q:
        return list(range(min(k, len(passages))))
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n or 1
    df = Counter(w for d in docs for w in set(d))
    scores = []
    for i, d in enumerate(docs):
        tf = Counter(d)
        s = 0.0
        for w in q:
            if w not in tf:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf[w] * 2.2 / (tf[w] + 1.2 * (0.25 + 0.75 * len(d) / avgdl))
        scores.append((s, i))
    scores.sort(reverse=True)
    return [i for _, i in scores[:k]]
