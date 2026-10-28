"""Tie it together: memo text in, one verdict per citation out."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .extract import Claim, extract_claims
from .judge import Judgment
from .match import QuoteResult, check_quote, split_passages, top_passages
from .sources import Opinion

# Final statuses, worst first.
STATUS = {
    "NOT_FOUND":     ("Case not found", "No case exists at this citation in the database."),
    "NAME_MISMATCH": ("Wrong case", "The citation points to a different case than the one named."),
    "MISQUOTED":     ("Misquoted", "The quoted words do not match the opinion."),
    "CONTRADICTED":  ("Contradicted", "The opinion says the opposite of the memo's claim."),
    "UNSUPPORTED":   ("Not supported", "The opinion doesn't back up the claim as stated."),
    "REVIEW":        ("Needs review", "Couldn't confirm automatically; a person should check."),
    "QUOTE_OK":      ("Quote verified", "Quote matches; support for the claim wasn't AI-checked."),
    "VERIFIED":      ("Verified", "Case exists, quotes match, and the opinion supports the claim."),
}


@dataclass
class CitationResult:
    claim: Claim
    opinion: Opinion
    status: str
    reason: str
    name_score: float = 100.0
    quotes: list[QuoteResult] = field(default_factory=list)
    excerpts: list[str] = field(default_factory=list)
    judgment: Judgment | None = None


_GENERIC = set("""v vs the of and in re ex rel matter estate state states united people commonwealth
county cnty city town township village dist district dep't dept department board bd comm'n commission
inc llc ltd co corp corporation company l.l.c. n.a. na lp llp plc p.c. pc assn ass'n association
bank national nat'l natl trust tr servs services service sys systems group grp ins insurance mut
fin financial home loan loans mortg mortgage auth authority govt gov't government court ct
of america u.s. us usa new york california florida texas""".split())


def _tokens(name: str) -> list[str]:
    import re as _re
    return [t for t in _re.findall(r"[a-z0-9']+", name.lower().replace("-", " ")) if len(t) > 1]


def _abbrev(a: str, b: str) -> bool:
    """True if `a` could abbreviate `b` (same first letter, letters appear in order): Int'l -> International."""
    a = a.replace("'", "")
    if len(a) < 3 or not b or a[0] != b[0]:
        return False
    it = iter(b)
    return all(ch in it for ch in a)


def _name_score(memo_name: str, db_name: str) -> float:
    """How well the case name in the memo matches the case at that citation (0-100).

    Generic words ("United States", "Inc.", "County") are ignored, so "United States v. Hazel"
    does not match "RYO Machine, LLC v. United States Department of Treasury".
    """
    if not memo_name or not db_name:
        return 100.0
    memo = [t for t in _tokens(memo_name) if t not in _GENERIC]
    db = _tokens(db_name)
    if not memo:
        return fuzz.token_set_ratio(memo_name.lower(), db_name.lower())
    initials = "".join(b[0] for b in db)
    hit = 0
    for t in memo:
        if any(t == b or fuzz.ratio(t, b) >= 85 or _abbrev(t, b) or _abbrev(b, t) for b in db) \
                or (len(t) >= 3 and t.isalpha() and t in initials):
            hit += 1
    return 100.0 * hit / len(memo)


def _excerpts(claim: Claim, op: Opinion, quotes: list[QuoteResult], k: int = 5) -> list[str]:
    passages = split_passages(op.text)
    if not passages:
        return []
    chosen: list[int] = []
    # 1) passages containing the best match for each quote
    for q in quotes:
        for i, p in enumerate(passages):
            if q.matched_text[:50] and q.matched_text[:50] in " ".join(p.split()):
                chosen.append(i)
                break
    # 2) the opening of the opinion (where holdings are usually summarized)
    chosen += [0, 1][: len(passages)]
    # 3) best keyword matches for the claim
    chosen += top_passages(claim.claim_text, passages, k=k)
    seen, ordered = set(), []
    for i in chosen:
        if i not in seen and i < len(passages):
            seen.add(i)
            ordered.append(i)
    budget, out = 14000, []
    for i in sorted(ordered[: k + 2]):
        if budget - len(passages[i]) < 0:
            break
        out.append(passages[i])
        budget -= len(passages[i])
    return out


def verify(text: str, source, judge=None, on_progress=None, max_cites: int | None = None) -> list[CitationResult]:
    claims = extract_claims(text)[:max_cites]
    opinions = source.get_many([c.citation for c in claims], {c.citation: c.case_name for c in claims})
    results: list[CitationResult] = []
    for n, cl in enumerate(claims, 1):
        if on_progress:
            on_progress(n, len(claims), cl)
        op = opinions[cl.citation]
        if not op.found and (op.source == "none" or op.note.startswith("lookup error")):
            results.append(CitationResult(cl, op, "REVIEW", op.note))
            continue
        if not op.found:
            results.append(CitationResult(cl, op, "NOT_FOUND", op.note or STATUS["NOT_FOUND"][1]))
            continue
        ns = _name_score(cl.case_name, op.case_name)
        if ns < 50:
            results.append(CitationResult(
                cl, op, "NAME_MISMATCH",
                f"The memo names “{cl.case_name}”, but {cl.citation} is {op.case_name.rstrip('.')}.", ns))
            continue
        if not op.text.strip():
            results.append(CitationResult(cl, op, "REVIEW", op.note or "No opinion text available.", ns))
            continue
        quotes = [check_quote(q, op.text) for q in cl.quotes]
        excerpts = _excerpts(cl, op, quotes)
        res = CitationResult(cl, op, "REVIEW", "", ns, quotes, excerpts)
        bad = [q for q in quotes if q.status != "exact"]
        if bad:
            q = bad[0]
            res.status = "MISQUOTED"
            res.reason = ("The quoted words don't match the opinion. The closest passage reads: "
                          f"“{q.matched_text}”" if q.status == "close"
                          else "This quotation does not appear anywhere in the opinion.")
        body = cl.claim_text[: max(0, len(cl.claim_text) - (cl.cite_span[1] - cl.cite_span[0]))]
        body = re.sub(r"\b(see also|see|e\.g\.|accord|cf\.)\b|[\W_]+", " ", body, flags=re.I).strip()
        if judge is not None and len(body) < 15 and res.status != "MISQUOTED":
            res.status, res.reason = "REVIEW", "The case exists, but there's no proposition next to the citation to check."
        elif judge is not None:
            try:
                j = judge.judge(cl.claim_text, op.case_name or cl.case_name, excerpts)
                res.judgment = j
            except Exception as e:  # network / parsing problems shouldn't sink the whole report
                res.reason = res.reason or f"AI check failed: {e}"
                results.append(res)
                continue
            if res.status != "MISQUOTED":
                res.status = {"SUPPORTED": "VERIFIED", "CONTRADICTED": "CONTRADICTED",
                              "NOT_SUPPORTED": "UNSUPPORTED"}.get(j.verdict, "REVIEW")
                res.reason = j.explanation
        elif res.status != "MISQUOTED":
            res.status = "QUOTE_OK" if quotes else "REVIEW"
            res.reason = STATUS[res.status][1]
        results.append(res)
    return results
