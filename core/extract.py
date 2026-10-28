"""Pull citations, the claims they support, and any quotations out of a memo."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from eyecite import get_citations
from eyecite.models import FullCaseCitation

# Straight or curly double quotes, 15+ chars inside.
_QUOTE_RE = re.compile(r"[\"“]([^\"“”]{15,}?)[\"”]")
# A quotation introduced as coming from a statute, rule or document is not the cited case's own words.
_NOT_CASE = re.compile(
    r"\b(?:(?i:provides?|provided|reads?|stating|statute|statutory|regulations?|lease|contract|agreement|"
    r"section|clause|policy)|Act|Rule|Code|Amendment|Constitution|U\.S\.C|C\.F\.R)\b|§")


def _case_quotes(body: str) -> list[str]:
    out = []
    for m in _QUOTE_RE.finditer(body):
        lead = body[max(0, m.start() - 60):m.start()]
        lead = re.split(r"[.;]\s", lead)[-1]           # same sentence only
        if _NOT_CASE.search(lead):
            continue
        out.append(m.group(1).strip())
    return out


@dataclass
class Claim:
    idx: int
    citation: str            # normalized, e.g. "540 U.S. 443"
    case_name: str           # as written in the memo, e.g. "Kontrick v. Ryan"
    year: str | None
    claim_text: str          # the sentence(s) the citation is offered for
    quotes: list[str] = field(default_factory=list)
    span: tuple[int, int] = (0, 0)       # whole claim, including the citation
    cite_span: tuple[int, int] = (0, 0)  # just the citation ("Kontrick v. Ryan, 540 U.S. 443 (2004)")


def _case_name(c: FullCaseCitation) -> str:
    m = c.metadata
    if m.plaintiff and m.defendant:
        return f"{m.plaintiff} v. {m.defendant}"
    return m.defendant or m.plaintiff or ""


def merge_parallel(text: str, cites: list) -> list[tuple]:
    """Parallel citations ("540 U.S. 443, 124 S. Ct. 906") name one case. Keep the first, and stretch its
    span to cover the others. Returns [(citation, end_of_group)]."""
    out: list[list] = []
    last_core = 0
    for c in cites:
        if out:
            gap = text[last_core:c.span()[0]]
            if re.fullmatch(r"\s*(,\s*(at\s*)?\d+(\s*[-–]\s*\d+)?(\s*n\.\s*\d+)?)?\s*,\s*", gap):
                out[-1][1] = max(out[-1][1], c.full_span()[1])
                last_core = c.span()[1]
                continue
        out.append([c, c.full_span()[1]])
        last_core = c.span()[1]
    return [(c, e) for c, e in out]


def extract_claims(text: str) -> list[Claim]:
    """One Claim per full case citation.

    The claim is the text between the previous citation (or the start of the
    paragraph) and the end of this citation - i.e. the proposition the
    citation is being used to support.
    """
    cites = merge_parallel(text, [c for c in get_citations(text) if isinstance(c, FullCaseCitation)])
    claims: list[Claim] = []
    prev_end = 0
    for i, (c, end_override) in enumerate(cites):
        start, end = c.full_span()
        end = max(end, end_override)
        para_start = text.rfind("\n\n", 0, start)
        para_start = 0 if para_start == -1 else para_start + 2
        seg_start = max(prev_end, para_start)
        while seg_start < start and text[seg_start] in " .;,)\n":
            seg_start += 1
        claim_text = text[seg_start:end].strip()
        # Drop the citation itself from the end of the claim text.
        body = text[seg_start:start].strip()
        quotes = _case_quotes(body)
        claims.append(
            Claim(
                idx=i + 1,
                citation=c.corrected_citation(),
                case_name=_case_name(c),
                year=c.metadata.year,
                claim_text=claim_text,
                quotes=quotes,
                span=(seg_start, end),
                cite_span=(start, end),
            )
        )
        prev_end = end
    return claims
