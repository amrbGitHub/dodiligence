"""Where opinion text comes from: recorded CourtListener responses, or the live CourtListener API."""
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

import requests

ROOT_EVAL = Path(__file__).resolve().parent.parent / "eval" / "data"
CL_BASE = "https://www.courtlistener.com"


@dataclass
class Opinion:
    citation: str
    found: bool
    case_name: str = ""
    court: str = ""
    year: str | int | None = None
    url: str = ""
    text: str = ""
    source: str = ""          # "cache" | "courtlistener"
    note: str = ""            # e.g. "multiple matches", "lookup error"


def _strip_html(s: str) -> str:
    s = re.sub(r"(?is)<(script|style).*?</\1>", " ", s)
    s = re.sub(r"(?i)</p\s*>|<br\s*/?>", "\n\n", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t]+", " ", s)
    return re.sub(r"\n\s*\n+", "\n\n", s).strip()


class HttpTransport:
    """Talks to the real CourtListener API."""

    def __init__(self, token: str, timeout: int = 30):
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Token {token}"
        self.timeout = timeout

    def _send(self, method: str, url: str, **kw):
        """CourtListener throttles new accounts hard (a few requests a minute), so wait and retry on 429."""
        import time
        for _ in range(5):
            r = self.s.request(method, url, timeout=self.timeout, **kw)
            if r.status_code != 429:
                return r
            time.sleep(min(30, int(r.headers.get("retry-after", "10") or 10)) + 1)
        return r

    def lookup(self, text: str) -> tuple[int, object]:
        r = self._send("POST", f"{CL_BASE}/api/rest/v4/citation-lookup/", data={"text": text})
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, r.text[:200]

    def get(self, url: str) -> tuple[int, dict]:
        r = self._send("GET", url)
        return r.status_code, (r.json() if r.status_code == 200 else {})


class ReplayTransport:
    """Serves recorded CourtListener responses (used by the benchmark, so results are reproducible)."""

    def __init__(self, recording: dict):
        self.lookups = recording["lookups"]          # citation -> lookup result item
        self.opinions = recording["opinions"]        # sub-opinion URL -> opinion JSON (trimmed)

    def lookup(self, text: str) -> tuple[int, object]:
        cites = [c.strip() for c in text.split("; ")]
        return 200, [self.lookups[c] for c in cites if c in self.lookups]

    def get(self, url: str) -> tuple[int, dict]:
        j = self.opinions.get(url)
        return (200, j) if j else (404, {})


def _name_match(a: str, b: str) -> float:
    from rapidfuzz import fuzz
    return fuzz.token_set_ratio(a.lower(), b.lower()) if a and b else 0.0


class CourtListenerSource:
    """Lookups via the CourtListener v4 API (free account token required)."""

    def __init__(self, token: str | None = None, timeout: int = 30, transport=None):
        self.t = transport or HttpTransport(token or "", timeout)

    def _opinion_text(self, sub_urls: list[str]) -> str:
        parts = []
        for u in sub_urls[:3]:  # majority + up to two more (concurrence/dissent)
            status, j = self.t.get(u)
            if status != 200:
                continue
            txt = j.get("plain_text") or ""
            if not txt.strip():
                for f in ("html_with_citations", "html", "html_lawbox", "html_columbia",
                          "xml_harvard", "html_anon_2020"):
                    if j.get(f):
                        txt = _strip_html(j[f])
                        break
            label = j.get("type", "")
            if txt.strip():
                parts.append(f"[{label}]\n{txt.strip()}" if label else txt.strip())
        return "\n\n".join(parts)

    def get_many(self, citations: list[str], names: dict[str, str] | None = None) -> dict[str, Opinion]:
        cites = list(dict.fromkeys(citations))
        names = names or {}
        out: dict[str, Opinion] = {}
        if not cites:
            return out
        status, results = self.t.lookup("; ".join(cites))
        if status != 200 or not isinstance(results, list):
            for c in cites:
                out[c] = Opinion(citation=c, found=False, source="courtlistener",
                                 note=f"lookup error {status}: {str(results)[:120]}")
            return out
        by_norm: dict[str, dict] = {}
        for item in results:
            keys = [item.get("citation", "")] + list(item.get("normalized_citations") or [])
            for k in keys:
                by_norm[re.sub(r"\s+", " ", k).strip()] = item
        for c in cites:
            item = by_norm.get(c)
            clusters = (item or {}).get("clusters") or []
            if not item or item.get("status") not in (200, 300) or not clusters:
                note = (item or {}).get("error_message") or ""
                out[c] = Opinion(citation=c, found=False, source="courtlistener", note=note)
                continue
            # Several cases can share a page (status 300): pick the one the memo names.
            cl = max(clusters, key=lambda k: _name_match(names.get(c, ""), k.get("case_name") or ""))
            subs = cl.get("sub_opinions") or []
            text = self._opinion_text(subs) if subs else ""
            date = cl.get("date_filed") or ""
            out[c] = Opinion(
                citation=c, found=True, case_name=cl.get("case_name") or cl.get("case_name_full", ""),
                court=cl.get("court") or "", year=date[:4], source="courtlistener",
                url=CL_BASE + cl.get("absolute_url", ""), text=text,
                note="" if text else "case found, but no opinion text available",
            )
        return out


def load_recording(path: Path | None = None) -> dict:
    import gzip
    path = path or (ROOT_EVAL / "recording.json.gz")
    if not path.exists():
        return {"lookups": {}, "opinions": {}}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


class CompositeSource:
    """Use recorded CourtListener responses when we have them (instant, free), else the live API."""

    def __init__(self, live: CourtListenerSource | None, recording: dict | None = None):
        self.recording = recording if recording is not None else load_recording()
        self.replay = CourtListenerSource(transport=ReplayTransport(self.recording))
        self.live = live

    def get_many(self, citations: list[str], names: dict[str, str] | None = None) -> dict[str, Opinion]:
        known = [c for c in citations if c in self.recording["lookups"]]
        unknown = [c for c in citations if c not in self.recording["lookups"]]
        res = self.replay.get_many(known, names) if known else {}
        if unknown:
            if self.live:
                res.update(self.live.get_many(unknown, names))
            else:
                for c in unknown:
                    res[c] = Opinion(citation=c, found=False, source="none",
                                     note="Live case lookups aren't configured on this deployment.")
        return res
