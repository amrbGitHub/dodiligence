"""Where opinion text comes from: recorded CourtListener responses, or the live CourtListener API."""
from __future__ import annotations

import html
import json
import re
import time
from collections import OrderedDict
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
    source: str = ""          # "courtlistener" | "lookup_error" | "none"
    note: str = ""            # e.g. why a lookup failed


def _strip_html(s: str) -> str:
    s = re.sub(r"(?is)<(script|style).*?</\1>", " ", s)
    s = re.sub(r"(?i)</p\s*>|<br\s*/?>", "\n\n", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t]+", " ", s)
    return re.sub(r"\n\s*\n+", "\n\n", s).strip()


# CourtListener's free quota is small (this account: 125 requests a day), so successful responses
# are kept for the life of the server process and shared by every visitor: checking the same memo
# again, or another memo citing the same case, costs no requests.
_CACHE: OrderedDict[tuple, tuple[int, object]] = OrderedDict()
_CACHE_MAX = 500
# Where an opinion's text can be, best first (plain text, then the HTML variants).
TEXT_FIELDS = ("plain_text", "html_with_citations", "html", "html_lawbox", "html_columbia",
               "xml_harvard", "html_anon_2020")
_blocked_until = 0.0    # after a long throttle, don't send requests that will only be refused
MAX_RETRY_WAIT = 30     # seconds; a longer Retry-After means a daily limit, not worth waiting for


def _cached(key: tuple, fetch) -> tuple[int, object]:
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    status, body = fetch()
    if status == 200:
        _CACHE[key] = (status, body)
        if len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    return status, body


class HttpTransport:
    """Talks to the real CourtListener API."""

    def __init__(self, token: str, timeout: int = 30):
        token = token.strip().strip('"').strip()
        if token.lower().startswith("token "):     # pasted with CourtListener's own "Token " prefix
            token = token[6:].strip()
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Token {token}"
        self.timeout = timeout

    def _send(self, method: str, url: str, **kw) -> tuple[int, object]:
        """Returns (status, JSON body or text). Retries a short throttle once or twice; a long one is
        remembered, so later requests fail at once instead of hitting CourtListener again."""
        global _blocked_until
        wait = _blocked_until - time.time()
        if wait > 0:
            return 429, {"detail": f"Request was throttled. Expected available in {int(wait)} seconds."}
        for attempt in range(3):
            try:
                r = self.s.request(method, url, timeout=self.timeout, **kw)
            except requests.RequestException as e:
                return 0, f"{type(e).__name__}: {e}"
            if r.status_code != 429:
                break
            retry_after = _retry_after(r)
            if retry_after > MAX_RETRY_WAIT:
                _blocked_until = time.time() + retry_after
                break
            if attempt < 2:
                time.sleep(retry_after + 1)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, r.text[:200]

    def lookup(self, text: str) -> tuple[int, object]:
        return _cached(("lookup", text), lambda: self._send(
            "POST", f"{CL_BASE}/api/rest/v4/citation-lookup/", data={"text": text}))

    def get(self, url: str) -> tuple[int, dict]:
        def fetch():
            status, body = self._send("GET", url)
            if status == 200 and isinstance(body, dict):    # keep only what _opinion_text reads
                text_field = next((f for f in TEXT_FIELDS if (body.get(f) or "").strip()), None)
                body = {"type": body.get("type", ""), **({text_field: body[text_field]} if text_field else {})}
            return status, body
        status, body = _cached(("get", url), fetch)
        return status, (body if status == 200 and isinstance(body, dict) else {})


def _retry_after(r) -> int:
    """Seconds until CourtListener accepts requests again, from the header or the error text."""
    try:
        return int(r.headers.get("retry-after") or 0) or _seconds_in(r.json().get("detail", "")) or 10
    except (ValueError, AttributeError):
        return 10


def _seconds_in(detail: str) -> int:
    m = re.search(r"available in (\d+) seconds", detail or "")
    return int(m.group(1)) if m else 0


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


def _lookup_error(status: int, body) -> str:
    """Why a CourtListener lookup failed, in words a visitor (or the app's owner) can act on."""
    detail = (body.get("detail") if isinstance(body, dict) else str(body or "")).strip()[:160]
    if status in (401, 403):
        why = ("CourtListener rejected the API token, so this case couldn't be looked up. "
               "The app's owner should check COURTLISTENER_TOKEN in the app's Secrets.")
    elif status == 429:
        secs = _seconds_in(detail)
        when = ("later" if not secs else f"in about {round(secs / 3600, 1):g} hours" if secs >= 3600
                else f"in about {max(1, round(secs / 60))} minutes")
        why = (f"CourtListener's free API limit was reached, so this case couldn't be looked up. "
               f"Lookups work again {when}. Citations already checked on this server still work.")
    elif status == 0:
        why = "Couldn't reach CourtListener, so this case couldn't be looked up."
    else:
        why = f"CourtListener returned an error (HTTP {status}), so this case couldn't be looked up."
    return why + (f" ({detail})" if detail else "")


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
                for f in TEXT_FIELDS[1:]:
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
            note = _lookup_error(status, results)
            for c in cites:
                out[c] = Opinion(citation=c, found=False, source="lookup_error", note=note)
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
