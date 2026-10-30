"""LLM check: does the cited opinion actually support the claim?"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_EFFORT = "medium"   # reasons before answering; None answers straight away (cheaper, much less accurate)
PROMPT_VERSION = 4   # part of the cache key, so changing the instructions re-asks the model

SYSTEM = """You are a meticulous legal citation checker. A lawyer will rely on your answer, so be precise and conservative.

You receive (1) a proposition from a legal memo and the case it cites, and (2) numbered excerpts from that case's opinion.
Decide whether the opinion supports the proposition AS STATED.

Rules:
- Judge only from the excerpts. Do not use outside knowledge of the case.
- Watch for excerpts that describe a party's argument, a lower court's view, or a position the Court rejected. Those do not support a proposition; if the Court rejected it, the proposition is CONTRADICTED.
- A proposition that overstates, reverses, or changes the scope of the holding is not supported.
- Judge the core legal point the citation is offered for. Paraphrase and fair inference are fine; the opinion need not use the same words. With "see, e.g." the case only has to illustrate the point, so a detail the case doesn't mention is not by itself a reason to answer NOT_SUPPORTED.
- The proposition may apply a rule to the facts or parties of the present dispute ("Here, the amendments are redundant. See X."). Those facts won't be in the cited opinion; judge whether the opinion supports the rule or reasoning being applied, not the facts.
- If the citation has a parenthetical describing the case, check the parenthetical against the opinion; the case then only needs to illustrate the proposition in the way the parenthetical says.
- You see a few excerpts, not the whole opinion: its opening and the passages that best match the proposition.
- Answer NOT_SUPPORTED when the excerpts address the point and say something different: they overstate, narrow, or change the holding, attribute to the Court what it did not decide, or reach the point and decline to adopt it.
- When the excerpts don't discuss the point at all, decide which of two situations this is:
  - OFF_TOPIC: the proposition is a substantive rule of law, and the opinion is about a different subject altogether (for example, a patent case cited for a rule of criminal sentencing). Such a case can't be authority for the rule.
  - NOT_ADDRESSED: the opinion is on a related subject, or the proposition is a point any case can stand for (a standard of review, a pleading or summary-judgment standard, a procedural rule, a general definition), so the opinion may address it in a part you weren't shown.
- If the excerpts touch the point but are insufficient to decide, answer UNCLEAR.

Respond with JSON only, no prose before or after:
{"verdict": "SUPPORTED" | "CONTRADICTED" | "NOT_SUPPORTED" | "OFF_TOPIC" | "NOT_ADDRESSED" | "UNCLEAR",
 "explanation": "<= 2 short sentences, plain English, citing the excerpt number",
 "key_excerpt": <excerpt number that best shows the answer, or 0>}"""


@dataclass
class Judgment:
    verdict: str
    explanation: str
    key_excerpt: int
    model: str
    cached: bool = False


def _prompt(claim: str, case_name: str, excerpts: list[str]) -> str:
    ex = "\n\n".join(f"[{i}] {e}" for i, e in enumerate(excerpts, 1))
    return f"PROPOSITION (from the memo):\n{claim}\n\nCITED CASE: {case_name}\n\nEXCERPTS FROM THE OPINION:\n{ex}"


def cache_key(claim: str, case_name: str, excerpts: list[str], model: str, effort: str | None = None) -> str:
    h = hashlib.sha256(json.dumps([claim, case_name, excerpts, model, PROMPT_VERSION, effort]).encode()).hexdigest()
    return h[:32]


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in model output: {text[:200]}")
    return json.loads(m.group(0))


class ClaudeJudge:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, store: dict | None = None,
                 effort: str | None = DEFAULT_EFFORT):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key, max_retries=4)
        self.model = model
        self.effort = effort or None
        self.store = store if store is not None else {}
        self.used: set[str] = set()   # cache keys this run needed
        self.calls = 0          # verdicts given (including cached ones)
        self.api_calls = 0      # verdicts that needed a new API request
        self.input_tokens = 0
        self.output_tokens = 0

    def judge(self, claim: str, case_name: str, excerpts: list[str]) -> Judgment:
        key = cache_key(claim, case_name, excerpts, self.model, self.effort)
        self.used.add(key)
        if key in self.store:
            d = self.store[key]
            # count what the verdict cost when it was made, so cached benchmark runs still report cost
            self.calls += 1
            self.input_tokens += d.get("in", 0)
            self.output_tokens += d.get("out", 0)
            return Judgment(d["verdict"], d["explanation"], d.get("key_excerpt", 0), self.model, cached=True)
        msg = self.client.messages.create(
            model=self.model,
            **({"max_tokens": 500, "thinking": {"type": "disabled"}} if not self.effort else
               {"max_tokens": 16000, "thinking": {"type": "adaptive"}, "output_config": {"effort": self.effort}}),
            system=SYSTEM,
            messages=[{"role": "user", "content": _prompt(claim, case_name, excerpts)}],
        )
        self.calls += 1
        self.api_calls += 1
        self.input_tokens += msg.usage.input_tokens
        self.output_tokens += msg.usage.output_tokens
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        d = _parse(text)
        verdict = str(d.get("verdict", "UNCLEAR")).upper()
        if verdict not in {"SUPPORTED", "CONTRADICTED", "NOT_SUPPORTED", "OFF_TOPIC", "NOT_ADDRESSED", "UNCLEAR"}:
            verdict = "UNCLEAR"
        d = {"verdict": verdict, "explanation": str(d.get("explanation", "")).strip(),
             "key_excerpt": int(d.get("key_excerpt") or 0),
             "in": msg.usage.input_tokens, "out": msg.usage.output_tokens}
        self.store[key] = d
        return Judgment(d["verdict"], d["explanation"], d["key_excerpt"], self.model)
