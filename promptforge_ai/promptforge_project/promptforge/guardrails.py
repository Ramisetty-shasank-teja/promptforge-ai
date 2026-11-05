"""Guardrails: input safety, constraint preservation and output policy checks."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) instructions",
    r"disregard (the )?(system|previous) prompt",
    r"reveal (your )?(system prompt|hidden instructions)",
    r"you are now (dan|in developer mode)",
]
PII_PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "phone": r"(?<!\d)(?:\+?\d{1,3}[ -]?)?\d{10}(?!\d)",
    "aadhaar_like": r"(?<!\d)\d{4}[ -]\d{4}[ -]\d{4}(?!\d)",
    "card_like": r"(?<!\d)(?:\d[ -]?){15}\d(?!\d)",
}
STOPWORDS = set("the a an and or to of in on for with it its this that be is are as at by from do not "
                "dont never must only without any all".split())


@dataclass
class Violation:
    rule: str
    detail: str
    severity: str = "medium"  # low | medium | high


@dataclass
class Guardrails:
    forbidden_phrases: list = field(default_factory=list)
    required_phrases: list = field(default_factory=list)
    max_words: int | None = None

    # ---- input side ------------------------------------------------------
    def check_input(self, text: str) -> list[Violation]:
        out = []
        for p in INJECTION_PATTERNS:
            if re.search(p, text, re.I):
                out.append(Violation("prompt_injection", f"matched /{p}/", "high"))
        for name, p in PII_PATTERNS.items():
            if re.search(p, text):
                out.append(Violation("pii", f"possible {name} found", "medium"))
        return out

    # ---- output side -----------------------------------------------------
    def check_output(self, text: str) -> list[Violation]:
        out = []
        low = text.lower()
        for ph in self.forbidden_phrases:
            if ph.lower() in low:
                out.append(Violation("forbidden_phrase", f"'{ph}' appeared", "high"))
        for ph in self.required_phrases:
            if ph.lower() not in low:
                out.append(Violation("missing_required", f"'{ph}' missing", "medium"))
        if self.max_words and len(text.split()) > self.max_words:
            out.append(Violation("too_long", f"{len(text.split())} > {self.max_words} words", "low"))
        for name, p in PII_PATTERNS.items():
            if re.search(p, text):
                out.append(Violation("pii_leak", f"output contains possible {name}", "high"))
        return out


def extract_constraints(prompt: str) -> list[str]:
    """Hard rules the user wrote ('do not ...', 'preserve ...', 'must ...')."""
    sents = re.split(r"(?<=[.!?])\s+", prompt.strip())
    return [s for s in sents if re.search(r"\b(do not|don't|never|must|preserve|only|without)\b", s, re.I)]


def _keywords(s: str) -> set:
    return {w for w in re.findall(r"[a-z]{3,}", s.lower()) if w not in STOPWORDS}


def check_constraint_preservation(original: str, optimized: str) -> list[Violation]:
    """Optimization must never silently drop the user's constraints."""
    out = []
    opt_words = _keywords(optimized)
    for c in extract_constraints(original):
        kw = _keywords(c)
        if kw and len(kw & opt_words) / len(kw) < 0.8:
            out.append(Violation("constraint_dropped", f"optimized prompt may lose: '{c}'", "high"))
    return out
