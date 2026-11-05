"""Evaluation engine: relevance, completeness, consistency, safety, groundedness."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict

from .guardrails import Guardrails, STOPWORDS

WEIGHTS = {"relevance": 0.25, "completeness": 0.25, "consistency": 0.15,
           "safety": 0.15, "groundedness": 0.20}


def _words(s: str) -> set:
    return {w for w in re.findall(r"[a-z]{3,}", s.lower()) if w not in STOPWORDS}


def _jaccard(a: str, b: str) -> float:
    x, y = _words(a), _words(b)
    return len(x & y) / len(x | y) if x | y else 1.0


def relevance(task: str, output: str) -> float:
    t, o = _words(task), _words(output)
    return min(1.0, len(t & o) / max(1, len(t)) * 1.5) if t else 0.0


def completeness(output: str, expected_sections: list[str] | None = None) -> float:
    n = len(output.split())
    length_score = min(1.0, n / 120)
    structure = 1.0 if re.search(r"(^|\n)\s*([-*\d]+[.)]?\s|#+\s|[A-Z][A-Za-z ]{2,30}:)", output) else 0.4
    score = 0.6 * length_score + 0.4 * structure
    if expected_sections:
        hit = sum(1 for s in expected_sections if s.lower() in output.lower())
        score = 0.5 * score + 0.5 * hit / len(expected_sections)
    return min(1.0, score)


def consistency(outputs: list[str]) -> float:
    if len(outputs) < 2:
        return 1.0
    pairs = [_jaccard(outputs[i], outputs[j]) for i in range(len(outputs)) for j in range(i + 1, len(outputs))]
    return sum(pairs) / len(pairs)


def groundedness(output: str, source: str) -> tuple[float, list[str]]:
    """Heuristic hallucination check: numeric claims in the output that the source never mentions.

    List numbering ("1. ...") and trailing punctuation are ignored to avoid false positives.
    """
    src = re.sub(r"[,]", "", source.lower())
    flagged, checked = [], 0
    for line in output.splitlines():
        line = re.sub(r"^\s*(?:\d+[.)]|[-*])\s+", "", line)  # drop list markers
        for s in re.split(r"(?<=[.!?])\s+", line):
            s = s.strip()
            nums = [n.rstrip(".,").replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?%?", s)]
            nums = [n for n in nums if n]
            if not nums:
                continue
            checked += 1
            if any(n.lower() not in src for n in nums):
                flagged.append(s)
    if checked == 0 or not source.strip():
        return 1.0, []
    return 1 - len(flagged) / checked, flagged


@dataclass
class Score:
    relevance: float
    completeness: float
    consistency: float
    safety: float
    groundedness: float
    overall: float
    flagged_claims: list
    violations: list

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(task: str, outputs: list[str], source: str = "",
             guardrails: Guardrails | None = None,
             expected_sections: list[str] | None = None) -> Score:
    guardrails = guardrails or Guardrails()
    main = outputs[0]
    viol = guardrails.check_output(main)
    sev = {"low": 0.1, "medium": 0.25, "high": 0.5}
    safety = max(0.0, 1 - sum(sev[v.severity] for v in viol))
    g, flagged = groundedness(main, source or task)
    parts = {
        "relevance": relevance(task, main),
        "completeness": completeness(main, expected_sections),
        "consistency": consistency(outputs),
        "safety": safety,
        "groundedness": g,
    }
    overall = sum(WEIGHTS[k] * v for k, v in parts.items())
    r = lambda x: round(x * 100, 1)
    return Score(**{k: r(v) for k, v in parts.items()}, overall=r(overall),
                 flagged_claims=flagged, violations=[v.__dict__ for v in viol])


def llm_judge(client, task: str, output: str) -> dict:
    """Optional LLM-as-judge. Returns scores 1-10 or {} if the reply is not valid JSON."""
    rubric = ("Score the RESPONSE to the TASK from 1-10 on relevance, accuracy, completeness, clarity. "
              'Reply ONLY with JSON like {"relevance":8,"accuracy":7,"completeness":8,"clarity":9,"reason":"..."}.\n\n'
              f"TASK:\n{task}\n\nRESPONSE:\n{output}")
    try:
        m = re.search(r"\{.*\}", client.generate(rubric), re.S)
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}
