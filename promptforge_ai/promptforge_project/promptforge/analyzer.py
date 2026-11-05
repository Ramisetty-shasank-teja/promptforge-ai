"""Prompt analyzer: detects task type and which prompt-engineering elements are missing."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

TASK_KEYWORDS = {
    "image_edit": ["image", "photo", "picture", "edit this", "render", "retouch"],
    "code": ["code", "function", "python", "java", "sql", "script", "bug", "api", "program"],
    "data_analysis": ["data", "dataset", "sales", "analyze", "analyse", "trend", "csv", "dashboard"],
    "email": ["email", "mail", "letter", "message to"],
    "resume": ["resume", "cv", "cover letter"],
    "summary": ["summarize", "summarise", "summary", "tl;dr"],
    "content": ["blog", "article", "post", "caption", "story", "essay"],
}

# element -> regex that signals it is already present in the prompt
ELEMENT_PATTERNS = {
    "role": r"\b(act as|you are|as an?)\b",
    "audience": r"\b(for (a|an|the)|audience|reader|beginner|student|manager|customer|recruiter)\b",
    "context": r"\b(context|background|because|so that|goal|purpose|applying|scenario)\b",
    "format": r"\b(format|bullet|table|json|markdown|steps|sections|headings|paragraph)\b",
    "constraints": r"\b(do not|don't|must|never|only|avoid|preserve|without|at most|at least)\b",
    "length": r"\b(\d+\s*(words|sentences|lines|paragraphs|bullets)|short|brief|concise|detailed)\b",
    "tone": r"\b(professional|formal|friendly|casual|persuasive|polite|luxur\w+|technical)\b",
}

ELEMENT_WEIGHTS = {"role": 1, "audience": 1, "context": 2, "format": 2,
                   "constraints": 2, "length": 1, "tone": 1}


@dataclass
class Analysis:
    original: str
    task_type: str
    word_count: int
    present: dict
    missing: list
    clarity_score: float  # 0-100

    def to_dict(self) -> dict:
        return asdict(self)


def detect_task_type(prompt: str) -> str:
    text = prompt.lower()
    best, best_hits = "general", 0
    for task, words in TASK_KEYWORDS.items():
        hits = sum(1 for w in words if w in text)
        if hits > best_hits:
            best, best_hits = task, hits
    return best


def analyze_prompt(prompt: str) -> Analysis:
    prompt = prompt.strip()
    text = prompt.lower()
    present = {k: bool(re.search(p, text)) for k, p in ELEMENT_PATTERNS.items()}
    missing = [k for k, v in present.items() if not v]
    total = sum(ELEMENT_WEIGHTS.values())
    got = sum(ELEMENT_WEIGHTS[k] for k, v in present.items() if v)
    words = len(prompt.split())
    # reward elements, lightly reward length up to ~40 words
    score = 80 * got / total + 20 * min(words, 40) / 40
    return Analysis(prompt, detect_task_type(prompt), words, present, missing, round(score, 1))
