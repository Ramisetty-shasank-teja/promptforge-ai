"""Prompt optimizer: produces systematically different prompt variants.

Variants map to real prompt-engineering techniques so you can explain them in interviews:
  basic    -> clarified zero-shot
  detailed -> role + context + format + constraints (role-based, structured)
  expert   -> detailed + chain-of-thought plan + self-check + structured output
  fewshot  -> detailed + worked example (few-shot)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from .analyzer import Analysis, analyze_prompt

ROLES = {
    "email": "an experienced professional communications specialist",
    "resume": "an experienced technical recruiter",
    "code": "a senior software engineer who writes clean, tested code",
    "data_analysis": "a senior data analyst",
    "image_edit": "a professional retoucher and art director",
    "summary": "an expert editor who writes precise summaries",
    "content": "a skilled content writer",
    "general": "a domain expert",
}

FORMATS = {
    "email": "a subject line, greeting, short body (3-5 sentences) and sign-off",
    "resume": "an ATS-friendly layout with Summary, Skills, Projects, Education and measurable achievements",
    "code": "a short explanation, the code in one block, and 2-3 example usages or tests",
    "data_analysis": "key findings as bullets, a table of metrics, and 3 recommended next steps",
    "image_edit": "a precise edit specification covering lighting, background and detail",
    "summary": "a one-line takeaway followed by 3-5 bullets",
    "content": "a headline, short intro, clearly labelled sections and a closing call to action",
    "general": "clearly labelled sections with concise bullets",
}

DEFAULT_CONSTRAINTS = ("Do not invent facts, numbers or sources. "
                       "If information is missing, state your assumption explicitly.")

EXAMPLES = {
    "email": ("Input: request a day off for a family event.\n"
              "Output: Subject: Leave Request - 12 March\n"
              "Dear Ms. Rao, I would like to request leave on 12 March to attend a family event. "
              "I will complete pending tasks beforehand. Kind regards, Asha"),
    "general": ("Input: a short topic.\nOutput: a one-line takeaway then supporting bullets."),
}


@dataclass
class Variant:
    name: str
    technique: str
    prompt: str

    def to_dict(self) -> dict:
        return asdict(self)


def _original_constraints(original: str) -> str:
    """Carry over the user's own hard rules verbatim (e.g. 'do not modify gemstones')."""
    import re
    sents = re.split(r"(?<=[.!?])\s+", original.strip())
    keep = [s for s in sents if re.search(r"\b(do not|don't|never|must|preserve|only|without)\b", s, re.I)]
    return " ".join(keep)


def build_variants(original: str, context: str = "", analysis: Analysis | None = None) -> list[Variant]:
    analysis = analysis or analyze_prompt(original)
    t = analysis.task_type
    role, fmt = ROLES[t], FORMATS[t]
    user_rules = _original_constraints(original)
    rules = (user_rules + " " if user_rules else "") + DEFAULT_CONSTRAINTS
    ctx = f"\nContext (use only this information as ground truth):\n{context.strip()}\n" if context.strip() else ""

    basic = Variant("basic", "clarified zero-shot",
                    f"{original.strip()}\nBe clear, specific and concise.")

    detailed = Variant(
        "detailed", "role-based + structured",
        f"Act as {role}.\nTask: {original.strip()}\n{ctx}"
        f"Tone: professional and clear.\nOutput format: {fmt}.\nConstraints: {rules}")

    expert = Variant(
        "expert", "chain-of-thought plan + self-check + structured output",
        f"You are {role}.\nTask: {original.strip()}\n{ctx}"
        "Process: (1) briefly list what the task requires, (2) note any missing information and "
        "your assumptions, (3) produce the final answer, (4) verify it against the constraints.\n"
        f"Output format: {fmt}. Put the final answer under the heading 'FINAL ANSWER'.\n"
        f"Constraints: {rules}\nTone: professional, precise, no filler.")

    example = EXAMPLES.get(t, EXAMPLES["general"])
    fewshot = Variant(
        "fewshot", "few-shot + role-based",
        f"Act as {role}.\nTask: {original.strip()}\n{ctx}"
        f"Here is an example of the expected style:\n{example}\n"
        f"Now do the task. Output format: {fmt}.\nConstraints: {rules}")

    return [basic, detailed, expert, fewshot]


def llm_rewrite(client, original: str, context: str = "") -> Variant:
    """Optional: ask an LLM to rewrite the prompt (meta-prompting)."""
    meta = ("You are a prompt engineer. Rewrite the prompt below to be specific, structured and "
            "unambiguous. Keep every hard constraint the user stated. Return only the new prompt.\n\n"
            f"PROMPT:\n{original}\n" + (f"\nCONTEXT:\n{context}" if context else ""))
    return Variant("llm_rewrite", "meta-prompting", client.generate(meta).strip())
