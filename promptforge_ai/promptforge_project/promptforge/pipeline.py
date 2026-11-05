"""End-to-end pipeline: analyze -> optimize -> execute -> evaluate -> recommend."""
from __future__ import annotations

from .analyzer import analyze_prompt
from .evaluator import evaluate, llm_judge
from .guardrails import Guardrails, check_constraint_preservation
from .llm import LLMClient, get_client
from .optimizer import build_variants, llm_rewrite
from .storage import Store


def run_pipeline(prompt: str, context: str = "", samples: int = 2,
                 client: LLMClient | None = None, guardrails: Guardrails | None = None,
                 store: Store | None = None, name: str | None = None,
                 use_llm_rewrite: bool = False, use_judge: bool = False) -> dict:
    client = client or get_client()
    guardrails = guardrails or Guardrails()
    analysis = analyze_prompt(prompt)

    input_violations = [v.__dict__ for v in guardrails.check_input(prompt)]
    if any(v["severity"] == "high" for v in input_violations):
        return {"blocked": True, "reason": "Input failed guardrails", "violations": input_violations}

    variants = build_variants(prompt, context, analysis)
    if use_llm_rewrite and client.name != "mock":
        variants.append(llm_rewrite(client, prompt, context))

    original_variant = {"name": "original", "technique": "user prompt (baseline)", "prompt": prompt}
    items = [original_variant] + [v.to_dict() for v in variants]

    results = []
    for it in items:
        outputs = [client.generate(it["prompt"]) for _ in range(max(1, samples))]
        score = evaluate(prompt, outputs, source=context, guardrails=guardrails)
        d = {**it, "output": outputs[0], "score": score.to_dict(),
             "constraint_violations": [v.__dict__ for v in check_constraint_preservation(prompt, it["prompt"])]}
        if use_judge:
            d["judge"] = llm_judge(client, prompt, outputs[0])
        results.append(d)

    # a variant that drops user constraints is disqualified from "best"
    eligible = [r for r in results if not r["constraint_violations"]] or results
    best = max(eligible, key=lambda r: r["score"]["overall"])
    base = results[0]["score"]["overall"]

    if store:
        pname = name or prompt[:40]
        for r in results:
            vid = store.add_version(pname, r["prompt"], r["name"])
            store.add_run(vid, client.name, r["output"], r["score"])

    return {"blocked": False, "model": client.name, "analysis": analysis.to_dict(),
            "results": results, "best": best["name"],
            "improvement_points": round(best["score"]["overall"] - base, 1)}
