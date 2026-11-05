# PromptForge AI

Systematic prompt engineering: **analyze -> optimize -> execute -> evaluate -> recommend**, with guardrails,
hallucination checks and prompt version control.

## Quick start
```bash
pip install -r requirements.txt
python -m pytest -q                 # run tests
streamlit run app.py                # dashboard
uvicorn api:app --reload            # API (docs at /docs)
```
Runs offline with a mock model. For real results set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`
(optionally `PROMPTFORGE_MODEL`).

## How it works
| Module | Job |
|---|---|
| `analyzer.py` | Detects task type, finds missing elements (role, context, format, constraints...), clarity score |
| `optimizer.py` | Builds variants: basic (zero-shot), detailed (role-based), expert (chain-of-thought + self-check), fewshot, optional LLM meta-rewrite |
| `guardrails.py` | Prompt-injection and PII detection, forbidden phrases, **constraint preservation** (e.g. "do not modify gemstones") |
| `evaluator.py` | Relevance, completeness, consistency (multi-sample), safety, groundedness (numeric hallucination check), optional LLM-as-judge |
| `pipeline.py` | Orchestrates everything; variants that drop user constraints can never be recommended |
| `storage.py` | SQLite prompt history, versions, runs and diffs |

## Roadmap to "advanced"
1. Add your API key and compare real model outputs.
2. Test-suite mode: run each prompt over several inputs and average scores.
3. Replace heuristic groundedness with an LLM claim-verification step (RAG context).
4. Add multi-model comparison, cost/latency tracking and Plotly analytics.
5. Auth + Docker + deploy (Streamlit Cloud / Render).
