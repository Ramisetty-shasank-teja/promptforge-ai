"""FastAPI backend.  Run: uvicorn api:app --reload"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from promptforge.analyzer import analyze_prompt
from promptforge.pipeline import run_pipeline
from promptforge.storage import Store

app = FastAPI(title="PromptForge AI", version="0.1.0")
store = Store()


class OptimizeRequest(BaseModel):
    prompt: str
    context: str = ""
    samples: int = 2
    name: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: OptimizeRequest):
    return analyze_prompt(req.prompt).to_dict()


@app.post("/optimize")
def optimize(req: OptimizeRequest):
    return run_pipeline(req.prompt, req.context, req.samples, store=store, name=req.name)


@app.get("/prompts")
def prompts():
    return store.list_prompts()


@app.get("/prompts/{name}/history")
def history(name: str):
    h = store.history(name)
    if not h:
        raise HTTPException(404, "unknown prompt")
    return h
