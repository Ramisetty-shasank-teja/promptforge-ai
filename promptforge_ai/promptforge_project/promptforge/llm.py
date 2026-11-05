"""LLM clients. Stdlib only. Falls back to a deterministic MockLLM so the app runs offline.

Set ANTHROPIC_API_KEY (and optionally PROMPTFORGE_MODEL) or OPENAI_API_KEY to use a real model.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request


class LLMClient:
    name = "base"

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        raise NotImplementedError


class MockLLM(LLMClient):
    """Deterministic fake model: richer prompts -> more structured answers. For demos and tests."""
    name = "mock"

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        words = len(prompt.split())
        task_m = re.search(r"Task:\s*(.+)", prompt)
        task = (task_m.group(1) if task_m else prompt.splitlines()[0]).strip()
        lines = [f"Response to: {task}"]
        if "Output format" in prompt:
            lines += ["", "Summary:", f"- Addressing the request: {task}",
                      "- Following the requested structure and tone."]
        if "Constraints" in prompt:
            lines.append("- Assumptions are stated; no unsupported facts are added.")
        if "FINAL ANSWER" in prompt:
            lines += ["", "FINAL ANSWER:", f"A complete, verified answer for: {task}"]
        if words > 40:
            lines += ["", "Details:", "1. Key point one.", "2. Key point two.", "3. Suggested next step."]
        return "\n".join(lines)


class AnthropicLLM(LLMClient):
    name = "anthropic"

    def __init__(self, api_key: str, model: str):
        self.api_key, self.model = api_key, model

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        body = json.dumps({"model": self.model, "max_tokens": 1024, "temperature": temperature,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body, method="POST",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
        return "".join(b.get("text", "") for b in data.get("content", []))


class OpenAILLM(LLMClient):
    name = "openai"

    def __init__(self, api_key: str, model: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key, self.model, self.base_url = api_key, model, base_url

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        body = json.dumps({"model": self.model, "temperature": temperature,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)["choices"][0]["message"]["content"]


def get_client() -> LLMClient:
    model = os.getenv("PROMPTFORGE_MODEL")
    if os.getenv("ANTHROPIC_API_KEY"):
        return AnthropicLLM(os.environ["ANTHROPIC_API_KEY"], model or "claude-sonnet-4-5")
    if os.getenv("OPENAI_API_KEY"):
        return OpenAILLM(os.environ["OPENAI_API_KEY"], model or "gpt-4o-mini")
    return MockLLM()
