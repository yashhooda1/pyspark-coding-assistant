"""Model adapters.

Three backends cover everything you actually need to score:

  ollama:<tag>       local Ollama server (your own models, GGUF quants)
  openai:<model>     any OpenAI-compatible /v1/chat/completions endpoint,
                     which includes vLLM, llama.cpp server, TGI, OpenRouter,
                     and the hosted frontier APIs -- set OPENAI_BASE_URL
  dummy:<mode>       no inference; for testing the harness itself

Adding a backend means implementing one method. Keep it that way.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

from .prompting import SYSTEM_PROMPT


class ModelError(RuntimeError):
    pass


class Model(ABC):
    name: str

    @abstractmethod
    def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Return the raw text response. Adapters do not extract code."""


def _post_json(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:500]
        raise ModelError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise ModelError(f"cannot reach {url}: {exc.reason}") from exc


class OllamaModel(Model):
    def __init__(self, tag: str, host: str | None = None, timeout: int = 300):
        self.name = f"ollama:{tag}"
        self.tag = tag
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = f"http://{self.host}"
        self.timeout = timeout

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        payload = {
            "model": self.tag,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                # Long fixtures plus a reasoning preamble overflow the 2k
                # default and the model silently loses the task statement.
                "num_ctx": 8192,
            },
        }
        data = _post_json(f"{self.host}/api/chat", payload, {}, self.timeout)
        return data.get("message", {}).get("content", "")


class OpenAICompatModel(Model):
    def __init__(self, model: str, base_url: str | None = None, timeout: int = 300):
        self.name = f"openai:{model}"
        self.model = model
        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = _post_json(
            f"{self.base_url}/chat/completions", payload, headers, self.timeout
        )
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise ModelError(f"unexpected response shape: {str(data)[:300]}") from exc


class DummyModel(Model):
    """Harness self-tests. Never talks to a model.

    reference  -> echo the gold solution. Every task must pass; if one fails,
                  the benchmark itself is broken.
    empty      -> return nothing. Every task must fail.
    """

    def __init__(self, mode: str = "reference"):
        self.name = f"dummy:{mode}"
        self.mode = mode
        self._solutions: dict[str, str] = {}

    def register(self, prompt_key: str, solution: str) -> None:
        self._solutions[prompt_key] = solution

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        if self.mode == "reference":
            return f"```python\n{self._solutions.get(prompt, '')}\n```"
        return ""


def build_model(spec: str, timeout: int = 300) -> Model:
    """Parse a `backend:name` spec into a Model."""
    if ":" not in spec:
        raise ValueError(
            f"model spec {spec!r} must look like 'ollama:qwen3:4b' or 'openai:gpt-4o-mini'"
        )
    backend, _, rest = spec.partition(":")
    backend = backend.lower()

    if backend == "ollama":
        return OllamaModel(rest, timeout=timeout)
    if backend in ("openai", "vllm", "openai-compat"):
        return OpenAICompatModel(rest, timeout=timeout)
    if backend == "dummy":
        return DummyModel(rest or "reference")
    raise ValueError(f"unknown backend {backend!r}")
