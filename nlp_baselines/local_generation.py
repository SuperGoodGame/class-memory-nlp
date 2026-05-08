from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests


DEFAULT_LOCAL_API_BASE = "http://127.0.0.1:8012/v1"
DEFAULT_LOCAL_MODEL = "gemma4-31b-gguf"
NLI_LABELS = ("entailment", "contradiction", "neutral")
_THREAD_LOCAL = threading.local()


@dataclass
class LocalChatResponse:
    text: str
    latency: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    finish_reason: str
    reasoning_content: str
    raw: dict[str, Any]
    empty_content_fallback: str | None = None


@dataclass
class NLIResult:
    label: str
    source: str
    response: LocalChatResponse


def local_api_base() -> str:
    return os.getenv("LOCAL_GEMMA4_API_BASE", DEFAULT_LOCAL_API_BASE).rstrip("/")


def local_model_name() -> str:
    return os.getenv("LOCAL_GEMMA4_MODEL", DEFAULT_LOCAL_MODEL)


def local_chat_completion(
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    timeout: int = 180,
    disable_thinking: bool = True,
    retry_on_empty: bool = True,
) -> LocalChatResponse:
    """Call the local llama.cpp OpenAI-compatible chat API.

    Gemma 4 served by llama.cpp can spend a short request entirely inside
    reasoning tokens, leaving `message.content` empty. The correct server-side
    control is the Jinja chat kwarg `enable_thinking=false`.
    """

    response = _post_chat(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        disable_thinking=disable_thinking,
    )
    if response.text.strip() or not retry_on_empty:
        return response

    retry = _post_chat(
        messages,
        max_tokens=max(max_tokens * 2, 64),
        temperature=temperature,
        timeout=timeout,
        disable_thinking=True,
    )
    if retry.text.strip():
        retry.empty_content_fallback = "retry_disable_thinking"
        return retry

    fallback_text = retry.reasoning_content.strip() or response.reasoning_content.strip()
    if fallback_text:
        retry.text = fallback_text
        retry.empty_content_fallback = "reasoning_content"
    return retry


def classify_nli(
    *,
    premise: str,
    hypothesis: str,
    timeout: int = 180,
) -> NLIResult:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an NLI classifier. Return exactly one label: "
                "entailment, contradiction, or neutral. Do not explain."
            ),
        },
        {
            "role": "user",
            "content": f"Premise:\n{premise}\n\nHypothesis:\n{hypothesis}\n\nLabel:",
        },
    ]
    response = local_chat_completion(
        messages,
        max_tokens=16,
        temperature=0.0,
        timeout=timeout,
        disable_thinking=True,
    )
    label = extract_nli_label(response.text)
    if label:
        return NLIResult(label=label, source="content", response=response)

    label = extract_nli_label(response.reasoning_content)
    if label:
        return NLIResult(label=label, source="reasoning_content", response=response)

    label = infer_nli_label_from_text(response.text + "\n" + response.reasoning_content)
    if label:
        return NLIResult(label=label, source="heuristic", response=response)

    return NLIResult(label="neutral", source="default_neutral", response=response)


def extract_nli_label(text: str) -> str | None:
    normalized = text.strip().lower()
    if not normalized:
        return None
    for label in NLI_LABELS:
        if re.fullmatch(rf"[\s\W_]*{label}[\s\W_]*", normalized):
            return label

    matches = re.findall(r"\b(entailment|contradiction|neutral)\b", normalized)
    if matches:
        return matches[-1]
    return None


def infer_nli_label_from_text(text: str) -> str | None:
    normalized = text.lower()
    if not normalized.strip():
        return None
    contradiction_markers = [
        "contradict",
        "cannot be true",
        "inconsistent",
        "opposite",
    ]
    entailment_markers = [
        "entail",
        "supports",
        "implies",
        "highly likely",
        "same meaning",
        "consistent with",
    ]
    if any(marker in normalized for marker in contradiction_markers):
        return "contradiction"
    if any(marker in normalized for marker in entailment_markers):
        return "entailment"
    return None


def _post_chat(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    timeout: int,
    disable_thinking: bool,
) -> LocalChatResponse:
    payload: dict[str, Any] = {
        "model": local_model_name(),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    start = time.time()
    http_response = _local_session().post(
        f"{local_api_base()}/chat/completions",
        json=payload,
        timeout=timeout,
    )
    latency = time.time() - start
    if not http_response.ok:
        raise RuntimeError(
            f"Local Gemma4 request failed with status {http_response.status_code}: "
            f"{http_response.text[:500]}"
        )

    parsed = http_response.json()
    choice = parsed["choices"][0]
    message = choice.get("message", {})
    usage = parsed.get("usage", {})
    return LocalChatResponse(
        text=str(message.get("content") or ""),
        latency=latency,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        finish_reason=str(choice.get("finish_reason") or ""),
        reasoning_content=str(message.get("reasoning_content") or ""),
        raw=parsed,
    )


def _local_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.trust_env = False
        _THREAD_LOCAL.session = session
    return session
