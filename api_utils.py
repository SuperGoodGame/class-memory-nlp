from __future__ import annotations

import os
import time
from dataclasses import dataclass

import requests
from dotenv import load_dotenv


load_dotenv()


DEFAULT_DASHSCOPE_CHAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_DASHSCOPE_MODEL = "qwen3.6-plus"
DEFAULT_AZURE_CHAT_URL = (
    "https://hkust.azure-api.net/openai/deployments/"
    "gpt-4o-mini/chat/completions?api-version=2025-02-01-preview"
)
OPENAI_COMPATIBLE_PROVIDERS = {"dashscope", "openai", "openai_compatible", "fluxcode"}


@dataclass
class ChatResponse:
    text: str
    latency: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


def resolve_chat_provider() -> str:
    explicit_provider = (os.getenv("CHAT_API_PROVIDER") or "").strip().lower()
    if explicit_provider:
        return explicit_provider

    if os.getenv("DASHSCOPE_API_KEY") or os.getenv("BAILIAN_API_KEY"):
        return "dashscope"

    return "azure"


def resolve_chat_api_key() -> str:
    provider = resolve_chat_provider()
    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("BAILIAN_API_KEY") or os.getenv("OPENAI_API_KEY")
    else:
        api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set the provider-specific API key before running chat scripts."
        )
    return api_key


def resolve_chat_api_url() -> str:
    explicit_url = os.getenv("CHAT_API_URL") or os.getenv("AZURE_OPENAI_API_URL")
    if explicit_url:
        return explicit_url

    provider = resolve_chat_provider()
    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        base_url = (os.getenv("DASHSCOPE_BASE_URL") or "").strip()
        if base_url:
            if base_url.endswith("/chat/completions"):
                return base_url
            return base_url.rstrip("/") + "/chat/completions"
        return DEFAULT_DASHSCOPE_CHAT_URL

    endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip()
    deployment = (os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-02-01-preview")
    if endpoint and deployment and "YOUR_RESOURCE_NAME" not in endpoint:
        return (
            f"{endpoint.rstrip('/')}/openai/deployments/"
            f"{deployment}/chat/completions?api-version={api_version}"
        )

    return DEFAULT_AZURE_CHAT_URL


def resolve_chat_model() -> str | None:
    explicit_model = (os.getenv("CHAT_MODEL") or "").strip()
    if explicit_model:
        return explicit_model

    provider = resolve_chat_provider()
    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        return (os.getenv("DASHSCOPE_MODEL") or DEFAULT_DASHSCOPE_MODEL).strip()
    return None


def describe_chat_target() -> str:
    provider = resolve_chat_provider()
    model = resolve_chat_model()
    if model:
        return f"{provider}:{model}"
    return f"{provider}:{resolve_chat_api_url()}"


def build_chat_headers() -> dict[str, str]:
    provider = resolve_chat_provider()
    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {resolve_chat_api_key()}",
        }
    return {
        "Content-Type": "application/json",
        "api-key": resolve_chat_api_key(),
    }


def call_chat_api(
    prompt: str,
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: int = 60,
) -> str:
    response = chat_completion(
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )
    return response.text


def chat_completion(
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: int = 60,
) -> ChatResponse:
    headers = build_chat_headers()
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    model = resolve_chat_model()
    if model:
        payload["model"] = model
    start = time.time()
    response = requests.post(resolve_chat_api_url(), headers=headers, json=payload, timeout=timeout)
    latency = time.time() - start
    if not response.ok:
        snippet = response.text[:500].strip()
        raise RuntimeError(
            f"Chat API request failed with status {response.status_code}: {snippet}"
        )

    parsed = response.json()
    usage = parsed.get("usage", {})
    return ChatResponse(
        text=parsed["choices"][0]["message"]["content"],
        latency=latency,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
    )
