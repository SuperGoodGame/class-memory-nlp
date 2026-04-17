from __future__ import annotations

import json
import re
from typing import Any


SUMMARY_SCHEMA: dict[str, Any] = {
    "section_summary": "",
    "key_entities": [],
    "key_events": [],
    "exact_facts": [],
    "supporting_quotes": [],
}

SUMMARY_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "been",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def clean_json_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return cleaned.strip()


def _extract_list_field(text: str, key: str) -> list[str] | None:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*(\[[\s\S]*?\])', text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(1))
    except Exception:
        return re.findall(r'"([^"]+)"', match.group(1)) or None
    if not isinstance(parsed, list):
        return None
    return [str(item) for item in parsed]


def _extract_string_field(text: str, key: str) -> str | None:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"([\s\S]*?)"', text)
    if not match:
        return None
    return match.group(1)


def parse_summary_output(text: str) -> dict[str, Any]:
    cleaned = clean_json_text(text)
    try:
        parsed = json.loads(cleaned)
    except Exception:
        parsed = {}
        for key, default in SUMMARY_SCHEMA.items():
            extracted = _extract_list_field(cleaned, key) if isinstance(default, list) else _extract_string_field(
                cleaned, key
            )
            if extracted is not None:
                parsed[key] = extracted
        if not parsed:
            parsed = {"section_summary": text.strip()}

    normalized: dict[str, Any] = dict(SUMMARY_SCHEMA)
    for key, default in SUMMARY_SCHEMA.items():
        value = parsed.get(key, default)
        if isinstance(default, list):
            if isinstance(value, list):
                normalized[key] = [str(item).strip() for item in value if str(item).strip()]
            elif value in ("", None):
                normalized[key] = []
            else:
                normalized[key] = [str(value).strip()]
        else:
            normalized[key] = str(value).strip() if value is not None else ""
    return normalized


def summary_output_needs_retry(text: str, parsed_summary: dict[str, Any]) -> bool:
    cleaned = clean_json_text(text)
    if cleaned.startswith("{"):
        try:
            loaded = json.loads(cleaned)
        except Exception:
            return True
        if not isinstance(loaded, dict):
            return True
    return not parsed_summary["section_summary"] and not parsed_summary["exact_facts"]


def format_summary_record(summary: dict[str, Any]) -> str:
    lines = [f"Section summary: {summary.get('section_summary', '').strip()}"]
    for key in ["key_entities", "key_events", "exact_facts", "supporting_quotes"]:
        values = [str(item).strip() for item in summary.get(key, []) if str(item).strip()]
        if values:
            lines.append(f"{key}: " + "; ".join(values))
    return "\n".join(lines)


def tokenize_for_retrieval(text: str) -> list[str]:
    return [normalize_retrieval_token(token) for token in TOKEN_PATTERN.findall(text.lower())]


def normalize_retrieval_token(token: str) -> str:
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es") and not token.endswith("ses"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def extract_query_terms(question: str) -> list[str]:
    terms: list[str] = []
    for token in tokenize_for_retrieval(question):
        if token in SUMMARY_QUERY_STOPWORDS:
            continue
        if len(token) <= 2 and not token.isdigit():
            continue
        terms.append(token)
    return terms


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def keyword_overlap_score(question: str, candidate: str) -> float:
    query_terms = dedupe_preserve_order(extract_query_terms(question))
    if not query_terms:
        return 0.0

    candidate_tokens = set(tokenize_for_retrieval(candidate))
    if not candidate_tokens:
        return 0.0

    unigram_hits = sum(1 for term in query_terms if term in candidate_tokens)
    score = unigram_hits / len(query_terms)

    normalized_candidate = " ".join(tokenize_for_retrieval(candidate))
    bigrams = [" ".join(query_terms[idx : idx + 2]) for idx in range(len(query_terms) - 1)]
    if bigrams:
        bigram_hits = sum(1 for bigram in bigrams if bigram in normalized_candidate)
        score += 0.5 * (bigram_hits / len(bigrams))
    return score
