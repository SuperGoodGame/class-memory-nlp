from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TestCase:
    query: str
    expected_in_answer: list[str]
    expected_in_docs: list[str]
    description: str


EVAL_DATASET = [
    TestCase(
        query="Who is the main character of this story?",
        expected_in_answer=["Alice"],
        expected_in_docs=["Alice"],
        description="Main character identity",
    ),
    TestCase(
        query="Where does Alice fall down into?",
        expected_in_answer=["rabbit-hole", "rabbit hole"],
        expected_in_docs=["Rabbit-Hole", "rabbit-hole"],
        description="Setting - fall location",
    ),
    TestCase(
        query="What does the White Rabbit look like?",
        expected_in_answer=["pink eyes", "white rabbit"],
        expected_in_docs=["White Rabbit", "pink eyes"],
        description="Character description",
    ),
    TestCase(
        query="What does Alice drink that makes her shrink?",
        expected_in_answer=["drink", "bottle", "shrinks", "small"],
        expected_in_docs=["drink", "shrinks", "bottle"],
        description="Potion/drink that shrinks her",
    ),
    TestCase(
        query="What does Alice eat that makes her grow?",
        expected_in_answer=["cake", "eat", "fan", "grow", "large", "tall"],
        expected_in_docs=["cake", "eat", "fan", "grow"],
        description="Food that makes her grow",
    ),
    TestCase(
        query="Who does Alice meet first after falling down?",
        expected_in_answer=["Rabbit", "rabbit"],
        expected_in_docs=["Rabbit", "rabbit"],
        description="First character met",
    ),
    TestCase(
        query="What does the Cat say to Alice?",
        expected_in_answer=["Cheshire", "grin", "cat"],
        expected_in_docs=["Cheshire", "grin", "cat"],
        description="Cheshire Cat dialogue",
    ),
    TestCase(
        query="What is the name of the garden Alice enters?",
        expected_in_answer=["garden", "Garden"],
        expected_in_docs=["garden"],
        description="Garden name",
    ),
]


def _any_keyword_match(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)


def check_hit(retrieved_docs: list[str], expected_keywords: list[str]) -> bool:
    return any(_any_keyword_match(doc, expected_keywords) for doc in retrieved_docs)


def check_accuracy(answer: str, expected_keywords: list[str]) -> bool:
    return _any_keyword_match(answer, expected_keywords)


def pad(value: object, width: int, align: str = "<") -> str:
    text = str(value)
    return text.ljust(width) if align == "<" else text.rjust(width)
