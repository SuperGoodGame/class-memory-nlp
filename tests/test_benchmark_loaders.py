from __future__ import annotations

import unittest

from nlp_baselines.benchmark_loaders import normalize_longbench_record


class BenchmarkLoaderTest(unittest.TestCase):
    def test_normalizes_longbench_record(self) -> None:
        example = normalize_longbench_record(
            "narrativeqa",
            {
                "_id": "abc",
                "input": "Who is Alice?",
                "context": "Alice is a character.",
                "answers": ["a character"],
                "length": 4,
            },
            fallback_id="1",
        )
        self.assertEqual(example.example_id, "abc")
        self.assertEqual(example.question, "Who is Alice?")
        self.assertEqual(example.answers, ["a character"])
        self.assertEqual(example.length, 4)

    def test_normalizes_string_answer(self) -> None:
        example = normalize_longbench_record(
            "qasper",
            {"input": "Q", "context": "C", "answer": "A"},
            fallback_id="7",
        )
        self.assertEqual(example.example_id, "7")
        self.assertEqual(example.answers, ["A"])


if __name__ == "__main__":
    unittest.main()
