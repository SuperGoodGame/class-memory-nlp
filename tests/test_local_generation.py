from __future__ import annotations

import unittest

from nlp_baselines.local_generation import (
    extract_nli_label,
    infer_nli_label_from_text,
)


class LocalGenerationParsingTest(unittest.TestCase):
    def test_extracts_exact_nli_label(self) -> None:
        self.assertEqual(extract_nli_label("entailment"), "entailment")

    def test_extracts_label_from_wrapped_text(self) -> None:
        self.assertEqual(extract_nli_label("LABEL: contradiction."), "contradiction")

    def test_infers_entailment_from_reasoning_only_text(self) -> None:
        text = "The premise is highly likely to support the hypothesis."
        self.assertEqual(infer_nli_label_from_text(text), "entailment")

    def test_empty_text_has_no_label(self) -> None:
        self.assertIsNone(extract_nli_label(""))
        self.assertIsNone(infer_nli_label_from_text(""))


if __name__ == "__main__":
    unittest.main()
