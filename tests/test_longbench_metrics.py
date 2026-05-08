from __future__ import annotations

import unittest

from nlp_baselines.evaluate_longbench import answer_score, is_answer_match, token_f1


class LongBenchMetricTest(unittest.TestCase):
    def test_substring_answer_match(self) -> None:
        self.assertTrue(is_answer_match("The answer is Alice.", ["Alice"]))

    def test_token_f1_handles_partial_paraphrase(self) -> None:
        self.assertGreater(token_f1("ground truth is not established", "Ground truth is not established in the paper"), 0.7)

    def test_unrelated_text_scores_low(self) -> None:
        self.assertLess(answer_score("I don't know.", ["the King"]), 0.35)


if __name__ == "__main__":
    unittest.main()
