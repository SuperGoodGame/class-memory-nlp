from __future__ import annotations

import unittest

from nlp_baselines.longbench_structured_memory import safe_example_id


class LongBenchStructuredMemoryTest(unittest.TestCase):
    def test_safe_example_id_preserves_simple_hashes(self) -> None:
        self.assertEqual(safe_example_id("abc123_DEF-4.5"), "abc123_DEF-4.5")

    def test_safe_example_id_replaces_path_separators(self) -> None:
        self.assertEqual(safe_example_id("../bad/id"), ".._bad_id")


if __name__ == "__main__":
    unittest.main()
