"""Unit tests for benchmark metrics (synthetic GT/pred; no OCR required)."""

from __future__ import annotations

import unittest

from benchmark.metrics import (
    aggregate,
    ascii_digit_contamination,
    cer_normalized,
    cer_strict,
    conjunct_error_rate,
    exact_match,
    find_conjunct_occurrences,
    is_reference_like,
    percentile,
    punctuation_f1,
    score_pair,
    wer_tokens,
)


class MetricsTest(unittest.TestCase):
    def test_exact_match_normalized_zwj(self) -> None:
        gt = "प्र\u200dक्र"
        pred = "प्रक्र"
        self.assertTrue(exact_match(gt, pred, strict=False))
        self.assertFalse(exact_match(gt, pred, strict=True))

    def test_dual_cer(self) -> None:
        gt = "अ\u200cब"
        pred = "अब"
        self.assertGreater(cer_strict(gt, pred), 0.0)
        self.assertEqual(cer_normalized(gt, pred), 0.0)

    def test_wer_tokens(self) -> None:
        self.assertEqual(wer_tokens("एक दोन तीन", "एक दोन तीन"), 0.0)
        self.assertAlmostEqual(wer_tokens("एक दोन तीन", "एक तीन"), 1 / 3)

    def test_marathi_numeral_and_contamination(self) -> None:
        gt = "प्र.क्र.३५७"
        good = "प्र.क्र.३५७"
        bad = "प्र.क्र.357"
        m_good = score_pair(gt, good)
        m_bad = score_pair(gt, bad)
        self.assertEqual(m_good["marathi_numeral_accuracy"], 1.0)
        self.assertFalse(m_good["ascii_digit_contamination"])
        self.assertTrue(m_bad["ascii_digit_contamination"])
        cont = ascii_digit_contamination(gt, bad)
        self.assertEqual(cont["ascii_digit_count_in_pred"], 3)

    def test_reference_id_subset(self) -> None:
        gt = "२०२२/प्र.क्र.३५७/का.१४"
        self.assertTrue(is_reference_like(gt))
        m = score_pair(gt, gt)
        self.assertEqual(m["reference_id_exact"], 1.0)
        m2 = score_pair(gt, "wrong")
        self.assertEqual(m2["reference_id_exact"], 0.0)
        plain = score_pair("साधा मजकूर", "साधा मजकूर")
        self.assertIsNone(plain["reference_id_exact"])

    def test_punctuation_f1(self) -> None:
        gt = "अ/ब.क,ड"
        pred = "अ/ब.क"
        f1 = punctuation_f1(gt, pred)
        self.assertGreater(f1["recall"], 0.5)
        self.assertLess(f1["recall"], 1.0)
        self.assertEqual(f1["precision"], 1.0)

    def test_conjunct_error_rate(self) -> None:
        gt = "क्षत्रिय ज्ञानी श्रद्धा द्वार"
        pred = "कत्रिय जानी श्रद्धा द्वार"  # missing क्ष/ज्ञ forms
        self.assertIn("क्ष", find_conjunct_occurrences(gt))
        rate = conjunct_error_rate(gt, pred)
        self.assertIsNotNone(rate)
        assert rate is not None
        self.assertGreater(rate, 0.0)
        self.assertEqual(conjunct_error_rate("साधा", "साधा"), None)

    def test_length_bands_and_aggregate(self) -> None:
        short_gt = "१२/प्र.क्र.३"  # short
        long_gt = "अ" * 40
        rows = [
            {**score_pair(short_gt, short_gt), "latency_ms": 10.0},
            {**score_pair(long_gt, "ब" * 40), "latency_ms": 30.0},
            {**score_pair(short_gt, "wrong"), "latency_ms": 20.0},
        ]
        # attach issue_type / source for slices
        rows[0]["issue_type"] = "complex_reference_number"
        rows[0]["source_id"] = "src_a"
        rows[1]["issue_type"] = "other"
        rows[1]["source_id"] = "src_b"
        rows[2]["issue_type"] = "complex_reference_number"
        rows[2]["source_id"] = "src_a"

        summary = aggregate(rows)
        self.assertIn("length_band_cer", summary)
        self.assertIn("short_le_25", summary["length_band_cer"])
        self.assertIn("long_gt_25", summary["length_band_cer"])
        self.assertEqual(summary["latency_ms_p50"], 20.0)
        self.assertIn("by_issue_type", summary["slices"])
        self.assertIn("by_contains_marathi_digits", summary["slices"])
        self.assertEqual(summary["slices"]["by_source_id"]["n_sources"], 2)

    def test_percentile(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 50), 2.5)
        self.assertIsNone(percentile([], 95))


if __name__ == "__main__":
    unittest.main()
