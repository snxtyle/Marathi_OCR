"""Unit-style checks: no REFERENCE_PATTERNS gate; slash titles not auto-hard."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scoring.complexity import classify_difficulty, demote_unconfirmed_hard
from scoring import reference_detector as rd


def test_no_reference_patterns_list() -> None:
    src = Path(rd.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {
        n.targets[0].id
        for n in tree.body
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
    }
    assert "REFERENCE_PATTERNS" not in names, "REFERENCE_PATTERNS must not exist"
    assert "REFERENCE_PATTERNS" not in rd.__dict__


def test_internship_slash_not_auto_hard() -> None:
    samples = [
        "(ज) इंटर्नशिप कालावधीत गैरशिस्तीच्या कारणामुळे एखाद्या विद्यार्थ्यांची इंटर्नशिप रद्द / समाप्त",
        "सर्व सहसचिव/उप सचिव/अवर सचिव/कक्ष अधिकारी, वित्त विभाग, मंत्रालय, मुंबई.",
        "सर्व मंत्री / राज्यमंत्री यांचे खाजगी सचिव / स्वीय सहायक",
        "महाविद्यालयातील / विद्यापीठातील, एकाच विद्यार्थ्याची निवड केली जाईल.",
    ]
    for text in samples:
        c = classify_difficulty(text, allow_ascii_digits=False, min_hard_score=3.0)
        assert c["difficulty"] != "hard", f"slash prose wrongly hard: {text[:40]}… → {c}"
        assert c.get("hard_signal_source") != "heuristic_strong" or c["difficulty"] != "hard"


def test_llm_can_mark_hard() -> None:
    text = "सर्व मंत्री / राज्यमंत्री यांचे खाजगी सचिव / स्वीय सहायक"
    c = classify_difficulty(text, llm_complexity="hard")
    assert c["difficulty"] == "hard"
    assert c["hard_signal_source"] == "llm"


def test_llm_cannot_force_ordinary_to_hard() -> None:
    text = "महाराष्ट्र राज्यातील नागरिकांसाठी विविध कल्याणकारी योजना राबविल्या जात आहेत."
    c = classify_difficulty(
        text,
        llm_complexity="hard",
        normal_min_words=6,
        normal_max_words=40,
    )
    assert c["difficulty"] == "normal", c
    assert c.get("hard_signal_source") == "llm_demoted_ordinary"


def test_paren_list_marker_can_be_ordinary() -> None:
    text = "(क) नागरिकांसाठी विविध कल्याणकारी योजना राबविल्या जात आहेत आज."
    c = classify_difficulty(text, allow_ascii_digits=False, min_hard_score=3.0)
    assert c["difficulty"] == "normal", c


def test_soft_fail_demotes_weak_hard() -> None:
    rec = {
        "difficulty": "hard",
        "hard_signal_source": "none",
        "expected_text": "अटी व शर्ती नमूद कराव्यात. या करारामध्ये कंपनी / संस्थेतर्फे काम करणाऱ्या अधिकारी",
        "features": {},
    }
    demote_unconfirmed_hard(rec)
    assert rec["difficulty"] == "reject"


def test_strong_digit_line_can_be_hard() -> None:
    # Generic numeral + separator structure (not a GR keyword checklist)
    text = "क्रमांक १४२६/८८/२०२४ नुसार पुढील कार्यवाही करावी."
    c = classify_difficulty(text, allow_ascii_digits=False, min_hard_score=2.0)
    assert c["difficulty"] == "hard", c
    assert c["hard_signal_source"] == "heuristic_strong"


def test_prefix_twins_dropped() -> None:
    from validation.duplicates import drop_near_duplicate_texts

    records = [
        {
            "id": "a",
            "expected_text": "(ज) इंटर्नशिप कालावधीत गैरशिस्तीच्या कारणामुळे एखाद्या विद्यार्थ्यांची इंटर्नशिप रद्द / समाप्त",
            "complexity_score": 2.0,
        },
        {
            "id": "b",
            "expected_text": "(ज) इंटर्नशिप कालावधीत गैरशिस्तीच्या कारणामुळे एखाद्या विद्यार्थ्यांची इंटर्नशिप रद्द /",
            "complexity_score": 1.5,
        },
    ]
    kept, stats = drop_near_duplicate_texts(records, threshold=0.88)
    assert stats["dropped"] == 1
    assert len(kept) == 1
    assert kept[0]["id"] == "a"


def test_require_hard_ref_unused_in_active_callers() -> None:
    for rel in (
        "scripts/score.py",
        "scripts/select_validation.py",
        "scripts/extract.py",
        "extraction/regions.py",
        "scoring/complexity.py",
    ):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "require_hard_ref_or_digits" not in text, f"still present in {rel}"
        assert "REFERENCE_PATTERNS" not in text, f"still present in {rel}"


def test_scale_lane_targets_ratio() -> None:
    from pipeline.quotas import scale_lane_targets

    profile = {"hard_count": 80, "normal_count": 20}
    h, n = scale_lane_targets(target_count=100, profile=profile)
    assert (h, n) == (80, 20), (h, n)
    h, n = scale_lane_targets(target_count=1000, profile=profile)
    assert (h, n) == (800, 200), (h, n)
    assert h + n == 1000
    h, n = scale_lane_targets(target_count=5, profile=profile)
    assert h + n == 5 and h >= 1 and n >= 1
    h, n = scale_lane_targets(hard_count=10, normal_count=3, profile=profile)
    assert (h, n) == (10, 3)


def test_run_pipeline_parses_target_count() -> None:
    import scripts.run_pipeline as rp

    parser_src = Path(rp.__file__).read_text(encoding="utf-8")
    assert "--target-count" in parser_src
    assert "--samples" in parser_src
    # Dry argparse
    old = sys.argv
    try:
        sys.argv = ["run_pipeline", "--help"]
        try:
            rp.main()
        except SystemExit as exc:
            assert exc.code in {0, None}
    finally:
        sys.argv = old


if __name__ == "__main__":
    tests = [
        test_no_reference_patterns_list,
        test_internship_slash_not_auto_hard,
        test_llm_can_mark_hard,
        test_llm_cannot_force_ordinary_to_hard,
        test_paren_list_marker_can_be_ordinary,
        test_soft_fail_demotes_weak_hard,
        test_strong_digit_line_can_be_hard,
        test_prefix_twins_dropped,
        test_require_hard_ref_unused_in_active_callers,
        test_scale_lane_targets_ratio,
        test_run_pipeline_parses_target_count,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"OK  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERR  {fn.__name__}: {exc}")
    if failed:
        sys.exit(1)
    print(f"All {len(tests)} checks passed.")
