"""Fully automatic Marathi OCR validation factory — one command, no UI.

discover → render → extract → score → dedup → select → ocr → validate
  → auto_accept → (loop while short of target) → export validation package

Use ``--target-count N`` to keep mining until N high-quality accepted samples
(profile 80/20 hard/normal ratio scales with N). Failures are discarded; never
pads weak samples. Exhaustion / safety-cap exits non-zero unless ``--allow-shortfall``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import load_config, resolve_path
from pipeline.io_utils import read_jsonl
from pipeline.profiles import get_active_profile
from pipeline.quotas import extract_headroom, scale_lane_targets


def _run(module: str, extra: list[str] | None = None, *, check: bool = True) -> int:
    cmd = [sys.executable, "-m", module, *(extra or [])]
    print(f"\n==> {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT, check=False)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc.returncode


def _count_verified_lanes(reviewed_path: Path) -> tuple[int, int]:
    if not reviewed_path.is_file():
        return 0, 0
    records = read_jsonl(reviewed_path)
    hard = sum(
        1
        for r in records
        if r.get("review_status") == "verified" and r.get("difficulty") == "hard"
    )
    normal = sum(
        1
        for r in records
        if r.get("review_status") == "verified" and r.get("difficulty") == "normal"
    )
    return hard, normal


def _manifest_source_count() -> int:
    from pipeline.io_utils import read_json

    cfg = load_config()
    manifest = resolve_path(cfg, "sources") / "manifest.json"
    if not manifest.is_file():
        return 0
    try:
        data = read_json(manifest)
    except Exception:  # noqa: BLE001
        return 0
    return len(data.get("sources") or [])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fully automatic validation factory (no Streamlit / no manual review)"
    )
    parser.add_argument(
        "--target-count",
        "--samples",
        dest="target_count",
        type=int,
        default=None,
        help="Mine until N high-quality accepted samples (scales profile 80/20). "
        "Alias: --samples. Refuses shortfall unless --allow-shortfall.",
    )
    parser.add_argument("--hard-count", type=int, default=None, help="Override hard lane target")
    parser.add_argument("--normal-count", type=int, default=None, help="Override normal lane target")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Docs to discover per mining round (default scales with target)",
    )
    parser.add_argument(
        "--max-docs-cap",
        type=int,
        default=None,
        help="Hard safety cap on cumulative docs discovered across rounds",
    )
    parser.add_argument(
        "--docs-growth",
        type=int,
        default=None,
        help="Extra docs requested each mining round after a shortfall",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Safety cap on discover→accept loops (default scales with target)",
    )
    parser.add_argument("--skip-discover", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument(
        "--allow-shortfall",
        action="store_true",
        help="Export draft if quota incomplete after mining (never pads quality). "
        "Default OFF when --target-count / explicit lane counts are set.",
    )
    parser.add_argument(
        "--strict-quota",
        action="store_true",
        help="Force refuse-shortfall even without --target-count (draft-mode override)",
    )
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None, help="Extract top-k (default scales with target)")
    parser.add_argument("--max-per-source", type=int, default=None)
    parser.add_argument(
        "--strict-vision",
        action="store_true",
        help="Drop vision-flagged mismatches during auto-accept",
    )
    parser.add_argument(
        "--require-pdf-text",
        action="store_true",
        help="Only keep crops with PDF text layer labels (no OCR-only GT)",
    )
    parser.add_argument(
        "--with-review",
        action="store_true",
        help="Optional: launch Streamlit human review instead of auto-accept",
    )
    parser.add_argument("--skip-llm-validate", action="store_true", help="Skip Kimi vision in validate")
    args = parser.parse_args()

    cfg = load_config()
    profile = get_active_profile(cfg)
    hard_target, normal_target = scale_lane_targets(
        target_count=args.target_count,
        hard_count=args.hard_count,
        normal_count=args.normal_count,
        profile=profile,
        cfg=cfg,
    )
    total_target = hard_target + normal_target
    explicit_quota = (
        args.target_count is not None
        or args.hard_count is not None
        or args.normal_count is not None
        or args.strict_quota
    )
    allow_shortfall = bool(args.allow_shortfall) and not args.strict_quota
    if explicit_quota and not args.allow_shortfall:
        allow_shortfall = False
    elif not explicit_quota and not args.strict_quota:
        # Legacy draft auto mode: export whatever quality inventory exists
        allow_shortfall = True

    # Mining budgets
    max_rounds = args.max_rounds or int(cfg.get("pipeline_max_rounds") or max(3, (total_target // 50) + 2))
    max_rounds = max(1, max_rounds)
    docs_per_round = args.max_docs or int(
        cfg.get("pipeline_docs_per_round")
        or max(8, min(40, (total_target // 10) + 8))
    )
    docs_growth = args.docs_growth if args.docs_growth is not None else int(cfg.get("pipeline_docs_growth") or max(8, docs_per_round // 2))
    max_docs_cap = args.max_docs_cap or int(cfg.get("pipeline_max_docs_cap") or max(80, total_target // 2 + 40))
    top_k = args.top_k or extract_headroom(hard_target, normal_target, min_top_k=200)
    max_per_source = args.max_per_source or int(profile.get("max_samples_per_source") or 8)
    # Slightly raise per-source cap for large packs without collapsing diversity
    if total_target >= 200 and args.max_per_source is None:
        max_per_source = max(max_per_source, min(12, 4 + total_target // 100))

    lane_args = ["--hard-count", str(hard_target), "--normal-count", str(normal_target)]
    word_args: list[str] = []
    if args.max_words is not None:
        word_args = ["--max-words", str(args.max_words)]

    reviewed_path = resolve_path(cfg, "reviewed") / "reviewed.jsonl"
    export_dir = cfg.get("validation_export_dir") or "data/export/marathi_ocr_validation_100"

    print(
        f"Targets: hard={hard_target} normal={normal_target} total={total_target} "
        f"(allow_shortfall={allow_shortfall}, max_rounds={max_rounds}, "
        f"docs/round≈{docs_per_round}, docs_cap={max_docs_cap})"
    )

    cumulative_docs_requested = 0
    stagnant_rounds = 0
    last_hard, last_normal = -1, -1

    for round_i in range(1, max_rounds + 1):
        print(f"\n======== Mining round {round_i}/{max_rounds} ========")
        sources_before = _manifest_source_count()

        if not args.skip_discover:
            remaining_cap = max(0, max_docs_cap - sources_before)
            if remaining_cap <= 0 and round_i > 1:
                print(
                    f"Hit max-docs safety cap ({max_docs_cap} sources). "
                    "Stopping discovery growth."
                )
            else:
                ask = min(docs_per_round, remaining_cap if remaining_cap > 0 else docs_per_round)
                if ask > 0:
                    _run("scripts.discover", ["--max-docs", str(ask)])
                    cumulative_docs_requested += ask
                else:
                    print("No discovery budget left this round.")
        elif round_i == 1:
            print("Skipping discover (--skip-discover); using existing sources.")
        else:
            print("Skipping discover on later rounds (--skip-discover); cannot mine more.")
            break

        sources_after = _manifest_source_count()
        new_sources = sources_after - sources_before

        _run("scripts.render")
        _run(
            "scripts.extract",
            [
                "--top-k",
                str(top_k),
                "--max-per-source",
                str(max_per_source),
                *lane_args,
                *word_args,
            ],
        )
        _run("scripts.score", word_args)
        _run("scripts.dedup_candidates")
        select_rc = _run("scripts.select_validation", lane_args, check=False)
        if select_rc not in {0, 2}:
            raise subprocess.CalledProcessError(select_rc, ["scripts.select_validation"])
        if select_rc == 2:
            print("Selection shortfall vs lane targets — continuing with quality-gated inventory.")

        _run("scripts.ocr")
        validate_args = list(word_args)
        if args.skip_llm_validate:
            validate_args.append("--skip-llm")
        _run("scripts.validate", validate_args)

        if args.with_review:
            print("\n==> Human review UI (--with-review). Close when done.")
            _run("scripts.review")
        else:
            auto_args: list[str] = []
            if args.strict_vision:
                auto_args.append("--strict-vision")
            if args.require_pdf_text:
                auto_args.append("--require-pdf-text")
            accept_rc = _run("scripts.auto_accept", auto_args, check=False)
            if accept_rc not in {0, 2}:
                raise subprocess.CalledProcessError(accept_rc, ["scripts.auto_accept"])

        hard_n, normal_n = _count_verified_lanes(reviewed_path)
        print(
            f"Round {round_i} inventory: hard={hard_n}/{hard_target} "
            f"normal={normal_n}/{normal_target} (new_sources={new_sources}, "
            f"sources_total={sources_after})"
        )

        if hard_n >= hard_target and normal_n >= normal_target:
            print("Lane targets met in verified inventory.")
            break

        if hard_n == last_hard and normal_n == last_normal and new_sources == 0:
            stagnant_rounds += 1
        else:
            stagnant_rounds = 0
        last_hard, last_normal = hard_n, normal_n

        if stagnant_rounds >= 2:
            print(
                "Sources exhausted or no inventory growth for 2 rounds. "
                "Stopping mining loop."
            )
            break

        if sources_after >= max_docs_cap and (hard_n < hard_target or normal_n < normal_target):
            print(f"Reached docs cap ({max_docs_cap}) with shortfall remaining.")
            break

        # Grow discover budget for next round
        docs_per_round = min(docs_per_round + docs_growth, max(16, max_docs_cap - sources_after))
        # Grow extract headroom slightly when still short
        top_k = min(top_k + max(50, total_target), total_target * 8)
    else:
        print(f"Reached max mining rounds ({max_rounds}) without filling quota.")

    hard_n, normal_n = _count_verified_lanes(reviewed_path)
    quota_met = hard_n >= hard_target and normal_n >= normal_target
    if not quota_met:
        msg = (
            f"SHORTFALL after mining: hard={hard_n}/{hard_target} "
            f"normal={normal_n}/{normal_target}. Refusing to pad weak samples."
        )
        print(msg)
        if not allow_shortfall:
            print("Pass --allow-shortfall for a draft export, or raise --max-docs-cap / --max-rounds.")
            if not args.skip_export:
                # Still attempt export to write validation_report.json with blocked status
                export_args = ["--validation-package", *lane_args]
                _run("scripts.export", export_args, check=False)
            sys.exit(2)

    if not args.skip_export:
        export_args = ["--validation-package", *lane_args]
        if allow_shortfall:
            export_args.append("--allow-shortfall")
        try:
            _run("scripts.export", export_args)
        except subprocess.CalledProcessError as exc:
            if exc.returncode == 2:
                print(
                    "Export blocked (final QA / quota gates). "
                    "Mine more sources or pass --allow-shortfall for draft."
                )
            raise

    print("\nAutomatic pipeline complete.")
    print(f"Package: {export_dir}/")
    print(f"Final verified lanes: hard={hard_n}/{hard_target} normal={normal_n}/{normal_target}")


if __name__ == "__main__":
    main()
