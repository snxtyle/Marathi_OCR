from __future__ import annotations

"""Streamlit human review UI for Marathi OCR candidates."""

import json
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import load_config, resolve_path
from pipeline.io_utils import append_jsonl, read_jsonl, write_jsonl
from pipeline.splits import ISSUE_TYPES, classify_issue_type


def load_queue(path: Path) -> list[dict]:
    return read_jsonl(path)


def save_queue(path: Path, records: list[dict]) -> None:
    write_jsonl(path, records)


def main() -> None:
    st.set_page_config(page_title="Marathi OCR Review", layout="wide")
    st.title("Marathi OCR Human Review")

    cfg = load_config()
    cand_dir = resolve_path(cfg, "candidates")
    reviewed_dir = resolve_path(cfg, "reviewed")
    reviewed_dir.mkdir(parents=True, exist_ok=True)
    queue_path = cand_dir / "candidates_valid.jsonl"
    if not queue_path.is_file():
        queue_path = cand_dir / "candidates.jsonl"
    reviewed_path = reviewed_dir / "reviewed.jsonl"

    records = load_queue(queue_path)
    # Prioritize LLM-flagged samples
    pending = [r for r in records if r.get("review_status") not in {"verified", "rejected"}]
    pending.sort(key=lambda r: 0 if r.get("review_status") == "needs_llm_attention" else 1)
    st.sidebar.write(f"Pending: {len(pending)} / {len(records)}")
    needs = sum(1 for r in pending if r.get("review_status") == "needs_llm_attention")
    if needs:
        st.sidebar.warning(f"LLM attention: {needs}")

    if "idx" not in st.session_state:
        st.session_state.idx = 0
    if not pending:
        st.success("No pending candidates. Export when ready.")
        if st.button("Show reviewed count"):
            st.write(len(read_jsonl(reviewed_path)))
        return

    idx = min(st.session_state.idx, len(pending) - 1)
    rec = pending[idx]
    st.session_state.idx = idx

    col1, col2 = st.columns([1.4, 1])
    img_path = ROOT / rec.get("image_filename", "")
    if not img_path.is_file():
        img_path = Path(rec.get("image_path", ""))

    with col1:
        st.subheader("Image")
        zoom = st.slider("Zoom", 1.0, 4.0, 1.5, 0.1)
        if img_path.is_file():
            img = Image.open(img_path)
            w = int(img.width * zoom)
            h = int(img.height * zoom)
            st.image(img.resize((w, h)), caption=img_path.name)
        else:
            st.error(f"Missing image: {img_path}")

    with col2:
        st.subheader("Labels")
        st.caption(f"ID: `{rec.get('id')}`")
        st.caption(
            f"difficulty={rec.get('difficulty')} source={rec.get('source_id')} "
            f"page={rec.get('source_page')} score={rec.get('complexity_score')} "
            f"type={rec.get('crop_type')} doc={rec.get('document_type')}"
        )
        if rec.get("source_url"):
            st.caption(f"url={rec.get('source_url')}")
        if rec.get("text_near_duplicate_flag"):
            st.warning("Text near-duplicate flagged — keep only if visually/document-diverse.")
        st.caption("Layer: human_review (only verified GT proceeds to final_qa)")
        st.text_area("OCR prediction (pre-label)", value=rec.get("ocr_prediction", ""), height=80, disabled=True)
        default_text = rec.get("expected_text") or rec.get("ocr_prediction") or ""
        edited = st.text_area("Expected text (editable)", value=default_text, height=120)
        issue = st.selectbox(
            "Issue type",
            ISSUE_TYPES,
            index=max(0, ISSUE_TYPES.index(classify_issue_type(edited, rec.get("crop_type", ""))))
            if classify_issue_type(edited, rec.get("crop_type", "")) in ISSUE_TYPES
            else 0,
        )
        st.write("Flags:", {
            "marathi_numerals": rec.get("contains_marathi_numerals"),
            "special_chars": rec.get("contains_special_chars"),
            "reference": rec.get("contains_reference_identifier"),
            "conjuncts": rec.get("contains_conjuncts"),
        })

        llm = (rec.get("validation") or {}).get("llm") or {}
        if llm:
            st.subheader("Kimi image↔text check")
            if rec.get("review_status") == "needs_llm_attention":
                st.error("Flagged by LLM — image and text may not match exactly.")
            if llm.get("exact_match"):
                st.success("LLM reports exact match (still confirm visually).")
            st.json(
                {
                    "mode": llm.get("mode"),
                    "exact_match": llm.get("exact_match"),
                    "visible_differs": llm.get("visible_differs"),
                    "suspicious": llm.get("suspicious"),
                    "diff_spans": llm.get("diff_spans"),
                    "likely_ocr_errors": llm.get("likely_ocr_errors"),
                    "confidence": llm.get("confidence"),
                    "notes": llm.get("notes"),
                    "error": llm.get("error"),
                    "vision_error": llm.get("vision_error"),
                    "skipped": llm.get("skipped"),
                }
            )

        b1, b2, b3, b4 = st.columns(4)
        if b1.button("ACCEPT", type="primary"):
            rec["expected_text"] = edited
            rec["issue_type"] = issue
            rec["review_status"] = "verified"
            append_jsonl(reviewed_path, rec)
            # update queue
            for i, r in enumerate(records):
                if r.get("id") == rec.get("id"):
                    records[i] = rec
            save_queue(queue_path, records)
            st.session_state.idx = min(idx + 1, max(len(pending) - 2, 0))
            st.rerun()
        if b2.button("EDIT+SAVE"):
            rec["expected_text"] = edited
            rec["issue_type"] = issue
            rec["review_status"] = "verified"
            append_jsonl(reviewed_path, rec)
            for i, r in enumerate(records):
                if r.get("id") == rec.get("id"):
                    records[i] = rec
            save_queue(queue_path, records)
            st.success("Saved")
            st.rerun()
        if b3.button("REJECT"):
            rec["review_status"] = "rejected"
            rec["expected_text"] = edited
            append_jsonl(reviewed_dir / "rejected.jsonl", rec)
            for i, r in enumerate(records):
                if r.get("id") == rec.get("id"):
                    records[i] = rec
            save_queue(queue_path, records)
            st.rerun()
        if b4.button("NEXT"):
            st.session_state.idx = min(idx + 1, len(pending) - 1)
            st.rerun()


if __name__ == "__main__":
    main()
