"""Hosted LLM client (OpenAI-compatible chat completions).

Two production uses:
  1) Discovery — Kimi web-aware PDF URL search + ranking
  2) Vision validation — image + candidate text exact-match precision check

Never auto-writes verified ground truth.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _image_to_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


class LLMClient:
    """Kimi / Juspay Grid client for discovery + vision validation."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        api_url: str = "",
        api_key: str = "",
        model: str = "",
    ) -> None:
        env_on = os.environ.get("LLM_ENABLED", "").lower() in {"1", "true", "yes"}
        self.enabled = bool(enabled) or env_on
        self.api_url = (api_url or os.environ.get("LLM_API_URL", "")).strip()
        self.api_key = (api_key or os.environ.get("LLM_API_KEY", "")).strip()
        self.model = (model or os.environ.get("LLM_MODEL", "")).strip()
        if self.api_url and not self.api_url.rstrip("/").endswith("chat/completions"):
            base = self.api_url.rstrip("/")
            if base.endswith("/v1"):
                self.api_url = f"{base}/chat/completions"
            else:
                self.api_url = f"{base}/v1/chat/completions"

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "LLMClient":
        return cls(
            enabled=bool(cfg.get("llm_enabled", False)),
            api_url=str(cfg.get("llm_api_url") or ""),
            api_key=str(cfg.get("llm_api_key") or ""),
            model=str(cfg.get("llm_model") or ""),
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: int = 90,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "error": "llm_disabled", "content": ""}
        if not self.api_url or not self.api_key:
            return {"ok": False, "error": "LLM_API_URL/LLM_API_KEY missing", "content": ""}
        if not self.model:
            return {"ok": False, "error": "LLM_MODEL missing", "content": ""}

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

        try:
            resp = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
            if resp.status_code >= 400:
                return {
                    "ok": False,
                    "error": f"http_{resp.status_code}: {resp.text[:400]}",
                    "content": "",
                }
            data = resp.json()
            msg = (data.get("choices") or [{}])[0].get("message") or {}
            content = msg.get("content") or msg.get("reasoning_content") or ""
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        parts.append(str(item.get("text") or item.get("content") or ""))
                content = "".join(parts)
            return {
                "ok": True,
                "content": content or "",
                "raw": data,
                "tool_calls": msg.get("tool_calls"),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM chat failed: %s", exc)
            return {"ok": False, "error": str(exc), "content": ""}

    # ------------------------------------------------------------------
    # LLM use #1 — web discovery of Marathi PDFs (Kimi)
    # ------------------------------------------------------------------

    def propose_pdf_urls(self, query: str, *, max_urls: int = 20) -> dict[str, Any]:
        """Ask Kimi (web-search capable) for public Marathi government PDF URLs."""
        system = (
            "You are a web-research assistant discovering publicly available Marathi "
            "government PDF documents for an OCR dataset. "
            "Focus: Maharashtra शासन निर्णय (GR), gazettes, circulars, official notices. "
            "You MUST use live web search if your gateway supports it. "
            "Prefer direct .pdf links on official .gov.in domains "
            "(gr.maharashtra.gov.in, maharashtra.gov.in, lj.maharashtra.gov.in, etc.). "
            "Reject blogs, blogs mirrors, paywalled, and non-PDF pages. "
            "Return diverse, downloadable PDF URLs useful for hard Devanagari OCR: "
            "documents likely to contain Marathi numerals, dense punctuation, "
            "file/reference-style identifiers, mixed special characters "
            "(judge usefulness in natural language — no fixed keyword checklist). "
            "Respond with JSON only: "
            '{"queries":["..."],"pdf_urls":["https://...pdf"],"notes":"..."}'
        )
        user = json.dumps(
            {
                "user_query": query,
                "max_urls": max_urls,
                "requirements": [
                    "Marathi language body text preferred",
                    "public PDF only with direct .pdf URL when possible",
                    "government / official Maharashtra sources preferred",
                    "hard OCR value: Marathi numerals ०-९, dense punctuation, "
                    "identifier-like strings (illustrative KIND only, not required phrases)",
                ],
            },
            ensure_ascii=False,
        )
        # Prefer built-in web search tool when the gateway exposes it.
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the public web for Marathi government PDF URLs",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "num_results": {"type": "integer"},
                        },
                        "required": ["query"],
                    },
                },
            }
        ]
        result = self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=2048,
            timeout=180,
            tools=tools,
            tool_choice="auto",
        )
        if not result.get("ok"):
            # Retry without tools (some gateways reject tools)
            result = self.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.2,
                max_tokens=2048,
                timeout=180,
            )
        if not result.get("ok"):
            return {
                "ok": False,
                "error": result.get("error"),
                "queries": [query],
                "pdf_urls": [],
            }
        parsed = _extract_json(result.get("content", "")) or {}
        urls = [
            u
            for u in (parsed.get("pdf_urls") or [])
            if isinstance(u, str) and u.startswith("http")
        ]
        queries = [q for q in (parsed.get("queries") or [query]) if isinstance(q, str)]
        return {
            "ok": True,
            "queries": queries or [query],
            "pdf_urls": urls[:max_urls],
            "notes": parsed.get("notes", ""),
            "raw_content": result.get("content", ""),
        }

    def rank_pdf_candidates(
        self, candidates: list[dict[str, str]], *, max_keep: int = 20
    ) -> list[dict[str, Any]]:
        """Rank discovered PDF links for hard Marathi OCR usefulness."""
        if not candidates:
            return []
        if not self.enabled:
            return [
                {"url": c.get("url", ""), "title": c.get("title", ""), "score": 0.5}
                for c in candidates[:max_keep]
            ]

        system = (
            "Rank PDF candidates for building a hard Marathi OCR dataset. "
            "Prefer official .gov.in PDFs likely to yield OCR-hard crops: "
            "Marathi numerals, dense punctuation, mixed identifiers/specials "
            "(natural-language judgment — not a fixed keyword list). "
            "Respond JSON only: "
            '{"ranked":[{"url":"...","score":0.0-1.0,"reason":"..."}]}'
        )
        result = self.chat(
            [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"candidates": candidates[:40], "max_keep": max_keep},
                        ensure_ascii=False,
                    ),
                },
            ],
            max_tokens=1500,
        )
        if not result.get("ok"):
            return [
                {"url": c.get("url", ""), "title": c.get("title", ""), "score": 0.5}
                for c in candidates[:max_keep]
            ]
        parsed = _extract_json(result.get("content", "")) or {}
        ranked = parsed.get("ranked") or []
        out: list[dict[str, Any]] = []
        for item in ranked:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if isinstance(url, str) and url.startswith("http"):
                out.append(
                    {
                        "url": url,
                        "title": str(item.get("title") or ""),
                        "score": float(item.get("score") or 0.0),
                        "reason": str(item.get("reason") or ""),
                    }
                )
        if not out:
            return [
                {"url": c.get("url", ""), "title": c.get("title", ""), "score": 0.5}
                for c in candidates[:max_keep]
            ]
        out.sort(key=lambda x: x.get("score", 0), reverse=True)
        return out[:max_keep]

    # ------------------------------------------------------------------
    # LLM use #2 — vision: image ↔ text exact-match precision
    # ------------------------------------------------------------------

    def vision_validate_exact(
        self,
        image_path: Path | str,
        candidate_text: str,
        *,
        ocr_prediction: str = "",
        context: dict[str, Any] | None = None,
        intended_lane: str = "",
    ) -> dict[str, Any]:
        """Vision end-gate: image↔text exactness AND OCR-hardness judgment.

        Does NOT invent/rewrite ground truth. Flags mismatches for review/reject.
        Complexity is judged from image+text in natural language — not keyword lists.
        Falls back to text-only semantic check if vision is rejected by the API.
        """
        path = Path(image_path)
        soft_complexity = {
            "is_complex_for_ocr": None,
            "suggested_lane": None,
            "complexity_notes": "",
        }
        if not self.enabled:
            return {
                "enabled": False,
                "ok": True,
                "skipped": True,
                "exact_match": False,
                "suspicious": False,
                "confidence": 0.0,
                "notes": "llm_disabled",
                "mode": "disabled",
                **soft_complexity,
            }
        if not path.is_file():
            return {
                "enabled": True,
                "ok": False,
                "skipped": True,
                "exact_match": False,
                "suspicious": True,
                "confidence": 0.0,
                "notes": "missing_image",
                "mode": "vision",
                "error": f"missing_image:{path}",
                **soft_complexity,
            }

        system = (
            "You are a precise Marathi OCR dataset validator. You SEE the crop image "
            "and a candidate text string. Make TWO judgments:\n\n"
            "1) EXACT MATCH — Is candidate_text an EXACT transcription of the visible "
            "text (character-level precision)?\n"
            "- Exact means every Devanagari character, Marathi numeral (०-९), "
            "punctuation (/ . , - () : ;), and spacing matches the image.\n"
            "- Flag ASCII digits if the image shows Marathi digits.\n"
            "- Flag missing/extra matras, conjuncts, or tokens.\n"
            "- Do NOT invent or rewrite ground truth. Only judge the given candidate.\n\n"
            "2) OCR COMPLEXITY — Is this sample genuinely HARD for OCR and suitable "
            "for an ~80% hard validation pack?\n"
            "- Hard means the visible text would challenge OCR: Marathi digits, dense "
            "punctuation, slashes/hyphens in identifiers, mixed special characters, "
            "currency/percent, dense conjuncts/matras, or similar structural difficulty.\n"
            "- Ordinary flowing Marathi prose without those challenges → suggested_lane "
            '"normal" (is_complex_for_ocr=false).\n'
            "- Too trivial / garbage / unreadable → suggested_lane \"reject\".\n"
            "- A lone slash in an office/job title list is NOT automatically hard.\n"
            "- Judge from structural OCR difficulty you see (numerals, dense "
            "punctuation, identifier-like digit+separator clusters, currency, "
            "dense conjuncts); do NOT require any fixed keyword/template list.\n\n"
            "Respond with JSON only:\n"
            "{"
            '"exact_match": true/false, '
            '"visible_differs": true/false, '
            '"diff_spans": ["short quotes of mismatches"], '
            '"likely_ocr_errors": ["..."], '
            '"is_complex_for_ocr": true/false, '
            '"suggested_lane": "hard"|"normal"|"reject", '
            '"complexity_notes": "short reason", '
            '"suspicious": true/false, '
            '"confidence": 0.0-1.0, '
            '"notes": "short reason"'
            "}"
        )
        try:
            data_url = _image_to_data_url(path)
        except Exception as exc:  # noqa: BLE001
            return {
                "enabled": True,
                "ok": False,
                "skipped": True,
                "exact_match": False,
                "suspicious": True,
                "confidence": 0.0,
                "notes": "image_encode_failed",
                "mode": "vision",
                "error": str(exc),
                **soft_complexity,
            }

        user_payload = {
            "candidate_text": candidate_text,
            "ocr_prediction": ocr_prediction,
            "context": context or {},
            "intended_lane": intended_lane or None,
            "instruction": (
                "Look at the image carefully. (1) Is candidate_text an exact match "
                "to the text visible in the image? (2) Is this sample complex/hard "
                "enough for the hard OCR validation pack?"
            ),
        }
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
        result = self.chat(messages, temperature=0.0, max_tokens=1500, timeout=120)

        # Soft fallback if gateway rejects vision payloads
        if not result.get("ok") and result.get("error"):
            err = str(result.get("error"))
            if any(x in err.lower() for x in ("image", "vision", "unsupported", "400", "415")):
                logger.warning("Vision rejected (%s); falling back to text semantic check", err[:120])
                fallback = self.semantic_validate(
                    candidate_text,
                    ocr_prediction=ocr_prediction,
                    context=context,
                )
                fallback["mode"] = "text_fallback"
                fallback["vision_error"] = err
                return fallback

        if not result.get("ok"):
            return {
                "enabled": True,
                "ok": True,
                "skipped": True,
                "exact_match": False,
                "suspicious": False,
                "confidence": 0.0,
                "notes": "llm_call_failed_soft_pass",
                "mode": "vision",
                "error": result.get("error"),
                **soft_complexity,
            }

        parsed = _extract_json(result.get("content", "")) or {}
        exact = bool(parsed.get("exact_match"))
        suspicious = bool(
            parsed.get("suspicious")
            or parsed.get("visible_differs")
            or (not exact)
            or parsed.get("likely_ocr_errors")
            or parsed.get("diff_spans")
        )
        # If model says exact_match true, don't force suspicious
        if exact and not parsed.get("suspicious") and not parsed.get("visible_differs"):
            suspicious = False

        lane_raw = str(parsed.get("suggested_lane") or "").strip().lower()
        if lane_raw not in {"hard", "normal", "reject"}:
            if parsed.get("is_complex_for_ocr") is True:
                lane_raw = "hard"
            elif parsed.get("is_complex_for_ocr") is False:
                lane_raw = "normal"
            else:
                lane_raw = ""
        is_complex = parsed.get("is_complex_for_ocr")
        if is_complex is None and lane_raw:
            is_complex = lane_raw == "hard"
        elif is_complex is not None:
            is_complex = bool(is_complex)

        return {
            "enabled": True,
            "ok": True,
            "skipped": False,
            "exact_match": exact,
            "visible_differs": bool(parsed.get("visible_differs", not exact)),
            "diff_spans": list(parsed.get("diff_spans") or []),
            "likely_ocr_errors": list(parsed.get("likely_ocr_errors") or []),
            "is_complex_for_ocr": is_complex,
            "suggested_lane": lane_raw or None,
            "ocr_complexity": lane_raw or None,
            "complexity_notes": str(parsed.get("complexity_notes") or ""),
            "suspicious": suspicious,
            "confidence": float(parsed.get("confidence") or 0.0),
            "notes": str(parsed.get("notes") or ""),
            "mode": "vision",
            "raw_content": result.get("content", ""),
        }

    def semantic_validate(
        self,
        text: str,
        *,
        ocr_prediction: str = "",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Text-only linguistic fallback (no image). Used if vision unavailable."""
        if not self.enabled:
            return {
                "enabled": False,
                "ok": True,
                "skipped": True,
                "is_plausible_marathi": True,
                "suspicious": False,
                "nonsense_tokens": [],
                "punctuation_issues": [],
                "likely_ocr_garble": False,
                "is_complex_for_ocr": None,
                "suggested_lane": None,
                "complexity_notes": "",
                "confidence": 0.0,
                "notes": "llm_disabled",
                "mode": "text",
            }

        system = (
            "You validate Marathi OCR candidate text for linguistic plausibility and "
            "OCR-hardness (text-only fallback; you cannot see the image). "
            "Do NOT invent or rewrite ground truth. "
            "Check: real Marathi words vs character soup; odd punctuation; OCR garble. "
            "Also suggest hard vs normal: hard = digits, dense punctuation, "
            "identifier-like digit+separator strings, specials, dense conjuncts; "
            "normal = ordinary flowing prose. A lone title slash is not automatically hard. "
            "No fixed keyword checklist. Respond with JSON only:\n"
            "{"
            '"is_plausible_marathi": true/false, '
            '"nonsense_tokens": ["..."], '
            '"punctuation_issues": ["..."], '
            '"likely_ocr_garble": true/false, '
            '"is_complex_for_ocr": true/false, '
            '"suggested_lane": "hard"|"normal"|"reject", '
            '"complexity_notes": "short reason", '
            '"suspicious": true/false, '
            '"confidence": 0.0-1.0, '
            '"notes": "short reason"'
            "}"
        )
        user = json.dumps(
            {
                "candidate_text": text,
                "ocr_prediction": ocr_prediction,
                "context": context or {},
            },
            ensure_ascii=False,
        )
        result = self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=2048,
        )
        if not result.get("ok"):
            return {
                "enabled": True,
                "ok": True,
                "skipped": True,
                "error": result.get("error"),
                "is_plausible_marathi": True,
                "suspicious": False,
                "nonsense_tokens": [],
                "punctuation_issues": [],
                "likely_ocr_garble": False,
                "is_complex_for_ocr": None,
                "suggested_lane": None,
                "complexity_notes": "",
                "confidence": 0.0,
                "notes": "llm_call_failed_soft_pass",
                "mode": "text",
            }

        parsed = _extract_json(result.get("content", "")) or {}
        suspicious = bool(
            parsed.get("suspicious")
            or parsed.get("likely_ocr_garble")
            or (parsed.get("is_plausible_marathi") is False)
            or parsed.get("nonsense_tokens")
        )
        lane_raw = str(parsed.get("suggested_lane") or "").strip().lower()
        if lane_raw not in {"hard", "normal", "reject"}:
            lane_raw = ""
        is_complex = parsed.get("is_complex_for_ocr")
        if is_complex is None and lane_raw:
            is_complex = lane_raw == "hard"
        elif is_complex is not None:
            is_complex = bool(is_complex)
        return {
            "enabled": True,
            "ok": True,
            "skipped": False,
            "is_plausible_marathi": bool(parsed.get("is_plausible_marathi", True)),
            "nonsense_tokens": list(parsed.get("nonsense_tokens") or []),
            "punctuation_issues": list(parsed.get("punctuation_issues") or []),
            "likely_ocr_garble": bool(parsed.get("likely_ocr_garble", False)),
            "is_complex_for_ocr": is_complex,
            "suggested_lane": lane_raw or None,
            "ocr_complexity": lane_raw or None,
            "complexity_notes": str(parsed.get("complexity_notes") or ""),
            "suspicious": suspicious,
            "confidence": float(parsed.get("confidence") or 0.0),
            "notes": str(parsed.get("notes") or ""),
            "mode": "text",
            "raw_content": result.get("content", ""),
        }
