"""Gemini vision provider.

Gemini is the default because it points natively in normalised 0-1000 space
(exactly the coordinate system in actions.py, so no rescaling sits between
the model and the click) and is cheap enough that a 40-step run is not a
budgeting decision.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request

from .base import Provider, Step, SYSTEM_PROMPT, extract_action
from ..actions import Action

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# How many past steps to replay. Screenshots are NOT resent — only the
# current frame is, with prior steps compressed to action+observation text.
# Resending frames is what makes these loops expensive, and it buys little:
# the current screen already reflects everything the earlier ones caused.
HISTORY_WINDOW = 12


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, api_key: str | None = None,
                 model: str = "gemini-2.5-flash", *, timeout: int = 90, opener=None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self.model = model.split("/")[-1]
        self.timeout = timeout
        self._opener = opener or urllib.request.urlopen

    def _history_text(self, history: list[Step]) -> str:
        if not history:
            return "No actions taken yet. This is the first step."
        lines = []
        for i, step in enumerate(history[-HISTORY_WINDOW:], start=max(1, len(history) - HISTORY_WINDOW + 1)):
            act = json.dumps(step.action.to_dict(), ensure_ascii=False)
            lines.append(f"{i}. {act} -> {step.observation or 'no observation'}")
        return "\n".join(lines)

    def propose(self, goal: str, screenshot_png: bytes, history: list[Step],
                extra_context: str = "") -> Action:
        prompt = [SYSTEM_PROMPT, f"\nGOAL: {goal}"]
        if extra_context:
            # Task-supplied facts (form values, the user's real details). Fenced
            # and labelled so the model treats it as data to use, not as
            # instructions that can redirect the goal.
            prompt.append(f"\nCONTEXT YOU MAY USE (do not invent anything beyond this):\n{extra_context}")
        prompt.append(f"\nSTEPS SO FAR:\n{self._history_text(history)}")
        prompt.append("\nThe image is the CURRENT screen. Emit the single next action as JSON.")

        body = {
            "contents": [{
                "parts": [
                    {"text": "\n".join(prompt)},
                    {"inline_data": {"mime_type": "image/png",
                                     "data": base64.b64encode(screenshot_png).decode("ascii")}},
                ]
            }],
            "generationConfig": {
                # Low but non-zero: a fully greedy loop that misreads the
                # screen re-derives the identical wrong action forever, and
                # the stall detector then kills an otherwise fine run.
                "temperature": 0.15,
                "maxOutputTokens": 700,
                "responseMimeType": "application/json",
            },
        }

        req = urllib.request.Request(
            API_URL.format(model=self.model),
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
        )
        try:
            with self._opener(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise ValueError(f"Gemini HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ValueError(f"Gemini unreachable: {exc.reason}") from exc

        try:
            text = payload["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            # Safety blocks and quota rejections both land here with no
            # candidate; surface the raw payload rather than a KeyError.
            raise ValueError(f"no usable candidate in Gemini response: {json.dumps(payload)[:300]}")

        return extract_action(text)
