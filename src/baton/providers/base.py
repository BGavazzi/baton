"""Provider interface: given a goal, a screenshot and what happened so far,
propose the next action.

Kept vendor-neutral on purpose. The interesting engineering in a computer-use
agent is the loop, the guards and the trace — not which vendor's vision model
is behind it — and being able to swap Gemini for a local model changes the
cost profile of a long run by orders of magnitude.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..actions import Action, parse_action


@dataclass
class Step:
    """One completed loop iteration, replayed back to the model as history."""
    action: Action
    observation: str = ""


SYSTEM_PROMPT = """You drive a computer by looking at screenshots and emitting one action at a time.

The screen is addressed in a normalised coordinate system: x and y both run 0-1000,
where (0,0) is the top-left corner and (1000,1000) is the bottom-right — regardless of
the screen's real pixel resolution. Point at the CENTRE of what you want to click.

Respond with ONE JSON object and nothing else. No markdown, no backticks, no prose.

Allowed actions:
  {"kind":"click","x":<0-1000>,"y":<0-1000>,"reason":"..."}
  {"kind":"double_click","x":...,"y":...,"reason":"..."}
  {"kind":"right_click","x":...,"y":...,"reason":"..."}
  {"kind":"type","text":"literal text to type","reason":"..."}
  {"kind":"key","key":"Return","reason":"..."}            // or "ctrl+a", "alt+Tab", "Escape"
  {"kind":"scroll","x":...,"y":...,"amount":3,"reason":"..."}  // negative scrolls up
  {"kind":"wait","seconds":2,"reason":"waiting for the page to load"}
  {"kind":"screenshot","reason":"re-checking the screen"}
  {"kind":"done","result":"what was accomplished","reason":"..."}
  {"kind":"fail","result":"why this cannot be completed","reason":"..."}
  {"kind":"ask","result":"the question for the human","reason":"..."}

Rules that matter:
- Look at the CURRENT screenshot before deciding. Do not assume an earlier action worked;
  if the screen does not show what you expected, deal with what is actually there.
- Prefer "key" over clicking when a keyboard route exists (Tab between fields, Return to
  submit) — it is far more reliable than pixel-hunting.
- "type" goes to whatever currently has focus. Click the field first.
- If the goal is already satisfied, emit "done" immediately. Do not add flourishes.
- If you are stuck in a loop, or the screen asks for something you were not given
  (a password, a 2FA code, a CAPTCHA), emit "ask" rather than guessing.
- Never invent credentials, personal details, or answers to form questions. If a required
  field has no supplied value, "ask".
"""


_JSON_RE = re.compile(r"\{.*\}", re.S)


def extract_action(raw: str) -> Action:
    """Pull one action out of a model response.

    Models wrap JSON in prose or code fences no matter how firmly the prompt
    says otherwise, so salvage the object rather than failing the step — but
    salvage it through the same parse_action validation as everything else,
    so a well-formed-but-illegal action is still rejected.
    """
    if not raw or not raw.strip():
        raise ValueError("empty model response")
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_RE.search(text)
        if not match:
            raise ValueError(f"no JSON object in model response: {raw[:200]!r}")
        data = json.loads(match.group(0))
    if isinstance(data, list):
        # Some models answer with a one-element plan even when told not to.
        if not data:
            raise ValueError("model returned an empty action list")
        data = data[0]
    return parse_action(data)


class Provider(ABC):
    name: str = "provider"

    @abstractmethod
    def propose(self, goal: str, screenshot_png: bytes, history: list[Step],
                extra_context: str = "") -> Action:
        """Next action for this goal given the current screen."""
