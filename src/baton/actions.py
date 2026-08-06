"""The action vocabulary a model is allowed to emit, and its validation.

Deliberately a closed set. A computer-use agent that can emit arbitrary
shell strings is a different (and much worse) security proposition than one
whose entire surface is "click a point, type a string, press a key" — every
action here is expressible as a synthetic input event and nothing else.

Coordinates are normalised to 0-1000 on both axes rather than pixels. Vision
models are trained to point in normalised space (Gemini emits [0,1000]
natively), and it decouples the model's output from whatever resolution the
backend happens to be running, so a trace recorded at 1280x800 replays
against a 1920x1080 screen without rewriting coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


MAX_COORD = 1000


class ActionError(ValueError):
    """A model proposed something outside the allowed vocabulary/ranges."""


@dataclass(frozen=True)
class Action:
    kind: str
    # click/scroll target, normalised 0-1000
    x: int | None = None
    y: int | None = None
    text: str | None = None
    key: str | None = None
    # scroll magnitude in notches; negative scrolls up
    amount: int | None = None
    seconds: float | None = None
    # free-text, shown to the user and stored in the trace; never executed
    reason: str = ""
    # terminal actions carry the agent's answer/summary
    result: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, "")}


KINDS = {"click", "double_click", "right_click", "type", "key", "scroll",
         "wait", "screenshot", "done", "fail", "ask"}

# Terminal kinds end the run. `ask` is terminal too: a computer-use agent
# that pauses mid-run for input is holding live UI state hostage, so it hands
# the question back and the caller decides whether to start a follow-up run.
TERMINAL = {"done", "fail", "ask"}

_NEEDS_POINT = {"click", "double_click", "right_click", "scroll"}


def _coord(value: Any, axis: str) -> int:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ActionError(f"{axis} must be a number, got {value!r}")
    value = int(value)
    if not 0 <= value <= MAX_COORD:
        raise ActionError(f"{axis}={value} outside 0..{MAX_COORD}")
    return value


def parse_action(raw: dict) -> Action:
    """Validate one model-proposed action.

    Raises ActionError rather than coercing. A malformed action means the
    model is confused about the screen, and silently rounding a bad
    coordinate into range would click somewhere real and arbitrary — the
    agent loop feeds the error back as an observation instead, which is
    recoverable and shows up in the trace.
    """
    if not isinstance(raw, dict):
        raise ActionError(f"action must be an object, got {type(raw).__name__}")

    kind = raw.get("kind")
    if kind not in KINDS:
        raise ActionError(f"unknown action kind {kind!r}; allowed: {sorted(KINDS)}")

    kwargs: dict[str, Any] = {"kind": kind, "reason": str(raw.get("reason") or "")}

    if kind in _NEEDS_POINT:
        kwargs["x"] = _coord(raw.get("x"), "x")
        kwargs["y"] = _coord(raw.get("y"), "y")

    if kind == "type":
        text = raw.get("text")
        if not isinstance(text, str) or text == "":
            raise ActionError("type requires a non-empty 'text'")
        kwargs["text"] = text

    if kind == "key":
        key = raw.get("key")
        if not isinstance(key, str) or key == "":
            raise ActionError("key requires a non-empty 'key' (e.g. 'Return', 'ctrl+a')")
        kwargs["key"] = key

    if kind == "scroll":
        amount = raw.get("amount", 3)
        if not isinstance(amount, (int, float)) or isinstance(amount, bool):
            raise ActionError(f"scroll amount must be a number, got {amount!r}")
        if amount == 0:
            raise ActionError("scroll amount must be non-zero")
        kwargs["amount"] = int(amount)

    if kind == "wait":
        seconds = raw.get("seconds", 1.0)
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
            raise ActionError(f"wait seconds must be a number, got {seconds!r}")
        # Capped: a model that decides to wait 300s has effectively hung the
        # run, and the step budget can't fire while we're blocked in sleep.
        if not 0 < seconds <= 30:
            raise ActionError(f"wait seconds must be in (0, 30], got {seconds}")
        kwargs["seconds"] = float(seconds)

    if kind in TERMINAL:
        kwargs["result"] = str(raw.get("result") or raw.get("answer") or "")

    return Action(**kwargs)


def to_pixels(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    """Normalised 0-1000 -> device pixels, clamped to the last valid pixel."""
    px = min(int(round(x / MAX_COORD * width)), max(width - 1, 0))
    py = min(int(round(y / MAX_COORD * height)), max(height - 1, 0))
    return px, py
