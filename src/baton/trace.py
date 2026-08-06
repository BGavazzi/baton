"""Every step, on disk: the screenshot the model saw, what it proposed, and
what happened.

This is the part hobby computer-use demos skip, and the part that makes the
thing debuggable. When a run goes wrong the only useful question is "what did
it actually see at step 7", and without the frame that produced the decision
there is no way to tell a perception failure from a reasoning failure.

Layout — a directory per run, so a trace is one thing you can zip and send:

    traces/2026-08-06T01-42-11_apply-to-job/
        meta.json          goal, provider, backend, budget, outcome
        steps.jsonl        one row per step
        step-001.png       exactly the frame passed to the model
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone


def _slug(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "run").lower()).strip("-")
    return (slug[:limit].rstrip("-") or "run")


class Trace:
    def __init__(self, root: str = "traces", label: str = "run", *,
                 meta: dict | None = None, save_screenshots: bool = True):
        # UTC with a filename-safe separator: these sort chronologically in a
        # plain directory listing, which is how they actually get browsed.
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        self.path = os.path.join(root, f"{stamp}_{_slug(label)}")
        os.makedirs(self.path, exist_ok=True)
        self.save_screenshots = save_screenshots
        self._steps_file = os.path.join(self.path, "steps.jsonl")
        self._meta = dict(meta or {})
        self._meta.setdefault("label", label)
        self._meta.setdefault("started_at", datetime.now(timezone.utc).isoformat())
        self._write_meta()

    def _write_meta(self) -> None:
        with open(os.path.join(self.path, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(self._meta, fh, indent=2, ensure_ascii=False)

    def record(self, step: int, screenshot_png: bytes | None,
               action=None, observation: str = "") -> None:
        frame_name = None
        if screenshot_png and self.save_screenshots:
            frame_name = f"step-{step:03d}.png"
            with open(os.path.join(self.path, frame_name), "wb") as fh:
                fh.write(screenshot_png)

        row = {
            "step": step,
            "at": datetime.now(timezone.utc).isoformat(),
            "screenshot": frame_name,
            "action": action.to_dict() if action is not None else None,
            "observation": observation,
        }
        with open(self._steps_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def finish(self, result) -> None:
        self._meta.update({
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": getattr(result, "status", None),
            "result": getattr(result, "result", ""),
            "steps": getattr(result, "steps", None),
            "seconds": round(getattr(result, "seconds", 0.0), 2),
        })
        self._write_meta()

    def steps(self) -> list[dict]:
        """Read the recorded steps back — used by the replay/eval harness."""
        if not os.path.exists(self._steps_file):
            return []
        with open(self._steps_file, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
