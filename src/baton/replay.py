"""Replay recorded traces against a provider, without a live screen.

This is the eval harness, and it exists because of a specific problem: you
cannot regression-test an agent by running it. A live run touches real UI
that changes underneath you, costs a full loop of vision calls, takes
minutes, and fails for reasons that have nothing to do with your change.

A trace already contains the exact frames the model saw. Replaying feeds
those frames back and asks what the model does *now* — same inputs, current
prompt/model/parser. That turns "did my prompt change break anything" from a
manual afternoon into a deterministic check.

Two modes, answering different questions:

  compare — re-ask the provider for each recorded frame and diff against
            what was recorded. Catches "my prompt edit made step 7 click
            somewhere else." Costs one call per step.

  check   — no provider at all. Re-validates recorded actions through the
            current parser and re-applies the guards. Free and instant, and
            catches the whole class of "the action schema changed and old
            behaviour is now illegal" — which is the change most likely to
            break an agent silently.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .actions import Action, ActionError, parse_action
from .providers.base import Step


@dataclass
class StepDiff:
    step: int
    recorded: dict | None
    current: dict | None = None
    verdict: str = ""          # same | drifted | invalid | error

    @property
    def ok(self) -> bool:
        return self.verdict == "same"


@dataclass
class ReplayReport:
    trace_path: str
    mode: str
    diffs: list = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.diffs)

    @property
    def agreed(self) -> int:
        return sum(1 for d in self.diffs if d.ok)

    @property
    def ok(self) -> bool:
        return all(d.ok for d in self.diffs)

    @property
    def agreement(self) -> float:
        return (self.agreed / self.total) if self.total else 1.0

    def summary(self) -> str:
        head = (f"{os.path.basename(self.trace_path)} [{self.mode}] "
                f"{self.agreed}/{self.total} agreed ({self.agreement:.0%})")
        bad = [d for d in self.diffs if not d.ok]
        if not bad:
            return head
        lines = [head]
        for d in bad:
            lines.append(f"  step {d.step}: {d.verdict}")
            lines.append(f"    was: {json.dumps(d.recorded, ensure_ascii=False)}")
            if d.current is not None:
                lines.append(f"    now: {json.dumps(d.current, ensure_ascii=False)}")
        return "\n".join(lines)


def _recorded_steps(trace_dir: str) -> list[dict]:
    path = os.path.join(trace_dir, "steps.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no steps.jsonl in {trace_dir}")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _same(a: dict, b: dict) -> bool:
    """Compare on what the action DOES, ignoring `reason`.

    Reason is free-text rationale that changes wording constantly without
    changing behaviour; diffing on it would make every replay noisy and the
    harness useless.
    """
    drop = lambda d: {k: v for k, v in (d or {}).items() if k != "reason"}
    return drop(a) == drop(b)


def check(trace_dir: str) -> ReplayReport:
    """Re-validate recorded actions against the CURRENT action schema.

    No provider, no network, no screen. Catches the change most likely to
    break an agent quietly: tightening the vocabulary so behaviour that used
    to be legal no longer is.
    """
    report = ReplayReport(trace_path=trace_dir, mode="check")
    for row in _recorded_steps(trace_dir):
        recorded = row.get("action")
        if recorded is None:
            continue  # a rejected proposal; nothing to re-validate
        try:
            parse_action(recorded)
            verdict = "same"
        except ActionError as exc:
            verdict = f"invalid: {exc}"
        report.diffs.append(StepDiff(step=row.get("step", 0), recorded=recorded,
                                     verdict=verdict))
    return report


def compare(trace_dir: str, provider, goal: str | None = None,
            extra_context: str = "") -> ReplayReport:
    """Re-ask `provider` for every recorded frame and diff against what was
    recorded.

    History is rebuilt from the RECORDED actions rather than the provider's
    new answers, so each step is judged against the same context it
    originally had. Letting the replay diverge would mean step 8's
    disagreement was caused by step 3's, and the report would blame the
    wrong step.
    """
    meta_path = os.path.join(trace_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    goal = goal or meta.get("goal") or meta.get("label") or ""

    report = ReplayReport(trace_path=trace_dir, mode="compare")
    history: list[Step] = []

    for row in _recorded_steps(trace_dir):
        recorded = row.get("action")
        frame_name = row.get("screenshot")
        if recorded is None or not frame_name:
            continue
        frame_path = os.path.join(trace_dir, frame_name)
        if not os.path.exists(frame_path):
            report.diffs.append(StepDiff(step=row.get("step", 0), recorded=recorded,
                                         verdict="error: frame missing"))
            continue

        with open(frame_path, "rb") as fh:
            screenshot = fh.read()

        try:
            proposed = provider.propose(goal, screenshot, history, extra_context)
            current = proposed.to_dict()
            verdict = "same" if _same(recorded, current) else "drifted"
        except Exception as exc:
            current, verdict = None, f"error: {exc}"

        report.diffs.append(StepDiff(step=row.get("step", 0), recorded=recorded,
                                     current=current, verdict=verdict))

        # Advance history using the RECORDED action, not the new one.
        try:
            history.append(Step(action=parse_action(recorded),
                                observation=row.get("observation", "")))
        except ActionError:
            history.append(Step(action=Action(kind="screenshot"),
                                observation=row.get("observation", "")))

    return report


def check_all(traces_root: str = "traces") -> list[ReplayReport]:
    """Schema-check every trace on disk — the cheap suite to run in CI."""
    if not os.path.isdir(traces_root):
        return []
    reports = []
    for name in sorted(os.listdir(traces_root)):
        path = os.path.join(traces_root, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "steps.jsonl")):
            reports.append(check(path))
    return reports
