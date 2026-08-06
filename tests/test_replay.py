"""Replay harness: the point is catching regressions without a live screen,
so every test here runs with no network, no X server and no real frames
beyond a few bytes on disk."""
import json
import os

import pytest

from baton.actions import Action
from baton.providers.base import Provider
from baton.replay import check, check_all, compare
from baton.trace import Trace


def _write_trace(tmp_path, steps, label="run"):
    """Build a trace directory the same way Trace does, then hand back its path."""
    trace = Trace(root=str(tmp_path), label=label, meta={"goal": "do the thing"})
    for i, (action, observation) in enumerate(steps, start=1):
        trace.record(i, b"\x89PNG-fake", action, observation)
    return trace.path


class Echo(Provider):
    """Returns a scripted action per call, regardless of the frame."""
    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = 0
        self.histories = []

    def propose(self, goal, screenshot_png, history, extra_context=""):
        self.histories.append([s.action.to_dict() for s in history])
        action = self.actions[min(self.calls, len(self.actions) - 1)]
        self.calls += 1
        if isinstance(action, Exception):
            raise action
        return action


# ── check: schema re-validation, no provider ──

def test_check_passes_when_recorded_actions_are_still_valid(tmp_path):
    path = _write_trace(tmp_path, [
        (Action(kind="click", x=100, y=200), "clicked"),
        (Action(kind="done", result="ok"), "ok"),
    ])
    report = check(path)
    assert report.ok and report.total == 2


def test_check_flags_actions_the_current_schema_would_reject(tmp_path):
    """The regression most likely to pass silently: tightening the vocabulary
    so previously-legal behaviour is now illegal."""
    path = _write_trace(tmp_path, [(Action(kind="click", x=10, y=10), "clicked")])
    # Hand-edit the trace to hold an action that is no longer valid.
    steps_file = os.path.join(path, "steps.jsonl")
    rows = [json.loads(l) for l in open(steps_file, encoding="utf-8") if l.strip()]
    rows[0]["action"] = {"kind": "click", "x": 5000, "y": 10}
    with open(steps_file, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    report = check(path)
    assert not report.ok
    assert "invalid" in report.diffs[0].verdict


def test_check_skips_steps_that_recorded_no_action(tmp_path):
    """Rejected proposals are recorded with action=None; there is nothing to
    re-validate and they must not count as failures."""
    trace = Trace(root=str(tmp_path), label="rejected")
    trace.record(1, b"png", None, "invalid proposal")
    report = check(trace.path)
    assert report.total == 0 and report.ok


def test_check_all_reports_every_trace_on_disk(tmp_path):
    _write_trace(tmp_path, [(Action(kind="done", result="a"), "")], label="one")
    _write_trace(tmp_path, [(Action(kind="done", result="b"), "")], label="two")
    assert len(check_all(str(tmp_path))) == 2


def test_check_all_on_a_missing_directory_is_empty_not_an_error(tmp_path):
    assert check_all(str(tmp_path / "nope")) == []


# ── compare: re-ask the provider for each recorded frame ──

def test_compare_reports_agreement_when_the_provider_is_unchanged(tmp_path):
    path = _write_trace(tmp_path, [
        (Action(kind="click", x=100, y=200), "clicked"),
        (Action(kind="done", result="ok"), "ok"),
    ])
    provider = Echo([Action(kind="click", x=100, y=200), Action(kind="done", result="ok")])

    report = compare(path, provider)

    assert report.ok and report.agreement == 1.0


def test_compare_flags_a_drifted_step(tmp_path):
    path = _write_trace(tmp_path, [(Action(kind="click", x=100, y=200), "clicked")])
    provider = Echo([Action(kind="click", x=900, y=900)])

    report = compare(path, provider)

    assert not report.ok
    assert report.diffs[0].verdict == "drifted"
    assert report.diffs[0].current["x"] == 900


def test_compare_ignores_reason_text(tmp_path):
    """Rationale wording churns constantly without changing behaviour —
    diffing on it would make every replay noisy and the harness useless."""
    path = _write_trace(tmp_path, [
        (Action(kind="click", x=10, y=20, reason="the blue Apply button"), "clicked"),
    ])
    provider = Echo([Action(kind="click", x=10, y=20, reason="clicking Apply now")])

    assert compare(path, provider).ok


def test_compare_advances_history_from_recorded_actions_not_new_ones(tmp_path):
    """Otherwise one early disagreement cascades and the report blames the
    wrong step."""
    path = _write_trace(tmp_path, [
        (Action(kind="click", x=1, y=1), "clicked"),
        (Action(kind="type", text="hello"), "typed"),
    ])
    provider = Echo([Action(kind="click", x=999, y=999), Action(kind="type", text="hello")])

    report = compare(path, provider)

    # Step 2 saw the RECORDED step-1 click, not the provider's divergent one.
    assert provider.histories[1] == [{"kind": "click", "x": 1, "y": 1}]
    assert report.diffs[0].verdict == "drifted"
    assert report.diffs[1].verdict == "same"


def test_compare_survives_a_provider_error_on_one_step(tmp_path):
    path = _write_trace(tmp_path, [
        (Action(kind="click", x=1, y=1), "clicked"),
        (Action(kind="done", result="ok"), "ok"),
    ])
    provider = Echo([RuntimeError("rate limited"), Action(kind="done", result="ok")])

    report = compare(path, provider)

    assert "error" in report.diffs[0].verdict
    assert report.diffs[1].ok
    assert report.total == 2


def test_compare_flags_a_missing_frame_rather_than_crashing(tmp_path):
    path = _write_trace(tmp_path, [(Action(kind="click", x=1, y=1), "clicked")])
    os.remove(os.path.join(path, "step-001.png"))

    report = compare(path, Echo([Action(kind="click", x=1, y=1)]))

    assert "frame missing" in report.diffs[0].verdict


def test_compare_reads_the_goal_from_trace_meta(tmp_path):
    path = _write_trace(tmp_path, [(Action(kind="done", result="ok"), "")])
    captured = {}

    class GoalSpy(Provider):
        def propose(self, goal, screenshot_png, history, extra_context=""):
            captured["goal"] = goal
            return Action(kind="done", result="ok")

    compare(path, GoalSpy())
    assert captured["goal"] == "do the thing"


def test_summary_names_the_failing_steps(tmp_path):
    path = _write_trace(tmp_path, [
        (Action(kind="click", x=1, y=1), "clicked"),
        (Action(kind="done", result="ok"), "ok"),
    ])
    provider = Echo([Action(kind="click", x=500, y=500), Action(kind="done", result="ok")])

    text = compare(path, provider).summary()

    assert "1/2 agreed" in text
    assert "step 1" in text and "drifted" in text


def test_missing_trace_directory_raises_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        check(str(tmp_path / "not-a-trace"))
