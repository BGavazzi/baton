"""The perceive → propose → act loop, plus the guards that make it safe to
leave running.

Most of this file is guards rather than loop, which is the honest ratio. The
loop itself is a dozen lines; what makes an agent with real mouse control
something you can walk away from is the budget, the stall detector, the
dry-run and the kill switch.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .actions import Action, ActionError, TERMINAL, to_pixels
from .backends.base import Backend
from .providers.base import Provider, Step
from .trace import Trace


@dataclass
class Budget:
    """Hard limits. A vision-model loop with no ceiling is an open-ended bill
    and an open-ended blast radius; both are capped here rather than trusted
    to the model's own judgement about when it is finished."""
    max_steps: int = 40
    max_seconds: float = 600.0
    # Identical consecutive actions almost always mean the model cannot see
    # that its last action did nothing (a dead button, an unfocused field).
    # Left alone it will happily click the same pixel until the step budget
    # runs out, so break early and say so.
    max_repeats: int = 3


@dataclass
class RunResult:
    status: str            # done | fail | ask | budget_exhausted | stalled | aborted
    result: str = ""
    steps: int = 0
    seconds: float = 0.0
    trace_path: str | None = None
    history: list[Step] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "done"


class Agent:
    def __init__(self, backend: Backend, provider: Provider, *,
                 budget: Budget | None = None, trace: Trace | None = None,
                 dry_run: bool = False, on_step=None):
        self.backend = backend
        self.provider = provider
        self.budget = budget or Budget()
        self.trace = trace
        # dry_run proposes and records everything but sends no input events —
        # the only honest way to watch what an agent WOULD do on a real screen
        # (and how the demo is filmed without touching anything).
        self.dry_run = dry_run
        # Return False from on_step to abort — the kill switch. Called before
        # the action is executed, never after.
        self.on_step = on_step

    def run(self, goal: str, extra_context: str = "") -> RunResult:
        started = time.monotonic()
        history: list[Step] = []
        repeats = 0
        last_signature = None

        while True:
            elapsed = time.monotonic() - started

            if len(history) >= self.budget.max_steps:
                return self._finish("budget_exhausted", history, started,
                                    f"stopped after {len(history)} steps")
            if elapsed >= self.budget.max_seconds:
                return self._finish("budget_exhausted", history, started,
                                    f"stopped after {elapsed:.0f}s")

            screenshot = self.backend.screenshot()

            try:
                action = self.provider.propose(goal, screenshot, history, extra_context)
            except (ActionError, ValueError) as exc:
                # A malformed proposal is recoverable: hand the error back as
                # the observation and let the model correct itself. It still
                # consumes a step, so a model that can only emit garbage
                # terminates on budget instead of spinning for free.
                history.append(Step(action=Action(kind="screenshot",
                                                  reason="invalid proposal"),
                                    observation=f"Your last response was rejected: {exc}. "
                                                "Reply with exactly one valid JSON action."))
                if self.trace:
                    self.trace.record(len(history), screenshot, None, str(exc))
                continue

            # `amount` belongs in the signature: scrolling up then down is a
            # model probing a page, not a model stuck on one dead control.
            signature = (action.kind, action.x, action.y, action.text,
                         action.key, action.amount)
            repeats = repeats + 1 if signature == last_signature else 0
            last_signature = signature
            if repeats >= self.budget.max_repeats and action.kind not in TERMINAL:
                if self.trace:
                    self.trace.record(len(history) + 1, screenshot, action, "stalled")
                return self._finish("stalled", history, started,
                                    f"repeated {action.kind} {repeats + 1}x with no visible change")

            if self.on_step is not None and self.on_step(len(history) + 1, action) is False:
                if self.trace:
                    self.trace.record(len(history) + 1, screenshot, action, "aborted")
                return self._finish("aborted", history, started, "aborted by caller")

            if action.kind in TERMINAL:
                if self.trace:
                    self.trace.record(len(history) + 1, screenshot, action, action.result)
                history.append(Step(action=action, observation=action.result))
                return self._finish(action.kind, history, started, action.result)

            observation = self._execute(action)
            history.append(Step(action=action, observation=observation))
            if self.trace:
                self.trace.record(len(history), screenshot, action, observation)

    # ── execution ──
    def _execute(self, action: Action) -> str:
        if self.dry_run:
            return "dry run: not executed"
        try:
            width, height = self.backend.size

            if action.kind in ("click", "double_click", "right_click"):
                px, py = to_pixels(action.x, action.y, width, height)
                button = 3 if action.kind == "right_click" else 1
                count = 2 if action.kind == "double_click" else 1
                self.backend.click(px, py, button=button, count=count)
                return f"{action.kind} at ({px},{py})"

            if action.kind == "type":
                self.backend.type_text(action.text)
                return f"typed {len(action.text)} chars"

            if action.kind == "key":
                self.backend.key(action.key)
                return f"pressed {action.key}"

            if action.kind == "scroll":
                px, py = to_pixels(action.x, action.y, width, height)
                self.backend.scroll(px, py, action.amount)
                return f"scrolled {action.amount} at ({px},{py})"

            if action.kind == "wait":
                time.sleep(action.seconds)
                return f"waited {action.seconds}s"

            if action.kind == "screenshot":
                return "re-observed"

            return f"unhandled action kind {action.kind}"
        except Exception as exc:
            # Backend failures are observations, not crashes: a failed click
            # is something the model can see and route around, and a run that
            # dies on one bad event loses the whole trace.
            return f"action failed: {exc}"

    def _finish(self, status: str, history: list[Step], started: float, result: str) -> RunResult:
        return RunResult(
            status=status,
            result=result,
            steps=len(history),
            seconds=time.monotonic() - started,
            trace_path=self.trace.path if self.trace else None,
            history=history,
        )
