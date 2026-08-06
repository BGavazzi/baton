"""Agent loop behaviour, driven by a scripted provider and a fake screen.

The guards are the subject here. A loop that reaches "done" on a happy path
is easy; what matters is that it stops when the model won't.
"""
import pytest

from baton.actions import Action
from baton.agent import Agent, Budget
from baton.backends.base import Backend
from baton.providers.base import Provider, extract_action


class FakeScreen(Backend):
    def __init__(self, size=(1000, 1000)):
        self._size = size
        self.events = []

    @property
    def size(self):
        return self._size

    def screenshot(self):
        return b"\x89PNG-fake"

    def click(self, x, y, button=1, count=1):
        self.events.append(("click", x, y, button, count))

    def type_text(self, text):
        self.events.append(("type", text))

    def key(self, combo):
        self.events.append(("key", combo))

    def scroll(self, x, y, amount):
        self.events.append(("scroll", x, y, amount))


class ScriptedProvider(Provider):
    """Emits a fixed sequence, then repeats the last one forever."""
    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = 0
        self.seen_history = []

    def propose(self, goal, screenshot_png, history, extra_context=""):
        self.seen_history.append(len(history))
        action = self.actions[min(self.calls, len(self.actions) - 1)]
        self.calls += 1
        if isinstance(action, Exception):
            raise action
        return action


def test_reaches_done_and_reports_the_result():
    provider = ScriptedProvider([
        Action(kind="click", x=500, y=500),
        Action(kind="done", result="finished"),
    ])
    result = Agent(FakeScreen(), provider).run("do the thing")

    assert result.ok and result.status == "done"
    assert result.result == "finished"


def test_click_is_translated_from_normalised_to_device_pixels():
    screen = FakeScreen(size=(1920, 1080))
    provider = ScriptedProvider([
        Action(kind="click", x=500, y=500),
        Action(kind="done", result="ok"),
    ])
    Agent(screen, provider).run("click the middle")

    assert screen.events[0] == ("click", 960, 540, 1, 1)


def test_stops_when_the_model_repeats_itself():
    """A model that cannot see its click did nothing will click forever."""
    provider = ScriptedProvider([Action(kind="click", x=10, y=10)])
    result = Agent(FakeScreen(), provider, budget=Budget(max_repeats=3)).run("stuck")

    assert result.status == "stalled"
    assert result.steps < 40


def test_step_budget_is_enforced():
    # Alternating actions so the stall detector never trips first.
    provider = ScriptedProvider([Action(kind="scroll", x=1, y=1, amount=3),
                                 Action(kind="scroll", x=1, y=1, amount=-3)] * 50)
    result = Agent(FakeScreen(), provider, budget=Budget(max_steps=5)).run("scroll forever")

    assert result.status == "budget_exhausted"
    assert result.steps == 5


def test_malformed_proposals_are_fed_back_instead_of_crashing():
    """One bad response should be recoverable — the model gets told why."""
    provider = ScriptedProvider([
        ValueError("not JSON"),
        Action(kind="done", result="recovered"),
    ])
    result = Agent(FakeScreen(), provider).run("recover")

    assert result.ok and result.result == "recovered"
    assert "rejected" in result.history[0].observation


def test_a_model_that_only_emits_garbage_still_terminates():
    provider = ScriptedProvider([ValueError("nope")])
    result = Agent(FakeScreen(), provider, budget=Budget(max_steps=4)).run("garbage")

    assert result.status == "budget_exhausted"


def test_dry_run_proposes_but_sends_no_input():
    screen = FakeScreen()
    provider = ScriptedProvider([
        Action(kind="click", x=1, y=1),
        Action(kind="type", text="hello"),
        Action(kind="done", result="done"),
    ])
    result = Agent(screen, provider, dry_run=True).run("look but do not touch")

    assert result.ok
    assert screen.events == []
    assert all(step.observation == "dry run: not executed" for step in result.history[:-1])


def test_on_step_can_abort_before_the_action_runs():
    screen = FakeScreen()
    provider = ScriptedProvider([Action(kind="click", x=1, y=1)])
    result = Agent(screen, provider, on_step=lambda n, a: False).run("abort me")

    assert result.status == "aborted"
    assert screen.events == []


def test_backend_failures_become_observations_not_crashes():
    class BrokenScreen(FakeScreen):
        def click(self, x, y, button=1, count=1):
            raise RuntimeError("display gone")

    provider = ScriptedProvider([
        Action(kind="click", x=1, y=1),
        Action(kind="done", result="carried on"),
    ])
    result = Agent(BrokenScreen(), provider).run("survive a bad click")

    assert result.ok
    assert "action failed: display gone" in result.history[0].observation


def test_ask_is_terminal():
    """Pausing mid-run would hold live UI state hostage, so a question ends
    the run and the caller decides whether to start a follow-up."""
    provider = ScriptedProvider([Action(kind="ask", result="what is the 2FA code?")])
    result = Agent(FakeScreen(), provider).run("log in")

    assert result.status == "ask"
    assert result.result == "what is the 2FA code?"
    assert not result.ok


# ── response salvaging ──

def test_extract_action_handles_a_fenced_code_block():
    action = extract_action('```json\n{"kind":"key","key":"Return"}\n```')
    assert action.kind == "key"


def test_extract_action_salvages_json_from_surrounding_prose():
    action = extract_action('Sure! {"kind":"done","result":"ok"} Hope that helps.')
    assert action.kind == "done"


def test_extract_action_still_validates_what_it_salvages():
    """Salvaging must not become a bypass around parse_action."""
    from baton.actions import ActionError
    with pytest.raises(ActionError):
        extract_action('{"kind":"click","x":9999,"y":0}')


def test_extract_action_rejects_an_empty_response():
    with pytest.raises(ValueError):
        extract_action("   ")
