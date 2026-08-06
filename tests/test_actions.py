"""The action vocabulary is the agent's entire security boundary — anything
that gets past parse_action becomes a real input event on a real screen — so
it is tested as a boundary, not as a parser."""
import pytest

from baton.actions import Action, ActionError, parse_action, to_pixels


def test_click_requires_coordinates():
    with pytest.raises(ActionError):
        parse_action({"kind": "click"})


@pytest.mark.parametrize("x,y", [(-1, 500), (500, -1), (1001, 500), (500, 1001)])
def test_coordinates_outside_the_normalised_range_are_rejected(x, y):
    """Rejected, never clamped: a clamped bad coordinate still clicks
    somewhere real and arbitrary."""
    with pytest.raises(ActionError):
        parse_action({"kind": "click", "x": x, "y": y})


def test_booleans_are_not_accepted_as_coordinates():
    """bool is an int subclass in Python, so a naive isinstance check lets
    True through as x=1."""
    with pytest.raises(ActionError):
        parse_action({"kind": "click", "x": True, "y": 10})


def test_unknown_kind_is_rejected():
    with pytest.raises(ActionError) as exc:
        parse_action({"kind": "run_shell", "text": "rm -rf /"})
    assert "run_shell" in str(exc.value)


def test_type_requires_non_empty_text():
    with pytest.raises(ActionError):
        parse_action({"kind": "type", "text": ""})


def test_scroll_amount_cannot_be_zero():
    with pytest.raises(ActionError):
        parse_action({"kind": "scroll", "x": 1, "y": 1, "amount": 0})


def test_scroll_defaults_to_a_usable_amount():
    action = parse_action({"kind": "scroll", "x": 1, "y": 1})
    assert action.amount == 3


@pytest.mark.parametrize("seconds", [0, -1, 31, 600])
def test_wait_is_capped(seconds):
    """An uncapped wait hangs the run in sleep, where the step and time
    budgets cannot fire."""
    with pytest.raises(ActionError):
        parse_action({"kind": "wait", "seconds": seconds})


def test_terminal_actions_carry_their_result():
    action = parse_action({"kind": "done", "result": "submitted the form"})
    assert action.kind == "done" and action.result == "submitted the form"


def test_non_dict_action_is_rejected():
    with pytest.raises(ActionError):
        parse_action(["click"])


def test_to_dict_omits_unset_fields():
    assert parse_action({"kind": "key", "key": "Return"}).to_dict() == {
        "kind": "key", "key": "Return"}


# ── coordinate mapping ──

def test_to_pixels_maps_the_normalised_corners():
    assert to_pixels(0, 0, 1920, 1080) == (0, 0)
    assert to_pixels(500, 500, 1920, 1080) == (960, 540)


def test_to_pixels_never_returns_an_out_of_bounds_pixel():
    """1000 maps to width, which is one past the last addressable pixel."""
    assert to_pixels(1000, 1000, 1920, 1080) == (1919, 1079)


def test_to_pixels_is_resolution_independent():
    """The same normalised point lands proportionally on any screen — this is
    what lets a trace recorded at one resolution replay against another."""
    assert to_pixels(250, 250, 800, 600) == (200, 150)
    assert to_pixels(250, 250, 1600, 1200) == (400, 300)
