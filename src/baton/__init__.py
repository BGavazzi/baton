"""baton — a computer-use agent that drives a real desktop via synthetic X11
input, rather than a browser automation protocol."""

from .actions import Action, parse_action, to_pixels
from .agent import Agent, Budget, RunResult
from .trace import Trace

__all__ = ["Action", "parse_action", "to_pixels", "Agent", "Budget", "RunResult", "Trace"]
__version__ = "0.1.0"
