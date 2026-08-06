"""What a 'screen' has to be able to do, and nothing more.

The whole point of the abstraction is that the agent loop never learns
whether it is driving a container's Xvfb, a VNC session, or the laptop it is
running on. Backends receive device pixels; normalisation lives in
actions.to_pixels so every backend gets it identically.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Backend(ABC):
    """A controllable screen: observe it, and send synthetic input to it."""

    @property
    @abstractmethod
    def size(self) -> tuple[int, int]:
        """(width, height) in device pixels."""

    @abstractmethod
    def screenshot(self) -> bytes:
        """Current screen as PNG bytes."""

    @abstractmethod
    def click(self, x: int, y: int, button: int = 1, count: int = 1) -> None:
        """Click at device-pixel (x, y). button: 1=left, 3=right."""

    @abstractmethod
    def type_text(self, text: str) -> None:
        """Type a literal string (no key interpretation)."""

    @abstractmethod
    def key(self, combo: str) -> None:
        """Press a key or chord, X11-style: 'Return', 'ctrl+a', 'alt+Tab'."""

    @abstractmethod
    def scroll(self, x: int, y: int, amount: int) -> None:
        """Scroll at (x, y). Negative amount scrolls up."""

    def close(self) -> None:
        """Release anything held open. Safe to call more than once."""
        return None
