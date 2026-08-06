"""X11 backend: xdotool for input, ImageMagick `import` for capture.

Why synthetic X11 events rather than a browser automation protocol: there is
no CDP connection, no injected script, no patched binary and no
`navigator.webdriver` — from the application's side these are ordinary
events off the X server, indistinguishable from a hand on a mouse. That is a
different detection class than even a patched Playwright, which is the whole
reason this backend exists.

It also means the agent is not browser-specific: the same loop drives a
terminal, a native app, or a PDF viewer.

Runs anywhere with a DISPLAY — including inside a container whose Xvfb is
exposed over VNC, which is how it is driven remotely.
"""
from __future__ import annotations

import re
import shutil
import subprocess

from .base import Backend


class X11Error(RuntimeError):
    pass


_GEOM_RE = re.compile(r"(\d+)x(\d+)")


class X11Backend(Backend):
    def __init__(self, display: str = ":0", *, runner=subprocess.run):
        self.display = display
        # Injected so tests exercise the real command construction without
        # needing an X server — the commands ARE the logic here.
        self._run = runner
        self._size: tuple[int, int] | None = None
        for tool in ("xdotool", "import"):
            if shutil.which(tool) is None and runner is subprocess.run:
                raise X11Error(
                    f"{tool!r} not found. Install xdotool and imagemagick "
                    "(apt-get install -y xdotool imagemagick)."
                )

    # ── plumbing ──
    def _exec(self, args: list[str], *, capture: bool = False):
        env_args = ["env", f"DISPLAY={self.display}"] + args
        proc = self._run(env_args, capture_output=True, check=False)
        if getattr(proc, "returncode", 0) != 0:
            err = (getattr(proc, "stderr", b"") or b"")
            if isinstance(err, bytes):
                err = err.decode("utf-8", "replace")
            raise X11Error(f"{' '.join(args)} failed: {err.strip()}")
        return proc.stdout if capture else None

    # ── observation ──
    @property
    def size(self) -> tuple[int, int]:
        if self._size is None:
            out = self._exec(["xdotool", "getdisplaygeometry"], capture=True)
            if isinstance(out, bytes):
                out = out.decode("utf-8", "replace")
            parts = (out or "").split()
            if len(parts) < 2:
                raise X11Error(f"could not parse display geometry from {out!r}")
            self._size = (int(parts[0]), int(parts[1]))
        return self._size

    def screenshot(self) -> bytes:
        # `import -window root` grabs the whole screen; png:- writes to stdout
        # so nothing touches disk (traces decide what to persist).
        out = self._exec(["import", "-window", "root", "png:-"], capture=True)
        if not out:
            raise X11Error("screenshot produced no data")
        return out

    # ── input ──
    def click(self, x: int, y: int, button: int = 1, count: int = 1) -> None:
        self._exec(["xdotool", "mousemove", str(x), str(y)])
        for _ in range(max(count, 1)):
            self._exec(["xdotool", "click", str(button)])

    def type_text(self, text: str) -> None:
        # --clearmodifiers so a stuck Shift/Ctrl from an earlier chord cannot
        # corrupt the string. --delay gives a human-ish cadence and, more
        # practically, stops fast typing from outrunning JS input handlers.
        self._exec(["xdotool", "type", "--clearmodifiers", "--delay", "40", "--", text])

    def key(self, combo: str) -> None:
        self._exec(["xdotool", "key", "--clearmodifiers", combo])

    def scroll(self, x: int, y: int, amount: int) -> None:
        # X11 has no scroll axis: buttons 4/5 ARE scroll up/down.
        button = "4" if amount < 0 else "5"
        self._exec(["xdotool", "mousemove", str(x), str(y)])
        for _ in range(abs(int(amount))):
            self._exec(["xdotool", "click", button])
