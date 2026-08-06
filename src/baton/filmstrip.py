"""Turn a recorded trace into something watchable.

A trace is the right format for debugging and the wrong one for showing
anyone. "Here are 40 PNGs and a JSONL" does not communicate; a short animation
of the agent working does, and it is the artefact that actually gets shown in
a demo, a README, or an interview.

Two outputs, because they answer different questions:

  filmstrip — a single wide PNG of every frame in sequence. Skimmable at a
              glance, pasteable into a PR, and it works anywhere an image
              works.
  gif       — the run playing back at readable speed.

Both are built from frames already on disk, so nothing here re-runs an agent
or touches a live screen. Pillow is imported lazily and its absence is a
clear error rather than an import-time crash — the core has no third-party
runtime dependencies and this must not quietly add one.
"""
from __future__ import annotations

import json
import os


class FilmstripError(RuntimeError):
    pass


def _require_pillow():
    try:
        from PIL import Image, ImageDraw  # noqa: F401
        return Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise FilmstripError(
            "Pillow is required to render a trace. Install it with "
            "`pip install pillow` — the rest of baton does not need it."
        ) from exc


def _frames(trace_dir: str) -> list:
    """(step, path, action_dict) for every recorded frame, in order."""
    steps_file = os.path.join(trace_dir, "steps.jsonl")
    if not os.path.exists(steps_file):
        raise FilmstripError(f"no steps.jsonl in {trace_dir}")

    out = []
    with open(steps_file, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            name = row.get("screenshot")
            if not name:
                continue
            path = os.path.join(trace_dir, name)
            if os.path.exists(path):
                out.append((row.get("step", 0), path, row.get("action")))
    return out


def _caption(action: dict | None) -> str:
    if not action:
        return "(rejected proposal)"
    kind = action.get("kind", "?")
    if kind in ("click", "double_click", "right_click", "scroll"):
        return f"{kind} ({action.get('x')},{action.get('y')})"
    if kind == "type":
        text = str(action.get("text", ""))
        # Truncated: a filmstrip is a public artefact and typed text is the
        # one field likely to carry something personal.
        return f'type "{text[:24]}{"…" if len(text) > 24 else ""}"'
    if kind == "key":
        return f"key {action.get('key')}"
    return kind


def gif(trace_dir: str, out_path: str = "", *, ms_per_frame: int = 900,
        scale: float = 0.5) -> str:
    """Render the run as an animated GIF. Returns the written path."""
    Image, _ = _require_pillow()
    frames = _frames(trace_dir)
    if not frames:
        raise FilmstripError(f"no usable frames in {trace_dir}")

    out_path = out_path or os.path.join(trace_dir, "run.gif")
    images = []
    for _, path, _action in frames:
        img = Image.open(path).convert("RGB")
        if scale != 1.0:
            img = img.resize((max(1, int(img.width * scale)),
                              max(1, int(img.height * scale))))
        images.append(img)

    images[0].save(out_path, save_all=True, append_images=images[1:],
                   duration=ms_per_frame, loop=0, optimize=True)
    return out_path


def filmstrip(trace_dir: str, out_path: str = "", *, columns: int = 4,
              scale: float = 0.35, label: bool = True) -> str:
    """Render every frame into one wide PNG grid. Returns the written path."""
    Image, ImageDraw = _require_pillow()
    frames = _frames(trace_dir)
    if not frames:
        raise FilmstripError(f"no usable frames in {trace_dir}")

    thumbs = []
    for step, path, action in frames:
        img = Image.open(path).convert("RGB")
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
        if label:
            draw = ImageDraw.Draw(img)
            text = f"{step}. {_caption(action)}"
            # Filled backing box, so the caption stays readable over both a
            # white form and a dark terminal.
            draw.rectangle([0, 0, img.width, 16], fill=(0, 0, 0))
            draw.text((4, 3), text, fill=(255, 255, 255))
        thumbs.append(img)

    columns = max(1, columns)
    rows = (len(thumbs) + columns - 1) // columns
    cell_w = max(t.width for t in thumbs)
    cell_h = max(t.height for t in thumbs)

    sheet = Image.new("RGB", (cell_w * columns, cell_h * rows), (24, 24, 24))
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % columns) * cell_w, (i // columns) * cell_h))

    out_path = out_path or os.path.join(trace_dir, "filmstrip.png")
    sheet.save(out_path, optimize=True)
    return out_path
