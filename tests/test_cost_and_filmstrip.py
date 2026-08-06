"""Per-step cost ledger, and rendering a trace into something watchable."""
import os

import pytest

from baton.actions import Action
from baton.cost import Ledger, Pricing
from baton.trace import Trace

PIL = pytest.importorskip("PIL", reason="Pillow only needed for filmstrip rendering")
from baton.filmstrip import FilmstripError, filmstrip, gif  # noqa: E402


# ── cost ledger ──

def test_cost_is_computed_from_injected_pricing():
    """Vendor pricing changes without warning; a stale hardcoded constant
    produces confidently wrong numbers, which is worse than none."""
    ledger = Ledger(pricing=Pricing(input_per_mtok=1.0, output_per_mtok=3.0))
    entry = ledger.record(1, input_tokens=1_000_000, output_tokens=1_000_000)
    assert entry.cost == pytest.approx(4.0)


def test_images_are_billed_separately():
    ledger = Ledger(pricing=Pricing(per_image=0.002))
    assert ledger.record(1, images=3).cost == pytest.approx(0.006)


def test_totals_accumulate_across_steps():
    ledger = Ledger(pricing=Pricing(input_per_mtok=2.0))
    ledger.record(1, input_tokens=500_000)
    ledger.record(2, input_tokens=500_000)
    assert ledger.total == pytest.approx(2.0)
    assert ledger.total_tokens == 1_000_000


def test_most_expensive_steps_are_identified():
    """A run that burned its budget re-reading one screen looks identical, in
    aggregate, to one that did forty productive things."""
    ledger = Ledger(pricing=Pricing(input_per_mtok=1.0))
    ledger.record(1, input_tokens=100)
    ledger.record(2, input_tokens=900_000)
    ledger.record(3, input_tokens=100)
    assert ledger.most_expensive(1)[0].step == 2


def test_most_expensive_is_stable_on_ties():
    ledger = Ledger(pricing=Pricing(input_per_mtok=1.0))
    for step in (3, 1, 2):
        ledger.record(step, input_tokens=1000)
    assert [s.step for s in ledger.most_expensive(3)] == [1, 2, 3]


def test_ceiling_check_looks_one_step_ahead():
    """Checking only what has been spent discovers the ceiling one step late,
    after the money is gone."""
    ledger = Ledger(pricing=Pricing(input_per_mtok=1.0))
    ledger.record(1, input_tokens=1_000_000)   # 1.0 spent, avg 1.0/step
    assert ledger.total == pytest.approx(1.0)
    assert ledger.would_exceed(1.5)            # 1.0 + 1.0 estimated > 1.5
    assert not ledger.would_exceed(2.5)


def test_ceiling_check_accepts_an_explicit_estimate():
    ledger = Ledger(pricing=Pricing(input_per_mtok=1.0))
    ledger.record(1, input_tokens=1_000_000)
    assert not ledger.would_exceed(1.5, next_step_estimate=0.1)


def test_empty_ledger_summarises_without_dividing_by_zero():
    assert Ledger().summary() == "No cost recorded."
    assert Ledger().cost_per_step == 0.0


def test_summary_reports_per_step_cost():
    ledger = Ledger(pricing=Pricing(input_per_mtok=1.0))
    ledger.record(1, input_tokens=1_000_000)
    ledger.record(2, input_tokens=1_000_000)
    assert "/step" in ledger.summary()
    assert "2 steps" in ledger.summary()


# ── rendering ──

def _trace_with_frames(tmp_path, n=3):
    from PIL import Image
    trace = Trace(root=str(tmp_path), label="demo", meta={"goal": "do it"})
    for i in range(1, n + 1):
        img_path = tmp_path / f"src-{i}.png"
        Image.new("RGB", (80, 50), (10 * i, 20, 30)).save(img_path)
        trace.record(i, img_path.read_bytes(),
                     Action(kind="click", x=10 * i, y=20, reason="because"),
                     "clicked")
    return trace.path


def test_gif_is_written_from_recorded_frames(tmp_path):
    out = gif(_trace_with_frames(tmp_path), scale=1.0)
    assert os.path.exists(out) and out.endswith(".gif")


def test_filmstrip_grid_is_written(tmp_path):
    out = filmstrip(_trace_with_frames(tmp_path, n=5), columns=2, scale=1.0)
    from PIL import Image
    sheet = Image.open(out)
    assert sheet.width >= 160 and sheet.height >= 150   # 2 cols x 3 rows of 80x50


def test_rendering_never_reruns_an_agent_or_touches_a_screen(tmp_path):
    """Both outputs are built purely from frames already on disk."""
    path = _trace_with_frames(tmp_path)
    before = sorted(os.listdir(path))
    gif(path, scale=1.0)
    after = sorted(os.listdir(path))
    assert set(before) <= set(after)   # only additions, nothing consumed


def test_typed_text_is_truncated_in_captions(tmp_path):
    """A filmstrip is a public artefact and typed text is the field most
    likely to carry something personal."""
    from baton.filmstrip import _caption
    caption = _caption({"kind": "type", "text": "x" * 100})
    assert len(caption) < 40 and "…" in caption


def test_a_trace_with_no_frames_fails_clearly(tmp_path):
    trace = Trace(root=str(tmp_path), label="empty")
    trace.record(1, None, Action(kind="done", result="ok"), "")
    with pytest.raises(FilmstripError):
        gif(trace.path)


def test_a_missing_trace_fails_clearly(tmp_path):
    with pytest.raises(FilmstripError):
        filmstrip(str(tmp_path / "nope"))
