# baton

A computer-use agent that drives a real desktop with synthetic X11 input, instead of a browser automation protocol.

It looks at the screen, decides one action, sends it as an ordinary input event, and looks again. That is the whole loop. Most of the code is the parts around it — the action boundary, the budgets, the trace — because those are what make an agent with real mouse control something you can leave running.

```
screenshot ──▶ provider ──▶ action ──▶ backend ──▶ screen
     ▲                         │                      │
     └─────────── observation ◀┴──────────────────────┘
```

## Why input events instead of CDP

Browser automation frameworks attach a debugger to the browser. That connection is observable, and so is everything downstream of it — `navigator.webdriver`, CDP runtime probes, patched binaries, and in headless mode a different browser build entirely (`chromium_headless_shell`, which is not Chrome with the window hidden).

`baton` never attaches to the application. It talks to the X server, and the application receives the same events it would from a hand on a mouse. There is nothing to detect because there is no automation channel — the cost is that it can only do what a person could do through the UI, which is also, usefully, the ceiling on what it can do wrong.

It also isn't browser-specific. The same loop drives a terminal, a native app, or a PDF viewer.

## Design

**One core, many tasks.** The agent loop knows nothing about jobs, browsers, or forms. Tasks are thin: a goal string, some context, and success criteria.

**A closed action vocabulary.** The model may click, type, press keys, scroll, wait, and terminate. That's it. There is no action that takes a shell string, so the blast radius is bounded by what is reachable through the UI. Everything crosses one validating boundary (`actions.parse_action`) that rejects rather than coerces — a clamped out-of-range coordinate still clicks somewhere real and arbitrary.

**Normalised coordinates.** The model points in 0–1000 space on both axes, not pixels. Vision models are trained to point this way (Gemini emits `[0,1000]` natively), and it decouples decisions from resolution — a trace recorded at 1280×800 replays against 1920×1080 without rewriting anything.

**Pluggable providers.** Gemini by default (native normalised pointing, cheap enough that a 40-step run isn't a budgeting decision). The provider interface is one method, so a local model or a different vendor slots in without touching the loop.

**Budgets, not vibes.** `max_steps`, `max_seconds`, and a stall detector that stops a model repeating an action that visibly isn't working. An unbounded vision loop is an open-ended bill and an open-ended blast radius.

**Traces.** Every run writes a directory: the exact frame passed to the model at each step, the action it proposed, and what happened. When a run goes wrong the only useful question is *what did it actually see at step 7*, and without that frame you cannot tell a perception failure from a reasoning failure.

**Dry run.** Proposes and records everything, sends no input. The honest way to watch what an agent *would* do on a real screen.

**Replay, so regressions are catchable.** You cannot regression-test an agent by running it — a live run touches UI that changes underneath you, costs a full loop of vision calls, and fails for reasons unrelated to your change. Traces already hold the exact frames the model saw, so `replay.compare()` feeds them back and diffs what the model does *now* against what it did, and `replay.check()` re-validates recorded actions against the current action schema with no provider at all: free, instant, and it catches the regression most likely to pass silently — tightening the vocabulary so previously-legal behaviour is now illegal.

## Layout

```
src/baton/
  actions.py           the vocabulary + its validation (the security boundary)
  agent.py             perceive → propose → act, and the guards
  trace.py             per-run directory: frames, steps.jsonl, meta.json
  backends/
    base.py            what a controllable screen has to do
    x11.py             xdotool + imagemagick; runs anywhere with a DISPLAY
  replay.py            eval harness: re-run recorded traces, no live screen
  providers/
    base.py            provider interface, system prompt, response salvaging
    gemini.py          Gemini vision
  tasks/               goal + context + success criteria, per task
```

## Status

Core loop, action boundary, X11 backend, Gemini provider, tracing, the task layer and the replay harness are in place — 48 tests, all with a scripted provider and fake screen, so no network and no X server. The X11 backend is additionally verified live against a real Xvfb display (geometry, PNG capture, coordinate mapping, real key events).

Next: the flagship job-application task on top of the task layer.

## Running the tests

```bash
pip install -r requirements.txt
python -m pytest
```

## Requirements

The core has no third-party runtime dependencies — the X11 backend shells out, the Gemini provider uses `urllib`.

```bash
apt-get install -y xdotool imagemagick   # for the X11 backend
export GEMINI_API_KEY=...                # for the Gemini provider
```
