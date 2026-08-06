"""A task is a goal, the facts the agent may use, and how to tell it worked.

Deliberately thin. The agent loop knows nothing about jobs, browsers or
forms; everything domain-specific lives here, which is what lets the same
core drive a job application and a spreadsheet without branching.

The `context` split matters: it is the ONLY place a task supplies real
values (names, emails, answers). The system prompt tells the model it may
not invent anything beyond that block, so a field the task didn't supply
becomes an "ask" rather than a plausible fabrication — which, on a form
submitted under someone's name, is the difference between an incomplete
application and a false one.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Task:
    name: str
    goal: str
    # Facts the agent may use. Rendered verbatim into the prompt.
    context: dict = field(default_factory=dict)
    max_steps: int = 40

    def render_context(self) -> str:
        if not self.context:
            return ""
        return "\n".join(f"- {k}: {v}" for k, v in self.context.items() if v not in (None, ""))
