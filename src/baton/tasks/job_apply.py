"""The flagship task: fill in and submit a job application already open on
screen.

Thin on purpose — that is the architectural claim. The agent loop has no idea
what a job application is; everything domain-specific is the goal text and
the permitted facts below.

The important design decision here is what is NOT in the context. Anything
the agent is not explicitly given, it must ask about rather than invent. On a
form submitted under someone's real name to a real employer, a plausible
fabrication is worse than an incomplete application: the applicant is the one
who has to stand behind it in an interview, and they will not know it was
said on their behalf.
"""
from __future__ import annotations

from dataclasses import dataclass

from .base import Task


GOAL = """Complete and submit the job application currently open on screen.

Work through the form field by field. For each one, use ONLY a value from the
context you were given. Where a field is optional and you have no value for
it, leave it empty and move on.

Stop and emit "ask" — do not guess — if any of these happens:
- a REQUIRED field has no corresponding value in your context
- the form asks a free-text question (cover letter, "why this company",
  "describe a time when...") that you were not given an answer for
- you are asked for a credential, a verification code, or a CAPTCHA
- you are asked to confirm something you cannot verify from the context,
  such as a salary expectation, a start date, or a work-authorisation status

Submit only once, and only when every required field is filled. After
submitting, confirm from the screen that a success state actually appeared
before emitting "done" — a form that silently failed validation looks very
similar to one that was accepted."""


@dataclass
class JobApplication(Task):
    """An application to one posting.

    `answers` maps a question (or a distinctive fragment of one) to the
    applicant's own previously-given answer — the same idea as Clawdinha's
    answer bank. Reusing a real past answer is the difference between
    automation and ghostwriting.
    """

    def __init__(self, *, applicant: dict, job_title: str = "", company: str = "",
                 answers: dict | None = None, resume_path: str = "", max_steps: int = 60):
        context = {
            "full name": applicant.get("name", ""),
            "email": applicant.get("email", ""),
            "phone": applicant.get("phone", ""),
            "location": applicant.get("location", ""),
            "linkedin": applicant.get("linkedin", ""),
            "portfolio/github": applicant.get("github", ""),
            "résumé file to attach": resume_path,
        }
        for question, answer in (answers or {}).items():
            context[f'answer to "{question}"'] = answer

        label = " — ".join(p for p in (job_title, company) if p) or "job application"
        super().__init__(
            name=f"apply: {label}",
            goal=GOAL,
            context={k: v for k, v in context.items() if v},
            max_steps=max_steps,
        )

    def render_context(self) -> str:
        # The explicit boundary. Without it a model treats an absent field as
        # something to fill in helpfully rather than something to ask about.
        return (super().render_context() +
                "\n\nAnything not listed above is UNKNOWN. Do not invent it, do not "
                "infer it from the job posting, and do not substitute a similar value "
                "from another field — ask instead.")
