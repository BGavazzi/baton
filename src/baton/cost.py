"""What a run actually cost, per step.

An agent loop's cost is invisible until the invoice arrives, and by then it
is one number for thousands of steps. That makes the useful questions
unanswerable: which task is expensive, whether a prompt change made things
worse, whether the budget is set anywhere near reality.

The ledger is per-step because that is where the decisions are. A run that
burned its budget re-reading the same screen looks identical, in aggregate,
to one that did forty productive things.

Prices are injected, never hardcoded. Vendor pricing changes without warning,
and a stale constant buried in a library produces confidently wrong numbers —
worse than no numbers, because nobody re-checks a figure that looks precise.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Pricing:
    """Cost per million tokens, in whatever currency the caller is using."""
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0
    # Vision models bill images separately from text on several providers.
    per_image: float = 0.0

    def cost(self, input_tokens: int, output_tokens: int, images: int = 0) -> float:
        return (input_tokens / 1_000_000 * self.input_per_mtok
                + output_tokens / 1_000_000 * self.output_per_mtok
                + images * self.per_image)


@dataclass
class StepCost:
    step: int
    input_tokens: int = 0
    output_tokens: int = 0
    images: int = 0
    cost: float = 0.0


@dataclass
class Ledger:
    """Per-step accounting for one run."""
    pricing: Pricing = field(default_factory=Pricing)
    steps: list = field(default_factory=list)

    def record(self, step: int, *, input_tokens: int = 0, output_tokens: int = 0,
               images: int = 0) -> StepCost:
        entry = StepCost(
            step=step, input_tokens=input_tokens, output_tokens=output_tokens,
            images=images,
            cost=self.pricing.cost(input_tokens, output_tokens, images),
        )
        self.steps.append(entry)
        return entry

    @property
    def total(self) -> float:
        return sum(s.cost for s in self.steps)

    @property
    def total_tokens(self) -> int:
        return sum(s.input_tokens + s.output_tokens for s in self.steps)

    @property
    def cost_per_step(self) -> float:
        return (self.total / len(self.steps)) if self.steps else 0.0

    def most_expensive(self, limit: int = 3) -> list:
        """Steps that cost the most. Ties break on step number so the output
        is stable and reviewable rather than hash-ordered."""
        return sorted(self.steps, key=lambda s: (-s.cost, s.step))[:limit]

    def would_exceed(self, ceiling: float, *, next_step_estimate: float | None = None) -> bool:
        """Whether continuing would blow a spend ceiling.

        Estimating the next step from the running average rather than from
        zero is the point: checking only what has already been spent means
        the ceiling is always discovered one step late, after the money is
        gone.
        """
        estimate = self.cost_per_step if next_step_estimate is None else next_step_estimate
        return (self.total + estimate) > ceiling

    def summary(self, currency: str = "$") -> str:
        if not self.steps:
            return "No cost recorded."
        lines = [
            f"{len(self.steps)} steps · {self.total_tokens:,} tokens · "
            f"{currency}{self.total:.4f} ({currency}{self.cost_per_step:.4f}/step)",
        ]
        top = self.most_expensive()
        if top and top[0].cost > 0:
            lines.append("Most expensive: " + ", ".join(
                f"step {s.step} ({currency}{s.cost:.4f})" for s in top))
        return "\n".join(lines)
