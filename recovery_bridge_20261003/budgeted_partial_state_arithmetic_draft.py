"""Unfrozen budget arithmetic for a future partial-state study.

No model calls, tokenization, scoring, experiment launch, or automatic retries.
Not used by the currently frozen five-trace study. Counts must come from a
separately qualified renderer/runtime ledger; this module cannot qualify them.
"""
from dataclasses import dataclass
from typing import Optional, Sequence


def _count(value: int, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


@dataclass(frozen=True)
class IncurredUsage:
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    qualified: bool

    def known_total(self) -> Optional[int]:
        if type(self.qualified) is not bool:
            raise ValueError('usage qualification must be a boolean')
        if self.input_tokens is not None:
            _count(self.input_tokens, 'incurred input tokens')
        if self.output_tokens is not None:
            _count(self.output_tokens, 'incurred output tokens')
        if not self.qualified or self.input_tokens is None or self.output_tokens is None:
            return None
        return (_count(self.input_tokens, 'incurred input tokens')
                + _count(self.output_tokens, 'incurred output tokens'))


@dataclass(frozen=True)
class BudgetDecision:
    send_reader: bool
    reason: str
    total_allowance: int
    prior_usage: Optional[int]
    intended_reader_input: int
    reader_output_cap: int
    planned_total_upper_bound: Optional[int]
    context_reserve: Optional[int]


def allocate_reader(*, total_allowance: int,
                    incurred: Sequence[IncurredUsage],
                    reader_input_tokens: int, maximum_output_cap: int,
                    context_limit: int, context_margin: int) -> BudgetDecision:
    """Allocate output from a common logical-token budget before transmission.

    The complete rendered reader request supplies reader_input_tokens. Every
    source/donor/compression model call required for this deployment belongs in
    incurred. Experimental sharing does not remove any such production cost.
    Margin is context headroom, not billed usage or extra logical-token budget.
    Declines stay in the planned case denominator; this function assigns no
    accuracy score and no fabricated zero cost to unknown prior usage.
    """
    budget = _count(total_allowance, 'total allowance', positive=True)
    inputs = _count(reader_input_tokens, 'reader input tokens', positive=True)
    maximum = _count(maximum_output_cap, 'maximum output cap', positive=True)
    context = _count(context_limit, 'context limit', positive=True)
    margin = _count(context_margin, 'context margin')
    totals = [receipt.known_total() for receipt in incurred]
    if any(total is None for total in totals):
        return BudgetDecision(False, 'unknown_or_unqualified_prior_usage',
                              budget, None, inputs, 0, None, None)
    prior = sum(totals)
    remaining = budget - prior - inputs
    if remaining <= 0:
        return BudgetDecision(False, 'no_positive_reader_output_allowance',
                              budget, prior, inputs, 0, None, None)
    cap = min(remaining, maximum)
    reserve = inputs + cap + margin
    if reserve > context:
        # Do not silently change context or further reduce the predeclared cap.
        return BudgetDecision(False, 'context_reserve_exceeds_limit',
                              budget, prior, inputs, 0, None, reserve)
    upper = prior + inputs + cap
    if upper > budget:
        raise ArithmeticError('planned full deployment exceeds common allowance')
    return BudgetDecision(True, 'within_common_allowance_and_context',
                          budget, prior, inputs, cap, upper, reserve)
