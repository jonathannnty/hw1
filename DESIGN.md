# DESIGN

## Decision 1: Multi-stage filtering before LLM calls

- What I decided: use symbolic length filter and keyword filter before LLM classification.
- Why: this reduces the number of expensive model calls and creates measurable per-stage dropoff.
- Agent contribution vs me: TODO (e.g., 60% me / 40% agent).

## Decision 2: Prompting strategy variants for eval

- What I decided: compare `base`, `fewshot`, and `cot` prompt variants under the same dataset.
- Why: this directly measures precision/recall/cost tradeoffs requested by the assignment.
- Agent contribution vs me: TODO.

## Decision 3: Optimization target and cost cap

- What I decided: optimize for recall under a fixed cost cap, with precision as a tie-breaker.
- Why: in monitoring systems, missing true positives is often more costly than handling some false positives.
- Agent contribution vs me: TODO.

## Notes

- Replace each TODO with concrete reflection and examples from your own workflow.
- Include links to commits or eval artifacts that support each decision.
