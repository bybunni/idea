"""Thin wrapper around the Anthropic SDK with cost tracking."""

from __future__ import annotations

import logging

import anthropic

log = logging.getLogger(__name__)

_PRICING = {
    "claude-opus-4-6": {"input": 15.0 / 1e6, "output": 75.0 / 1e6},
    "claude-sonnet-4-5-20250929": {"input": 3.0 / 1e6, "output": 15.0 / 1e6},
    "claude-sonnet-4-20250514": {"input": 3.0 / 1e6, "output": 15.0 / 1e6},
    "claude-haiku-4-5-20251001": {"input": 0.80 / 1e6, "output": 4.0 / 1e6},
}
_DEFAULT_PRICING = {"input": 15.0 / 1e6, "output": 75.0 / 1e6}


class BudgetExhausted(Exception):
    pass


class LLM:
    def __init__(self, model: str = "claude-opus-4-6", max_cost: float = 50.0):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_cost = max_cost
        self.total_cost = 0.0

    @property
    def exhausted(self) -> bool:
        return self.total_cost >= self.max_cost

    def _track(self, usage) -> None:
        p = _PRICING.get(self.model, _DEFAULT_PRICING)
        cost = usage.input_tokens * p["input"] + usage.output_tokens * p["output"]
        self.total_cost += cost
        log.info(
            f"LLM: {usage.input_tokens}in/{usage.output_tokens}out "
            f"${cost:.4f} (total ${self.total_cost:.4f})"
        )

    def _check(self) -> None:
        if self.exhausted:
            raise BudgetExhausted(
                f"Budget exhausted: ${self.total_cost:.2f} >= ${self.max_cost:.2f}"
            )

    def __call__(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 16384,
    ) -> str:
        """Single-turn text completion."""
        self._check()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
        )
        self._track(resp.usage)
        return "".join(b.text for b in resp.content if b.type == "text")

    def structured(
        self,
        system: str,
        user: str,
        schema: dict,
        *,
        temperature: float = 0.7,
        max_tokens: int = 16384,
    ) -> dict:
        """Structured output via forced tool_use. Returns validated JSON."""
        self._check()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
            tools=[
                {
                    "name": "respond",
                    "description": "Respond with structured data",
                    "input_schema": schema,
                }
            ],
            tool_choice={"type": "tool", "name": "respond"},
        )
        self._track(resp.usage)
        for block in resp.content:
            if block.type == "tool_use":
                return block.input
        return {}

    def with_tools(
        self,
        system: str,
        user: str,
        tools: list[dict],
        *,
        temperature: float = 0.7,
        max_tokens: int = 16384,
    ) -> str:
        """Single API call with server-side tools (e.g. web search)."""
        self._check()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
            tools=tools,
        )
        self._track(resp.usage)
        return "".join(b.text for b in resp.content if b.type == "text")
