"""Thin wrapper around the Anthropic SDK with budget tracking."""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from .budget import Budget

log = logging.getLogger(__name__)

# rough per-token pricing (USD) — update as needed
_PRICING = {
    "claude-sonnet-4-5-20250514": {"input": 3.0 / 1e6, "output": 15.0 / 1e6},
    "claude-sonnet-4-20250514": {"input": 3.0 / 1e6, "output": 15.0 / 1e6},
    "claude-haiku-4-5-20251001": {"input": 0.80 / 1e6, "output": 4.0 / 1e6},
}

DEFAULT_MODEL = "claude-sonnet-4-5-20250514"


def _estimate_cost(model: str, usage: Any) -> float:
    p = _PRICING.get(model, {"input": 3.0 / 1e6, "output": 15.0 / 1e6})
    return usage.input_tokens * p["input"] + usage.output_tokens * p["output"]


class LLM:
    def __init__(self, model: str = DEFAULT_MODEL, budget: Budget | None = None):
        self.client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env
        self.model = model
        self.budget = budget

    def __call__(
        self,
        system: str,
        user: str,
        *,
        tools: list[dict] | None = None,
        max_tokens: int = 16384,
        temperature: float = 0.7,
        parse_json: bool = False,
    ) -> str | dict:
        """Single-turn completion. Returns text or parsed JSON."""
        if self.budget and self.budget.exhausted():
            raise BudgetExhausted("Budget exhausted before API call")

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools

        resp = self.client.messages.create(**kwargs)
        cost = _estimate_cost(self.model, resp.usage)

        if self.budget:
            self.budget.tick(cost)

        log.info(
            f"LLM call: {resp.usage.input_tokens}in/{resp.usage.output_tokens}out "
            f"${cost:.4f} (total ${self.budget.cost_usd:.4f})"
            if self.budget
            else f"LLM call: {resp.usage.input_tokens}in/{resp.usage.output_tokens}out ${cost:.4f}"
        )

        # extract text from response
        text = ""
        for block in resp.content:
            if block.type == "text":
                text += block.text

        if parse_json:
            return _extract_json(text)
        return text

    def with_tools(
        self,
        system: str,
        user: str,
        tools: list[dict],
        *,
        max_tokens: int = 16384,
        temperature: float = 0.7,
        max_rounds: int = 10,
    ) -> str:
        """Multi-turn tool-use loop. Runs until the model stops calling tools."""
        if self.budget and self.budget.exhausted():
            raise BudgetExhausted("Budget exhausted before API call")

        messages = [{"role": "user", "content": user}]

        for _ in range(max_rounds):
            if self.budget and self.budget.exhausted():
                break

            resp = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                tools=tools,
                temperature=temperature,
            )
            cost = _estimate_cost(self.model, resp.usage)
            if self.budget:
                self.budget.tick(cost)

            # check if model wants to use tools
            tool_calls = [b for b in resp.content if b.type == "tool_use"]
            if not tool_calls:
                # done — extract final text
                return "".join(b.text for b in resp.content if b.type == "text")

            # append assistant response + tool results
            messages.append({"role": "assistant", "content": resp.content})

            tool_results = []
            for tc in tool_calls:
                result = self._execute_tool(tc.name, tc.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": result,
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        # fell through — return whatever text we have
        return "".join(
            b.text for b in resp.content if hasattr(b, "text") and b.type == "text"
        )

    def _execute_tool(self, name: str, input_data: dict) -> str:
        """Dispatch tool calls. Override or extend for custom tools."""
        # built-in tools are handled server-side by Anthropic
        # for custom tools (sandbox execution), subclasses override this
        return json.dumps({"error": f"Unknown tool: {name}"})


def _extract_json(text: str) -> dict:
    """Pull JSON from LLM output, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        # strip ```json ... ```
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        return {"_raw": text, "_parse_error": True}


class BudgetExhausted(Exception):
    pass
