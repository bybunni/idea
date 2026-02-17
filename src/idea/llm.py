import anthropic


class BudgetExhausted(Exception):
    pass


class LLM:
    def __init__(self, model="claude-opus-4-6", max_cost=50.0):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_cost = max_cost
        self.total_cost = 0.0

    @property
    def exhausted(self):
        return self.total_cost >= self.max_cost

    def _check(self):
        if self.exhausted:
            raise BudgetExhausted(f"${self.total_cost:.2f} >= ${self.max_cost:.2f}")

    def _call(self, system, user, temperature=0.7, max_tokens=16384, **kw):
        self._check()
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature, **kw,
        )
        cost = resp.usage.input_tokens * 15e-6 + resp.usage.output_tokens * 75e-6
        self.total_cost += cost
        print(f"  llm: {resp.usage.input_tokens}in/{resp.usage.output_tokens}out "
              f"${cost:.4f} (${self.total_cost:.4f})")
        return resp

    def __call__(self, system, user, *, temperature=0.7, max_tokens=16384):
        resp = self._call(system, user, temperature, max_tokens)
        return "".join(b.text for b in resp.content if b.type == "text")

    def structured(self, system, user, schema, *, temperature=0.7, max_tokens=16384):
        resp = self._call(system, user, temperature, max_tokens,
            tools=[{"name": "respond", "description": "Respond with structured data",
                    "input_schema": schema}],
            tool_choice={"type": "tool", "name": "respond"},
        )
        for b in resp.content:
            if b.type == "tool_use":
                return b.input
        return {}

    def with_tools(self, system, user, tools, *, temperature=0.7, max_tokens=16384):
        resp = self._call(system, user, temperature, max_tokens, tools=tools)
        return "".join(b.text for b in resp.content if b.type == "text")
