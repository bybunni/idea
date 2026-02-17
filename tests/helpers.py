from types import SimpleNamespace


class FakeLLM:
    """Configurable mock LLM for testing."""

    def __init__(self, *, call_return="response text", structured_return=None,
                 with_tools_return="tools text", max_cost=50.0):
        self.model = "test-model"
        self.max_cost = max_cost
        self.total_cost = 0.0
        self._call_return = call_return
        self._structured_return = structured_return or {}
        self._with_tools_return = with_tools_return
        self.calls = []

    @property
    def exhausted(self):
        return self.total_cost >= self.max_cost

    def __call__(self, system, user, *, temperature=0.7, max_tokens=16384):
        self.calls.append(("__call__", system, user))
        return self._call_return

    def structured(self, system, user, schema, *, temperature=0.7, max_tokens=16384):
        self.calls.append(("structured", system, user, schema))
        return self._structured_return.copy() if isinstance(self._structured_return, dict) else self._structured_return

    def with_tools(self, system, user, tools, *, temperature=0.7, max_tokens=16384):
        self.calls.append(("with_tools", system, user, tools))
        return self._with_tools_return


def make_response(input_tokens=100, output_tokens=50, content=None):
    """Build a fake Anthropic response object."""
    if content is None:
        content = [SimpleNamespace(type="text", text="hello")]
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(usage=usage, content=content)
