from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from idea.llm import LLM, BudgetExhausted
from helpers import make_response


class TestBudgetExhausted:
    def test_is_exception(self):
        assert issubclass(BudgetExhausted, Exception)

    def test_message(self):
        e = BudgetExhausted("over budget")
        assert str(e) == "over budget"


class TestLLMInit:
    @patch("idea.llm.anthropic.Anthropic")
    def test_defaults(self, mock_cls):
        llm = LLM()
        assert llm.model == "claude-opus-4-6"
        assert llm.max_cost == 50.0
        assert llm.total_cost == 0.0

    @patch("idea.llm.anthropic.Anthropic")
    def test_custom(self, mock_cls):
        llm = LLM(model="claude-sonnet-4-5-20250929", max_cost=10.0)
        assert llm.model == "claude-sonnet-4-5-20250929"
        assert llm.max_cost == 10.0


class TestExhausted:
    @patch("idea.llm.anthropic.Anthropic")
    def test_under_budget(self, mock_cls):
        llm = LLM(max_cost=10.0)
        llm.total_cost = 5.0
        assert not llm.exhausted

    @patch("idea.llm.anthropic.Anthropic")
    def test_at_budget(self, mock_cls):
        llm = LLM(max_cost=10.0)
        llm.total_cost = 10.0
        assert llm.exhausted

    @patch("idea.llm.anthropic.Anthropic")
    def test_over_budget(self, mock_cls):
        llm = LLM(max_cost=10.0)
        llm.total_cost = 15.0
        assert llm.exhausted


class TestCheck:
    @patch("idea.llm.anthropic.Anthropic")
    def test_raises_when_exhausted(self, mock_cls):
        llm = LLM(max_cost=1.0)
        llm.total_cost = 2.0
        with pytest.raises(BudgetExhausted):
            llm._check()

    @patch("idea.llm.anthropic.Anthropic")
    def test_no_raise_under_budget(self, mock_cls):
        llm = LLM(max_cost=10.0)
        llm._check()  # should not raise


class TestCall:
    @patch("idea.llm.anthropic.Anthropic")
    def test_returns_concatenated_text(self, mock_cls):
        llm = LLM()
        resp = make_response(content=[
            SimpleNamespace(type="text", text="hello "),
            SimpleNamespace(type="text", text="world"),
        ])
        llm.client.messages.create.return_value = resp
        result = llm("system", "user")
        assert result == "hello world"

    @patch("idea.llm.anthropic.Anthropic")
    def test_ignores_non_text_blocks(self, mock_cls):
        llm = LLM()
        resp = make_response(content=[
            SimpleNamespace(type="text", text="hello"),
            SimpleNamespace(type="tool_use", id="x", name="respond",
                            input={"key": "val"}),
        ])
        llm.client.messages.create.return_value = resp
        result = llm("system", "user")
        assert result == "hello"

    @patch("idea.llm.anthropic.Anthropic")
    def test_cost_accumulation(self, mock_cls):
        llm = LLM()
        resp = make_response(input_tokens=1000, output_tokens=100)
        llm.client.messages.create.return_value = resp
        llm("sys", "user1")
        cost1 = llm.total_cost
        assert cost1 > 0
        llm("sys", "user2")
        assert llm.total_cost == pytest.approx(cost1 * 2)


class TestStructured:
    @patch("idea.llm.anthropic.Anthropic")
    def test_returns_tool_input(self, mock_cls):
        llm = LLM()
        resp = make_response(content=[
            SimpleNamespace(type="tool_use", id="x", name="respond",
                            input={"verdict": "positive"}),
        ])
        llm.client.messages.create.return_value = resp
        result = llm.structured("sys", "user", {"type": "object"})
        assert result == {"verdict": "positive"}

    @patch("idea.llm.anthropic.Anthropic")
    def test_returns_empty_when_no_tool_use(self, mock_cls):
        llm = LLM()
        resp = make_response(content=[
            SimpleNamespace(type="text", text="no tool"),
        ])
        llm.client.messages.create.return_value = resp
        result = llm.structured("sys", "user", {"type": "object"})
        assert result == {}

    @patch("idea.llm.anthropic.Anthropic")
    def test_passes_tools_and_tool_choice(self, mock_cls):
        llm = LLM()
        resp = make_response(content=[
            SimpleNamespace(type="tool_use", id="x", name="respond", input={}),
        ])
        llm.client.messages.create.return_value = resp
        llm.structured("sys", "user", {"type": "object"})
        call_kw = llm.client.messages.create.call_args
        assert call_kw.kwargs["tool_choice"] == {"type": "tool", "name": "respond"}


class TestWithTools:
    @patch("idea.llm.anthropic.Anthropic")
    def test_returns_concatenated_text(self, mock_cls):
        llm = LLM()
        resp = make_response(content=[
            SimpleNamespace(type="text", text="search "),
            SimpleNamespace(type="text", text="results"),
        ])
        llm.client.messages.create.return_value = resp
        result = llm.with_tools("sys", "user", [{"type": "web_search"}])
        assert result == "search results"
