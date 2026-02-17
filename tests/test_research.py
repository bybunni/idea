from idea.research import research, SYSTEM, SCHEMA, WEB_SEARCH_TOOL
from helpers import FakeLLM


class TestResearchWebSearch:
    def test_web_search_enabled(self):
        llm = FakeLLM(
            with_tools_return="web results here",
            structured_return={"prior_work": "found stuff"},
        )
        result = research(llm, "hyp block", web_search=True)
        assert result == {"prior_work": "found stuff"}
        # with_tools called first, then structured
        assert llm.calls[0][0] == "with_tools"
        assert llm.calls[1][0] == "structured"

    def test_web_search_context_appended(self):
        llm = FakeLLM(
            with_tools_return="web findings",
            structured_return={},
        )
        research(llm, "hyp block", web_search=True)
        structured_call = llm.calls[1]
        assert "web findings" in structured_call[2]

    def test_web_search_disabled(self):
        llm = FakeLLM(structured_return={"prior_work": "no web"})
        result = research(llm, "hyp block", web_search=False)
        assert result == {"prior_work": "no web"}
        assert len(llm.calls) == 1
        assert llm.calls[0][0] == "structured"

    def test_web_search_exception_caught(self):
        class FailToolsLLM(FakeLLM):
            def with_tools(self, system, user, tools, **kw):
                self.calls.append(("with_tools", system, user, tools))
                raise RuntimeError("network error")

        llm = FailToolsLLM(structured_return={"prior_work": "fallback"})
        result = research(llm, "hyp block", web_search=True)
        assert result == {"prior_work": "fallback"}
        # structured still called despite with_tools failure
        assert llm.calls[1][0] == "structured"

    def test_empty_web_search_not_appended(self):
        llm = FakeLLM(
            with_tools_return="",
            structured_return={},
        )
        research(llm, "hyp block", web_search=True)
        structured_call = llm.calls[1]
        assert "Web search findings" not in structured_call[2]

    def test_context_appended(self):
        llm = FakeLLM(structured_return={})
        research(llm, "hyp block", context="prior context", web_search=False)
        call = llm.calls[0]
        assert "prior context" in call[2]
