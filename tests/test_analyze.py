from idea.analyze import analyze, synthesize, SYSTEM, SYNTH_SYSTEM, SCHEMA
from helpers import FakeLLM


class TestAnalyze:
    def test_passes_correct_args(self):
        llm = FakeLLM(structured_return={
            "verdict": "positive", "analysis": "good",
            "headline": "hl", "next_steps": [],
        })
        result = analyze(llm, "hyp block", {"baseline_success": True})
        assert result["verdict"] == "positive"
        call = llm.calls[0]
        assert call[0] == "structured"
        assert call[1] == SYSTEM
        assert "hyp block" in call[2]
        assert "baseline_success" in call[2]

    def test_includes_context(self):
        llm = FakeLLM(structured_return={
            "verdict": "inconclusive", "analysis": "",
            "headline": "", "next_steps": [],
        })
        analyze(llm, "hyp", {}, context="prior context")
        call = llm.calls[0]
        assert "prior context" in call[2]
        assert "## Context" in call[2]

    def test_no_context_section_when_empty(self):
        llm = FakeLLM(structured_return={
            "verdict": "negative", "analysis": "",
            "headline": "", "next_steps": [],
        })
        analyze(llm, "hyp", {}, context="")
        call = llm.calls[0]
        assert "## Context" not in call[2]


class TestSynthesize:
    def test_passes_correct_args(self):
        llm = FakeLLM(structured_return={
            "verdict": "positive", "analysis": "synth",
            "headline": "combined", "next_steps": [],
        })
        parent = {"verdict": "inconclusive", "analysis": "parent analysis"}
        children = [{"question": "c1", "verdict": "positive"}]
        result = synthesize(llm, parent, children)
        assert result["verdict"] == "positive"
        call = llm.calls[0]
        assert call[0] == "structured"
        assert call[1] == SYNTH_SYSTEM
        assert "parent analysis" in call[2]
        assert "c1" in call[2]
