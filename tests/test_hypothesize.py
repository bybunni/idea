from idea.hypothesize import hypothesize, format_hypothesis, SYSTEM, SCHEMA
from helpers import FakeLLM


class TestFormatHypothesis:
    def test_full_dict(self):
        h = {
            "hypothesis": "Test hyp",
            "expected_positive": "good",
            "expected_negative": "bad",
            "key_metrics": ["acc", "f1"],
            "scope_notes": "small scale",
        }
        result = format_hypothesis(h)
        assert "## Hypothesis" in result
        assert "Test hyp" in result
        assert "Expected positive: good" in result
        assert "Expected negative: bad" in result
        assert "acc, f1" in result
        assert "Scope: small scale" in result

    def test_missing_optional_keys(self):
        h = {
            "hypothesis": "Test",
            "expected_positive": "pos",
            "expected_negative": "neg",
        }
        result = format_hypothesis(h)
        assert "Test" in result
        assert "Key metrics: " in result  # empty join
        assert "Scope: " in result

    def test_empty_metrics(self):
        h = {
            "hypothesis": "h",
            "expected_positive": "p",
            "expected_negative": "n",
            "key_metrics": [],
            "scope_notes": "s",
        }
        result = format_hypothesis(h)
        assert "Key metrics: \n" in result


class TestHypothesize:
    def test_passes_correct_args(self):
        llm = FakeLLM(structured_return={
            "hypothesis": "refined hyp",
            "expected_positive": "p",
            "expected_negative": "n",
            "key_metrics": [],
            "scope_notes": "",
        })
        result = hypothesize(llm, "seed question")
        assert result["hypothesis"] == "refined hyp"
        call = llm.calls[0]
        assert call[0] == "structured"
        assert call[1] == SYSTEM
        assert "seed question" in call[2]
        assert call[3] == SCHEMA

    def test_sets_default_hypothesis(self):
        llm = FakeLLM(structured_return={})
        result = hypothesize(llm, "my question")
        assert result["hypothesis"] == "my question"

    def test_appends_context(self):
        llm = FakeLLM(structured_return={"hypothesis": "h"})
        hypothesize(llm, "seed", context="prior findings")
        call = llm.calls[0]
        assert "prior findings" in call[2]

    def test_no_context_in_user_when_empty(self):
        llm = FakeLLM(structured_return={"hypothesis": "h"})
        hypothesize(llm, "seed", context="")
        call = llm.calls[0]
        assert "Context from prior" not in call[2]
