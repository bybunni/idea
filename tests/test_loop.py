from unittest.mock import patch, MagicMock

import pytest

from idea.llm import BudgetExhausted
from idea.loop import idea_loop
from idea.report import Report
from helpers import FakeLLM


@pytest.fixture
def mock_phases():
    """Patch all four phases + format_hypothesis."""
    with (
        patch("idea.loop.hypothesize") as m_hyp,
        patch("idea.loop.format_hypothesis") as m_fmt,
        patch("idea.loop.research") as m_res,
        patch("idea.loop.experiment") as m_exp,
        patch("idea.loop.analyze") as m_ana,
        patch("idea.loop.synthesize") as m_syn,
    ):
        m_hyp.return_value = {"hypothesis": "test hyp"}
        m_fmt.return_value = "formatted hyp"
        m_res.return_value = {"prior_work": "lit"}
        m_exp.return_value = {
            "baseline_description": "b", "baseline_metrics": {},
            "baseline_success": True, "idea_description": "i",
            "idea_metrics": {}, "idea_success": True,
        }
        m_ana.return_value = {
            "verdict": "positive", "analysis": "good result",
            "headline": "it works", "next_steps": [],
        }
        m_syn.return_value = {
            "verdict": "positive", "analysis": "synth",
            "headline": "combined", "next_steps": [],
        }
        yield {
            "hypothesize": m_hyp,
            "format_hypothesis": m_fmt,
            "research": m_res,
            "experiment": m_exp,
            "analyze": m_ana,
            "synthesize": m_syn,
        }


class TestHappyPath:
    def test_all_phases_run(self, tmp_path, mock_phases):
        llm = FakeLLM()
        report = idea_loop("seed q", tmp_path / "run", llm)
        assert report.question == "seed q"
        assert report.hypothesis == "test hyp"
        assert report.verdict == "positive"
        assert report.headline == "it works"
        assert report.analysis == "good result"
        mock_phases["hypothesize"].assert_called_once()
        mock_phases["research"].assert_called_once()
        mock_phases["experiment"].assert_called_once()
        mock_phases["analyze"].assert_called_once()

    def test_report_saved(self, tmp_path, mock_phases):
        llm = FakeLLM()
        idea_loop("seed", tmp_path / "saved", llm)
        assert (tmp_path / "saved" / "report.json").exists()

    def test_cost_recorded(self, tmp_path, mock_phases):
        llm = FakeLLM()
        llm.total_cost = 1.5
        report = idea_loop("seed", tmp_path / "cost", llm)
        assert report.cost == 1.5


class TestBudgetExhausted:
    def test_verdict_budget_exceeded(self, tmp_path, mock_phases):
        mock_phases["hypothesize"].side_effect = BudgetExhausted("over")
        llm = FakeLLM()
        report = idea_loop("seed", tmp_path / "budget", llm)
        assert report.verdict == "budget_exceeded"
        assert (tmp_path / "budget" / "report.json").exists()


class TestGenericException:
    def test_verdict_error(self, tmp_path, mock_phases):
        mock_phases["research"].side_effect = RuntimeError("api down")
        llm = FakeLLM()
        report = idea_loop("seed", tmp_path / "err", llm)
        assert report.verdict == "error"
        assert "api down" in report.analysis
        assert (tmp_path / "err" / "report.json").exists()


class TestRecursion:
    def test_spawns_child_loops(self, tmp_path, mock_phases):
        call_count = 0

        def analyze_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Parent: returns next_steps to trigger recursion
                return {
                    "verdict": "inconclusive", "analysis": "needs more",
                    "headline": "unclear", "next_steps": ["follow up 1", "follow up 2"],
                }
            # Children: no further recursion
            return {
                "verdict": "positive", "analysis": "done",
                "headline": "resolved", "next_steps": [],
            }

        mock_phases["analyze"].side_effect = analyze_side_effect
        llm = FakeLLM()
        report = idea_loop("seed", tmp_path / "recurse", llm,
                           max_depth=2, max_branches=3)
        assert len(report.children) == 2
        assert report.children[0].question == "follow up 1"
        assert report.children[1].question == "follow up 2"
        # synthesize should be called once (for the parent with its children)
        mock_phases["synthesize"].assert_called_once()

    def test_no_recursion_at_max_depth(self, tmp_path, mock_phases):
        mock_phases["analyze"].return_value = {
            "verdict": "inconclusive", "analysis": "",
            "headline": "", "next_steps": ["step"],
        }
        llm = FakeLLM()
        report = idea_loop("seed", tmp_path / "maxd", llm,
                           depth=2, max_depth=2)
        assert len(report.children) == 0

    def test_no_recursion_empty_next_steps(self, tmp_path, mock_phases):
        mock_phases["analyze"].return_value = {
            "verdict": "positive", "analysis": "",
            "headline": "", "next_steps": [],
        }
        llm = FakeLLM()
        report = idea_loop("seed", tmp_path / "empty", llm)
        assert len(report.children) == 0

    def test_exhausted_stops_recursion(self, tmp_path, mock_phases):
        mock_phases["analyze"].return_value = {
            "verdict": "inconclusive", "analysis": "",
            "headline": "", "next_steps": ["a", "b"],
        }
        llm = FakeLLM(max_cost=0.0)
        llm.total_cost = 0.0  # exhausted = total >= max, so 0 >= 0 = True
        report = idea_loop("seed", tmp_path / "exhaust", llm)
        # exhausted before recursion starts
        assert len(report.children) == 0

    def test_max_branches_limits_children(self, tmp_path, mock_phases):
        mock_phases["analyze"].return_value = {
            "verdict": "inconclusive", "analysis": "",
            "headline": "", "next_steps": ["a", "b", "c", "d"],
        }
        llm = FakeLLM()
        report = idea_loop("seed", tmp_path / "branch", llm,
                           max_depth=2, max_branches=2)
        assert len(report.children) == 2

    def test_exhausted_mid_recursion(self, tmp_path, mock_phases):
        mock_phases["analyze"].return_value = {
            "verdict": "inconclusive", "analysis": "",
            "headline": "", "next_steps": ["a", "b", "c"],
        }
        call_count = 0

        class ExhaustAfterOneLLM(FakeLLM):
            """Becomes exhausted after the first child loop completes."""
            @property
            def exhausted(self):
                # Count how many times analyze is called to track child loops
                return self._exhausted_flag

            def __init__(self):
                super().__init__()
                self._exhausted_flag = False

        llm = ExhaustAfterOneLLM()
        # We'll make the first child loop succeed, then exhaust the budget
        original_analyze = mock_phases["analyze"].return_value

        def side_effect_for_children(*args, **kwargs):
            # After being called once for parent, once for first child
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                llm._exhausted_flag = True
            return original_analyze

        mock_phases["analyze"].side_effect = side_effect_for_children
        report = idea_loop("seed", tmp_path / "midex", llm, max_depth=2)
        # Should have at most 1-2 children (stops when exhausted)
        assert len(report.children) <= 2
