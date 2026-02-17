from unittest.mock import patch

from idea.experiment import _generate_and_run, experiment
from idea.sandbox import ExecResult
from helpers import FakeLLM


class TestGenerateAndRun:
    @patch("idea.experiment.run_code")
    def test_success_no_retry(self, mock_run, tmp_path):
        mock_run.return_value = ExecResult(0, '{"acc":0.9}\n', "", {"acc": 0.9})
        llm = FakeLLM(structured_return={"code": "print(1)", "description": "test"})
        desc, result = _generate_and_run(llm, "prompt", tmp_path)
        assert result.success
        assert desc == "test"
        assert len(llm.calls) == 1  # no retry

    @patch("idea.experiment.run_code")
    def test_fail_then_succeed(self, mock_run, tmp_path):
        mock_run.side_effect = [
            ExecResult(1, "", "error msg", None),
            ExecResult(0, '{"ok":1}\n', "", {"ok": 1}),
        ]
        llm = FakeLLM(structured_return={"code": "print(1)", "description": "fixed"})
        desc, result = _generate_and_run(llm, "prompt", tmp_path)
        assert result.success
        assert len(llm.calls) == 2  # original + retry

    @patch("idea.experiment.run_code")
    def test_fail_twice(self, mock_run, tmp_path):
        mock_run.side_effect = [
            ExecResult(1, "", "err1", None),
            ExecResult(1, "", "err2", None),
        ]
        llm = FakeLLM(structured_return={"code": "bad", "description": "broken"})
        desc, result = _generate_and_run(llm, "prompt", tmp_path)
        assert not result.success
        assert len(llm.calls) == 2


class TestExperiment:
    @patch("idea.experiment.run_code")
    def test_returns_all_keys(self, mock_run, tmp_path):
        mock_run.return_value = ExecResult(0, '{"acc":0.9}\n', "", {"acc": 0.9})
        llm = FakeLLM(structured_return={
            "code": "print(1)", "description": "desc",
        })
        result = experiment(llm, "hyp block", {"prior_work": "x"}, tmp_path)
        assert "baseline_description" in result
        assert "baseline_metrics" in result
        assert "baseline_success" in result
        assert "idea_description" in result
        assert "idea_metrics" in result
        assert "idea_success" in result

    @patch("idea.experiment.run_code")
    def test_baseline_and_idea_both_run(self, mock_run, tmp_path):
        mock_run.return_value = ExecResult(0, '{"x":1}\n', "", {"x": 1})
        llm = FakeLLM(structured_return={"code": "c", "description": "d"})
        result = experiment(llm, "hyp", {}, tmp_path)
        assert result["baseline_success"] is True
        assert result["idea_success"] is True
        # structured called twice (baseline + idea), each with run_code
        assert len(llm.calls) == 2
