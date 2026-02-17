import json
from unittest.mock import patch, MagicMock

import pytest

from idea.sandbox import ExecResult, run_code, _extract_metrics


class TestExecResult:
    def test_success_true(self):
        r = ExecResult(returncode=0, stdout="ok", stderr="", metrics={})
        assert r.success is True

    def test_success_false_nonzero(self):
        r = ExecResult(returncode=1, stdout="", stderr="error", metrics=None)
        assert r.success is False

    def test_success_false_negative(self):
        r = ExecResult(returncode=-1, stdout="", stderr="timeout", metrics=None)
        assert r.success is False


class TestExtractMetrics:
    def test_valid_json_last_line(self):
        stdout = "some output\n{\"accuracy\": 0.95}\n"
        assert _extract_metrics(stdout) == {"accuracy": 0.95}

    def test_no_json(self):
        assert _extract_metrics("just text\nno json here") is None

    def test_empty_stdout(self):
        assert _extract_metrics("") is None

    def test_json_not_on_last_line(self):
        stdout = '{\"a\": 1}\nsome trailing text'
        assert _extract_metrics(stdout) == {"a": 1}

    def test_invalid_json_brace(self):
        stdout = "{not valid json}"
        assert _extract_metrics(stdout) is None

    def test_multiple_json_lines(self):
        stdout = '{\"first\": 1}\n{\"second\": 2}\n'
        # reversed iteration finds the last one first
        assert _extract_metrics(stdout) == {"second": 2}

    def test_whitespace_around_json(self):
        stdout = "  \n  {\"x\": 42}  \n  "
        assert _extract_metrics(stdout) == {"x": 42}


class TestRunCode:
    def test_success_path(self, tmp_path):
        code = 'import json; print(json.dumps({"result": 1}))'
        result = run_code(code, tmp_path / "test_run")
        assert result.success
        assert result.metrics == {"result": 1}
        assert (tmp_path / "test_run" / "experiment.py").exists()

    def test_creates_workdir(self, tmp_path):
        wd = tmp_path / "a" / "b" / "c"
        run_code("print('hi')", wd)
        assert wd.exists()
        assert (wd / "experiment.py").read_text() == "print('hi')"

    def test_nonzero_returncode(self, tmp_path):
        result = run_code("import sys; sys.exit(1)", tmp_path / "fail")
        assert not result.success
        assert result.returncode == 1

    def test_syntax_error(self, tmp_path):
        result = run_code("def broken(", tmp_path / "syntax")
        assert not result.success
        assert "SyntaxError" in result.stderr

    def test_timeout(self, tmp_path):
        result = run_code("import time; time.sleep(10)", tmp_path / "slow", timeout=1)
        assert not result.success
        assert result.returncode == -1
        assert "Timed out" in result.stderr

    @patch("idea.sandbox.subprocess.run", side_effect=OSError("no python"))
    def test_generic_exception(self, mock_run, tmp_path):
        result = run_code("print(1)", tmp_path / "err")
        assert not result.success
        assert "no python" in result.stderr
        assert result.metrics is None
