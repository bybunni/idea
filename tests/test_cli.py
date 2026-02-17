import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from idea.cli import _load_env, main
from idea.report import Report


class TestLoadEnv:
    def test_sets_env_vars(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("FOO=bar\nBAZ=qux\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("FOO", raising=False)
        monkeypatch.delenv("BAZ", raising=False)
        _load_env()
        assert os.environ["FOO"] == "bar"
        assert os.environ["BAZ"] == "qux"

    def test_skips_comments_and_blanks(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("# comment\n\nKEY=val\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("KEY", raising=False)
        _load_env()
        assert os.environ["KEY"] == "val"

    def test_strips_quotes(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text('QUOTED="hello world"\nSINGLE=\'value\'\n')
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("QUOTED", raising=False)
        monkeypatch.delenv("SINGLE", raising=False)
        _load_env()
        assert os.environ["QUOTED"] == "hello world"
        assert os.environ["SINGLE"] == "value"

    def test_equals_in_value(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("URL=https://example.com?a=1&b=2\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("URL", raising=False)
        _load_env()
        assert os.environ["URL"] == "https://example.com?a=1&b=2"

    def test_setdefault_no_override(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("EXISTING=new_value\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("EXISTING", "old_value")
        _load_env()
        assert os.environ["EXISTING"] == "old_value"

    def test_missing_env_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _load_env()  # should not raise


class TestMainNoArgs:
    def test_prints_help(self, capsys, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", ["idea"])
        with patch("idea.cli._load_env"):
            main()
        captured = capsys.readouterr()
        assert "IDEA" in captured.out or "usage" in captured.out.lower()


class TestMainReport:
    def test_report_subcommand(self, tmp_path, monkeypatch, capsys):
        # Create a valid report to load
        r = Report(question="test q", verdict="positive")
        r.save(tmp_path / "test_run")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", ["idea", "report", str(tmp_path / "test_run")])
        with patch("idea.cli._load_env"):
            main()
        captured = capsys.readouterr()
        assert "test q" in captured.out
