import json

import pytest

from idea.report import Report


class TestDefaults:
    def test_default_fields(self):
        r = Report(question="test q")
        assert r.question == "test q"
        assert r.hypothesis == ""
        assert r.lit_review == ""
        assert r.experiment == {}
        assert r.analysis == ""
        assert r.verdict == ""
        assert r.headline == ""
        assert r.next_steps == []
        assert r.children == []
        assert r.cost == 0.0
        assert r.depth == 0
        assert isinstance(r.timestamp, str)


class TestSaveLoad:
    def test_round_trip_flat(self, tmp_path):
        r = Report(question="q1", hypothesis="h1", verdict="positive",
                    cost=1.23, depth=0)
        r.save(tmp_path / "run1")
        loaded = Report.load(tmp_path / "run1")
        assert loaded.question == "q1"
        assert loaded.hypothesis == "h1"
        assert loaded.verdict == "positive"
        assert loaded.cost == 1.23

    def test_round_trip_with_children(self, tmp_path):
        child = Report(question="child q", verdict="negative", depth=1)
        parent = Report(question="parent q", children=[child], depth=0)
        parent.save(tmp_path / "nested")
        loaded = Report.load(tmp_path / "nested")
        assert len(loaded.children) == 1
        assert loaded.children[0].question == "child q"
        assert loaded.children[0].depth == 1

    def test_load_from_file_path(self, tmp_path):
        r = Report(question="fp")
        r.save(tmp_path / "fp_test")
        loaded = Report.load(tmp_path / "fp_test" / "report.json")
        assert loaded.question == "fp"

    def test_load_from_dir_path(self, tmp_path):
        r = Report(question="dp")
        r.save(tmp_path / "dir_test")
        loaded = Report.load(tmp_path / "dir_test")
        assert loaded.question == "dp"

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Report.load(tmp_path / "nonexistent")

    def test_save_creates_directory(self, tmp_path):
        r = Report(question="mkdir")
        path = tmp_path / "a" / "b" / "c"
        r.save(path)
        assert (path / "report.json").exists()


class TestFromDict:
    def test_with_children(self):
        d = {
            "question": "p", "hypothesis": "", "lit_review": "",
            "experiment": {}, "analysis": "", "verdict": "",
            "headline": "", "next_steps": [], "cost": 0.0,
            "depth": 0, "timestamp": "t",
            "children": [
                {"question": "c1", "hypothesis": "", "lit_review": "",
                 "experiment": {}, "analysis": "", "verdict": "",
                 "headline": "", "next_steps": [], "cost": 0.0,
                 "depth": 1, "timestamp": "t"},
            ],
        }
        r = Report._from_dict(d)
        assert len(r.children) == 1
        assert r.children[0].question == "c1"

    def test_without_children_key(self):
        d = {
            "question": "q", "hypothesis": "", "lit_review": "",
            "experiment": {}, "analysis": "", "verdict": "",
            "headline": "", "next_steps": [], "cost": 0.0,
            "depth": 0, "timestamp": "t",
        }
        r = Report._from_dict(d)
        assert r.children == []


class TestContextSummary:
    def test_basic(self):
        r = Report(question="q", hypothesis="h", verdict="v", headline="hl")
        s = r.context_summary()
        assert "Prior finding (depth 0): q" in s
        assert "Hypothesis: h" in s
        assert "Verdict: v" in s
        assert "Headline: hl" in s

    def test_includes_experiment(self):
        r = Report(question="q", experiment={"k": "v"})
        s = r.context_summary()
        assert "Experiment:" in s

    def test_omits_empty_fields(self):
        r = Report(question="q")
        s = r.context_summary()
        assert "Hypothesis:" not in s
        assert "Verdict:" not in s


class TestStr:
    def test_basic_output(self):
        r = Report(question="q", verdict="positive", hypothesis="h",
                    lit_review="lit", analysis="analysis text")
        s = str(r)
        assert "IDEA Report: q" in s
        assert "Verdict: positive" in s
        assert "Hypothesis: h" in s
        assert "Literature Review:" in s
        assert "Analysis:" in s

    def test_no_experiment(self):
        r = Report(question="q")
        s = str(r)
        assert "(none)" in s

    def test_with_experiment(self):
        r = Report(question="q", experiment={"acc": 0.9})
        s = str(r)
        assert "0.9" in s

    def test_with_next_steps(self):
        r = Report(question="q", next_steps=["step1", "step2"])
        s = str(r)
        assert "Next Steps:" in s
        assert "1. step1" in s
        assert "2. step2" in s

    def test_with_children(self):
        child = Report(question="child", depth=1)
        r = Report(question="parent", children=[child])
        s = str(r)
        assert "Child (depth 1)" in s
        assert "child" in s
