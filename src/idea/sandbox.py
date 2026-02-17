import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str
    metrics: dict  # None if no JSON found

    @property
    def success(self):
        return self.returncode == 0


def run_code(code, workdir, timeout=300):
    workdir.mkdir(parents=True, exist_ok=True)
    script = workdir / "experiment.py"
    script.write_text(code)
    try:
        proc = subprocess.run(
            ["python3", str(script)],
            capture_output=True, text=True, timeout=timeout, cwd=str(workdir),
        )
        return ExecResult(proc.returncode, proc.stdout, proc.stderr,
                          _extract_metrics(proc.stdout))
    except subprocess.TimeoutExpired:
        return ExecResult(-1, "", f"Timed out after {timeout}s", None)
    except Exception as e:
        return ExecResult(-1, "", str(e), None)


def _extract_metrics(stdout):
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None
