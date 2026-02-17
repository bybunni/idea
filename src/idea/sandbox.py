"""Sandbox — subprocess-based code execution with timeout."""

from __future__ import annotations

import subprocess
import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 300  # 5 minutes


@dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str
    metrics: dict | None  # parsed from last line if JSON
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def summary(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        if self.timed_out:
            status = "TIMEOUT"
        parts = [f"[{status}] exit={self.returncode}"]
        if self.metrics:
            parts.append(f"metrics={json.dumps(self.metrics)}")
        if self.stderr and not self.success:
            # truncate stderr for readability
            err = self.stderr[-2000:] if len(self.stderr) > 2000 else self.stderr
            parts.append(f"stderr:\n{err}")
        return "\n".join(parts)


def run_code(
    code: str,
    workspace: Path,
    filename: str = "experiment.py",
    timeout: int = DEFAULT_TIMEOUT,
    venv: Path | None = None,
) -> ExecResult:
    """Write code to file and execute it, capturing metrics from stdout."""
    code_dir = workspace / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    script = code_dir / filename
    script.write_text(code)

    python = str(venv / "bin" / "python") if venv else "python3"
    cmd = [python, str(script)]

    log.info(f"Running {script} (timeout={timeout}s)")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(code_dir),
        )
        metrics = _extract_metrics(proc.stdout)
        return ExecResult(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            metrics=metrics,
        )
    except subprocess.TimeoutExpired:
        return ExecResult(
            returncode=-1,
            stdout="",
            stderr=f"Process timed out after {timeout}s",
            metrics=None,
            timed_out=True,
        )
    except Exception as e:
        return ExecResult(
            returncode=-1,
            stdout="",
            stderr=str(e),
            metrics=None,
        )


def install_deps(deps: list[str], workspace: Path) -> bool:
    """Install Python dependencies. Returns True on success."""
    if not deps:
        return True
    cmd = ["pip", "install", "--break-system-packages", "-q"] + deps
    log.info(f"Installing: {deps}")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, cwd=str(workspace)
        )
        return proc.returncode == 0
    except Exception as e:
        log.error(f"Failed to install deps: {e}")
        return False


def _extract_metrics(stdout: str) -> dict | None:
    """Try to parse the last line of stdout as JSON metrics."""
    if not stdout.strip():
        return None
    lines = stdout.strip().split("\n")
    # search from end for a JSON line
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None
