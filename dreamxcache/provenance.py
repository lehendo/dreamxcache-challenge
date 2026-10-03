"""Run provenance: every result artifact records the code version it came from.

Results are append-only: a new run always gets a new timestamped directory
(`config.timestamped_results_subdir`), and its ``report.json`` records the git
SHA, a UTC timestamp and the exact parameters, so a number can always be traced
to the code and settings that produced it.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dreamxcache.config import timestamped_results_subdir


def git_sha() -> str | None:
    """The current commit SHA, or None outside a git checkout."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_report(label: str, report: dict[str, Any]) -> Path:
    """Write ``report.json`` (with provenance fields added) into a new
    timestamped results directory and return the file's path."""
    out_dir = timestamped_results_subdir(label)
    full = {"run_at_utc": datetime.now(UTC).isoformat(), "git_sha": git_sha(), **report}
    path = out_dir / "report.json"
    path.write_text(json.dumps(full, indent=2, default=str))
    print(f"Wrote {path}")
    return path
