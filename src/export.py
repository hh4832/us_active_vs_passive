from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


OUTPUT_DIRS = ["summary", "daily", "trades", "risk", "benchmarks", "figures", "metadata"]


def create_run_directory(base: str | Path = "outputs") -> Path:
    run = Path(base) / datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    for name in OUTPUT_DIRS: (run / name).mkdir(parents=True, exist_ok=True)
    return run


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def write_json(path: str | Path, value: object) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def base_run_info(transaction_file: str | Path, **extra) -> dict:
    return {"timestamp": datetime.now(timezone.utc).isoformat(), "git_commit_sha": git_sha(),
            "input_transaction_filename": Path(transaction_file).name, "input_transaction_hash": sha256_file(transaction_file),
            "software_versions": {"python": platform.python_version(), "pandas": pd.__version__}, **extra}

