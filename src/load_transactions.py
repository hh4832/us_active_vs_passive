from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load a Firstrade export without silently changing its contents."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Transaction file not found: {path}")
    if path.suffix.lower() == ".csv":
        errors: list[str] = []
        for encoding in ("utf-8-sig", "utf-8", "big5", "cp1252"):
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError as exc:
                errors.append(f"{encoding}: {exc}")
        raise UnicodeError("Unable to decode CSV; " + " | ".join(errors))
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported transaction format: {path.suffix}")

