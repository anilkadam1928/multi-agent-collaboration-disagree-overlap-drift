# data_loader.py
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


DEFAULT_PATHS = (
    Path("data/german_credit_data.csv"),
    Path("german_credit_data.csv"),
)


def _resolve_dataset_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)

    for candidate in DEFAULT_PATHS:
        if candidate.exists():
            return candidate

    return DEFAULT_PATHS[0]


def load_dataset(path: str | Path | None = None) -> list[dict[str, str]]:
    dataset_path = _resolve_dataset_path(path)

    with dataset_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []

        for row in reader:
            row.pop("Unnamed: 0", None)
            rows.append(row)

    return rows


def get_profiles(rows: list[dict[str, str]], n: int = 50) -> list[dict[str, str]]:
    """Return first n rows — fixed slice, never random."""
    return rows[:n]


if __name__ == "__main__":
    rows = load_dataset()
    profiles = get_profiles(rows, 50)

    print(f"Total rows: {len(rows)}")
    print(f"Columns: {list(rows[0].keys()) if rows else []}")
    print(f"Profiles for baseline: {len(profiles)}")

    if rows:
        risk_counts = Counter(row.get("Risk", "unknown") for row in rows)
        print("\nRisk distribution in full dataset:")
        print(dict(risk_counts))
        print("\nFirst borrower:")
        print(rows[0])
