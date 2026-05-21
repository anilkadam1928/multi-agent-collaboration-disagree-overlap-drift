from __future__ import annotations

import csv
from pathlib import Path


CSV_COLUMNS = [
    "run_id",
    "borrower_id",
    "agent_name",
    "raw_output",
    "parsed_decision",
    "confidence_flag",
    "kaggle_truth_label",
]


def ensure_parent_dir(file_path: str | Path) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def initialize_csv(file_path: str | Path) -> None:
    ensure_parent_dir(file_path)

    if Path(file_path).exists():
        return

    with open(file_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()


def append_result(file_path: str | Path, row: dict) -> None:
    with open(file_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writerow(row)


def fail_if_baseline_exists(file_path: str | Path) -> None:
    if Path(file_path).exists():
        raise FileExistsError(
            f"{file_path} already exists. Baseline file must never be overwritten."
        )


def save_results(file_path: str | Path, rows: list[dict]) -> None:
    """Write a fresh CSV for the current run."""
    initialize_csv(file_path)
    with open(file_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {file_path}")
