from __future__ import annotations

"""Week 4 Tuesday - ABA anchor examples.

This file is intentionally standalone. It does not edit the baseline,
disagreement, overlap, or Monday drift files.

The anchor set gives the ABA module a small calibration reference:
- borrower_profile: the structured German Credit profile text
- correct_decision: approve for Kaggle good, reject for Kaggle bad
- reasoning_keywords: short risk/stability cues for prompt injection
"""

from pathlib import Path

from config import ANCHOR_SET_SIZE
from data_loader import get_profiles, load_dataset
from profile_builder import build_borrower_profile, clean


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATHS = (
    PROJECT_ROOT / "data" / "german_credit_data.csv",
    PROJECT_ROOT / "german_credit_data.csv",
)


def _dataset_path() -> Path | None:
    for path in DATASET_PATHS:
        if path.exists():
            return path
    return None


def _as_int(value, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def _decision_from_risk(risk: str) -> str:
    return "approve" if str(risk).lower() == "good" else "reject"


def _reasoning_keywords(row: dict[str, str]) -> list[str]:
    keywords: list[str] = []

    age = _as_int(row.get("Age"))
    credit_amount = _as_int(row.get("Credit amount"))
    duration = _as_int(row.get("Duration"))
    housing = str(clean(row.get("Housing"))).lower()
    savings = str(clean(row.get("Saving accounts"))).lower()
    checking = str(clean(row.get("Checking account"))).lower()
    purpose = str(clean(row.get("Purpose"))).lower()

    if age >= 60:
        keywords.append("elderly borrower")
    elif age <= 25:
        keywords.append("young borrower")

    if housing == "own":
        keywords.append("stable housing")
    elif housing in {"rent", "free"}:
        keywords.append(f"housing={housing}")

    if savings in {"rich", "quite rich", "moderate"}:
        keywords.append("visible savings buffer")
    elif savings in {"little", "unknown"}:
        keywords.append("weak or unknown savings")

    if checking in {"little", "unknown"}:
        keywords.append("limited checking liquidity")
    elif checking in {"moderate", "rich"}:
        keywords.append("checking liquidity available")

    if credit_amount >= 5000:
        keywords.append("high credit exposure")
    elif credit_amount <= 1500:
        keywords.append("small credit amount")

    if duration >= 36:
        keywords.append("long repayment duration")
    elif 0 < duration <= 12:
        keywords.append("short repayment duration")

    if purpose:
        keywords.append(f"purpose={purpose}")

    return keywords[:7]


def build_anchor_examples(limit: int = ANCHOR_SET_SIZE) -> list[dict[str, object]]:
    dataset_path = _dataset_path()
    rows = load_dataset(dataset_path) if dataset_path else load_dataset()
    profiles = get_profiles(rows, limit)

    anchors: list[dict[str, object]] = []
    for index, row in enumerate(profiles, start=1):
        profile_text, risk = build_borrower_profile(row)
        anchors.append(
            {
                "borrower_id": f"B{index:03d}",
                "borrower_profile": profile_text,
                "correct_decision": _decision_from_risk(risk),
                "reasoning_keywords": _reasoning_keywords(row),
                "kaggle_truth_label": str(risk).lower(),
            }
        )
    return anchors


ANCHOR_EXAMPLES = build_anchor_examples()


def get_anchor_examples(limit: int = ANCHOR_SET_SIZE, decision: str | None = None) -> list[dict[str, object]]:
    examples = ANCHOR_EXAMPLES
    if decision:
        examples = [item for item in examples if item["correct_decision"] == decision]
    return examples[:limit]


if __name__ == "__main__":
    print(f"Loaded {len(ANCHOR_EXAMPLES)} ABA anchor examples.")
    for item in ANCHOR_EXAMPLES[:5]:
        keywords = ", ".join(item["reasoning_keywords"])
        print(
            f"{item['borrower_id']} | truth={item['kaggle_truth_label']} | "
            f"correct={item['correct_decision']} | {keywords}"
        )
