# profile_builder.py
from __future__ import annotations


def clean(value):
    """Normalize blank-ish values without relying on pandas."""
    if value is None:
        return "unknown"

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"nan", "na", "n/a", "none", "null"}:
            return "unknown"
        return stripped

    return value


def build_borrower_profile(row):
    age = clean(row.get("Age", "unknown"))
    sex = clean(row.get("Sex", "unknown"))
    job = clean(row.get("Job", "unknown"))
    housing = clean(row.get("Housing", "unknown"))
    savings = clean(row.get("Saving accounts", "unknown"))
    checking = clean(row.get("Checking account", "unknown"))
    credit_amount = clean(row.get("Credit amount", "unknown"))
    duration = clean(row.get("Duration", "unknown"))
    purpose = clean(row.get("Purpose", "unknown"))
    risk = clean(row.get("Risk", "unknown"))

    profile_text = (
        "Borrower profile from the German Credit research dataset.\n"
        "Only the structured fields below are available. No salary slips, bank statements, "
        "KYC documents, identity files, or uploaded evidence exist for this case.\n"
        f"- Age: {age}\n"
        f"- Sex: {sex}\n"
        f"- Job type: {job}\n"
        f"- Housing: {housing}\n"
        f"- Saving accounts status: {savings}\n"
        f"- Checking account status: {checking}\n"
        f"- Credit amount: {credit_amount} DM\n"
        f"- Duration: {duration} months\n"
        f"- Purpose: {purpose}\n"
        "Important interpretation rule: if a value is shown as 'unknown', treat it as the dataset's "
        "recorded field value and assess risk from it directly. Do not ask for extra documents."
    )

    return profile_text, risk


if __name__ == "__main__":
    from data_loader import get_profiles, load_dataset

    rows = load_dataset()
    profiles = get_profiles(rows, 50)

    print("Checking for missing values in key columns:")
    savings_unknown = sum(clean(row.get("Saving accounts")) == "unknown" for row in profiles)
    checking_unknown = sum(clean(row.get("Checking account")) == "unknown" for row in profiles)
    print({"Saving accounts": savings_unknown, "Checking account": checking_unknown})
    print()

    for i, row in enumerate(profiles[:3], start=1):
        text, label = build_borrower_profile(row)
        print(f"Borrower {i}:")
        print(f"  Profile: {text}")
        print(f"  Truth label: {label}")
        print()
