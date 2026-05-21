# output_parser.py
import re

DECISION_KEYWORDS = {
    "approve": ["approve", "approved", "accept", "granted"],
    "reject": ["reject", "rejected", "deny", "denied", "decline", "declined"],
    "refer": ["refer", "manual review", "review further", "needs review", "escalate"],
}

CONFIDENCE_KEYWORDS = {
    "high": ["high confidence", "confident", "strongly recommend", "clearly", "high"],
    "medium": ["medium confidence", "moderate confidence", "moderate", "likely", "medium"],
    "low": ["low confidence", "uncertain", "not sure", "unclear", "low"],
}

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

def parse_decision(text: str) -> str:
    normalized = normalize_text(text)
    for label, keywords in DECISION_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return label
    return "unknown"

def parse_confidence(text: str) -> str:
    normalized = normalize_text(text)
    for label, keywords in CONFIDENCE_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return label
    return "unknown"

def parse_all(text: str) -> dict:
    return {
        "parsed_decision": parse_decision(text),
        "confidence_flag": parse_confidence(text)
    }

if __name__ == "__main__":
    tests = [
        ("Decision: Approve. High confidence. Stable income verified.", "approve", "high"),
        ("I recommend reject. Borrower has too much existing debt.", "reject", "unknown"),
        ("This case should be referred for manual review. Borderline profile.", "refer", "unknown"),
        ("Decline this application. Low confidence due to missing documents.", "reject", "low"),
    ]
    print("Testing output_parser...\n")
    for text, expected_d, expected_c in tests:
        d = parse_decision(text)
        c = parse_confidence(text)
        d_ok = "✓" if d == expected_d else f"✗ got {d}"
        c_ok = "✓" if c == expected_c else f"✗ got {c}"
        print(f"Decision {d_ok} | Confidence {c_ok} | {text[:50]}...")