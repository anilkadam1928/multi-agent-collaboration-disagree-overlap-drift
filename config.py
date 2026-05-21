# config.py

DISAGREEMENT_THRESHOLD = 0.80
SYCOPHANCY_COSINE_THRESHOLD = 0.80
DRIFT_KL_THRESHOLD = 0.15
ANCHOR_SET_SIZE = 20
BASELINE_SAMPLE_SIZE = 50

REDUNDANCY_FLAG_KEYWORDS = [
    "income",
    "document",
    "verify",
    "salary",
    "statement",
    "bank statement",
    "payslip",
]

BASELINE_RESULTS_PATH = "data/baseline_results.csv"
DISAGREEMENT_RESULTS_PATH = "data/disagreement_results.csv"
OVERLAP_RESULTS_PATH = "data/overlap_results.csv"
TEST10_RESULTS_PATH = "test10_results.csv"

AGENT_NAMES = [
    "RouterManager",
    "IncomeAgent",
    "FraudAgent",
    "CreditAgent",
    "ComplianceAgent",
    "SummariserAgent",
    "WeakModelAgent",
]
