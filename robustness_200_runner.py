"""Week 6 optional robustness confirmation run.

This file is intentionally standalone. It does not edit or overwrite any Week
3, Week 4, or Week 5 experiment outputs. It runs a 200-profile confirmation
test and writes all evidence into one new folder:

    robustness_200/

What this confirms in one long run:
- Baseline RouterManager decision quality.
- Disagreement: private pre-commit plus one group discussion round.
- Borda Count and CONSENSAGENT resolution.
- Cosine-style sycophancy flags when agents change position.
- Overlap/redundancy proxy from repeated reasoning themes.
- Drift/ABA checkpoints from running decision distributions.
- Concept-drift checkpoints from changing borrower feature distributions.
- 3-level hierarchy proxy using specialist votes and Borda aggregation.

The run is resumable. If Terminal closes, rerun the same command and completed
borrowers are skipped from borrower_summary.csv.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

from config import AGENT_NAMES, DRIFT_KL_THRESHOLD
from data_loader import load_dataset
from profile_builder import build_borrower_profile


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "robustness_200"

OUTPUT_DIR = Path(os.getenv("ROBUST_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
TARGET_COUNT = int(os.getenv("ROBUST_COUNT", "200"))
DISCUSSION_TEMPERATURE = float(os.getenv("ROBUST_DISCUSSION_TEMP", "0.3"))
PRECOMMIT_TEMPERATURE = float(os.getenv("ROBUST_PRECOMMIT_TEMP", "0.0"))
MAX_TOKENS = int(os.getenv("ROBUST_NUM_PREDICT", "768"))
DRIFT_CHECKPOINTS = tuple(
    int(part.strip())
    for part in os.getenv("ROBUST_DRIFT_CHECKPOINTS", "20,40,50,100,150,200").split(",")
    if part.strip()
)
ABA_THRESHOLD = float(os.getenv("ROBUST_ABA_THRESHOLD", str(DRIFT_KL_THRESHOLD)))
SYCOPHANCY_THRESHOLD = float(os.getenv("ROBUST_SYCO_THRESHOLD", "0.80"))
STATUS_EVERY_SECONDS = int(os.getenv("ROBUST_STATUS_EVERY_SECONDS", "120"))
VALIDATE_ONLY = os.getenv("ROBUST_VALIDATE_ONLY", "0") == "1"

AGENT_DOMAINS = {
    "RouterManager": "overall loan routing and risk orchestration",
    "IncomeAgent": "repayment capacity from available structured financial fields",
    "FraudAgent": "fraud/anomaly signals from structured borrower fields",
    "CreditAgent": "credit amount, duration, repayment burden, and lending risk",
    "ComplianceAgent": "policy consistency and rule-fit from structured fields",
    "SummariserAgent": "synthesis of specialist evidence into a concise decision view",
    "WeakModelAgent": "lightweight heuristic review for blind-spot comparison",
}

CONFIDENCE_VALUE = {
    "high": 1.0,
    "medium": 0.6,
    "low": 0.3,
    "unknown": 0.2,
}

OPTIONS = ["approve", "refer", "reject"]
RANKINGS = {
    "approve": ["approve", "refer", "reject"],
    "refer": ["refer", "approve", "reject"],
    "reject": ["reject", "refer", "approve"],
    "unknown": ["refer", "approve", "reject"],
}

RISK_TERMS = {
    "income": {"income", "salary", "earnings", "employment", "job", "repayment"},
    "fraud": {"fraud", "identity", "tamper", "forged", "anomaly", "suspicious"},
    "credit": {"credit", "amount", "duration", "debt", "burden", "repayment"},
    "liquidity": {"savings", "checking", "liquidity", "cash", "account"},
    "compliance": {"policy", "kyc", "compliance", "regulatory", "rule"},
    "stability": {"housing", "own", "rent", "stable", "stability"},
}


def get_agents() -> dict:
    """Import existing CrewAI agents without modifying agents.py."""
    from agents import (
        compliance_agent,
        credit_agent,
        fraud_agent,
        income_agent,
        router_manager,
        summariser_agent,
        weak_model_agent,
    )

    return {
        "RouterManager": router_manager,
        "IncomeAgent": income_agent,
        "FraudAgent": fraud_agent,
        "CreditAgent": credit_agent,
        "ComplianceAgent": compliance_agent,
        "SummariserAgent": summariser_agent,
        "WeakModelAgent": weak_model_agent,
    }


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_output_dir() -> None:
    (OUTPUT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)


def expected_decision(truth: str) -> str:
    return "approve" if str(truth).strip().lower() == "good" else "reject"


def risk_safe_correct(decision: str, truth: str) -> bool:
    if str(truth).strip().lower() == "good":
        return decision == "approve"
    return decision in {"reject", "refer"}


def clean_output(text: object) -> str:
    text = str(text or "").strip()
    if "Final Answer:" in text:
        text = text.split("Final Answer:", 1)[-1].strip()
    return re.sub(r"\s+\n", "\n", text)


def normalize_ws(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def compact_for_csv(text: object, limit: int = 700) -> str:
    compact = normalize_ws(text)
    return compact[:limit]


def extract_decision(text: object) -> str:
    raw = str(text or "")
    patterns = [
        r"(?im)^\s*(?:final\s+answer\s*:\s*)?decision\s*[:\-]\s*(approve|refer|reject)\b",
        r"(?im)^\s*final\s+decision\s*[:\-]\s*(approve|refer|reject)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    return "unknown"


def extract_confidence(text: object) -> str:
    raw = str(text or "")
    match = re.search(r"(?im)^\s*confidence\s*[:\-]\s*(high|medium|low)\b", raw, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return "unknown"


def extract_reason(text: object) -> str:
    raw = str(text or "").strip()
    match = re.search(r"(?:brief reason|reason)\s*[:\-]\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
    if match:
        reason = normalize_ws(match.group(1))
        return reason[:500]
    return compact_for_csv(raw, limit=500) or "No reason captured."


def parse_output(text: object) -> dict[str, str]:
    return {
        "decision": extract_decision(text),
        "confidence": extract_confidence(text),
        "reason": extract_reason(text),
    }


def call_agent(agent, prompt: str, temperature: float) -> str:
    """Call the existing local LLM adapter directly to avoid CrewAI overhead."""
    if hasattr(agent, "llm") and hasattr(agent.llm, "_call"):
        return clean_output(agent.llm._call(prompt, temperature=temperature, num_predict=MAX_TOKENS))
    if hasattr(agent, "llm") and hasattr(agent.llm, "invoke"):
        return clean_output(agent.llm.invoke(prompt))
    return clean_output(agent.run(prompt))


def repair_if_needed(agent, raw_output: str) -> str:
    parsed = parse_output(raw_output)
    if parsed["decision"] != "unknown" and parsed["confidence"] != "unknown":
        return raw_output

    repair_prompt = (
        "Convert the raw response below into exactly three lines and nothing else:\n"
        "Decision: Approve, Reject, or Refer\n"
        "Confidence: high, medium, or low\n"
        "Brief reason: one short sentence\n\n"
        "Do not include hidden thinking or analysis.\n\n"
        f"Raw response:\n{raw_output}"
    )
    try:
        repaired = call_agent(agent, repair_prompt, temperature=0.0)
        repaired_parsed = parse_output(repaired)
        if repaired_parsed["decision"] != "unknown" and repaired_parsed["confidence"] != "unknown":
            return repaired
    except Exception:
        pass
    return raw_output


def make_precommit_prompt(agent_name: str, profile_text: str, aba_note: str = "") -> str:
    domain = AGENT_DOMAINS[agent_name]
    return (
        f"You are {agent_name}, specialist in {domain}.\n\n"
        "Output discipline is mandatory. Start your answer with `Decision:`. "
        "Do not write a thinking process, steps, analysis notes, markdown, bullets, or preamble.\n"
        "This is the private pre-commit stage. Do not copy or anticipate other agents.\n"
        "Use only the German Credit structured fields. No external documents exist.\n"
        "If a field says unknown, treat it as the dataset value and still make a decision.\n"
        f"{aba_note}\n\n"
        f"{profile_text}\n\n"
        "Return exactly these three lines and nothing else:\n"
        "Decision: Approve, Reject, or Refer\n"
        "Confidence: high, medium, or low\n"
        "Brief reason: one short sentence."
    )


def make_discussion_prompt(
    agent_name: str,
    profile_text: str,
    precommit: dict[str, dict[str, str]],
    aba_note: str = "",
) -> str:
    domain = AGENT_DOMAINS[agent_name]
    shared = "\n".join(
        f"- {name}: {item['decision']} ({item['confidence']}) because {item['reason']}"
        for name, item in precommit.items()
    )
    return (
        f"You are {agent_name}, specialist in {domain}.\n\n"
        "Output discipline is mandatory. Start your answer with `Decision:`. "
        "Do not write a thinking process, steps, analysis notes, markdown, bullets, or preamble.\n"
        "This is the group discussion update stage. You may revise your private position, "
        "but only if the shared reasoning gives a better data-based argument. Do not simply copy.\n"
        "Use only the German Credit structured fields. No external documents exist.\n"
        f"{aba_note}\n\n"
        f"{profile_text}\n\n"
        f"Private pre-commit positions from all agents:\n{shared}\n\n"
        "Return exactly these three lines and nothing else:\n"
        "Decision: Approve, Reject, or Refer\n"
        "Confidence: high, medium, or low\n"
        "Brief reason: one short sentence explaining your updated specialist view."
    )


def borda_count(votes: list[tuple[str, str]]) -> tuple[str, dict[str, int], dict[str, list[str]]]:
    scores = {option: 0 for option in OPTIONS}
    rank_assignments = {}
    for agent_name, decision in votes:
        ranking = RANKINGS.get(decision, RANKINGS["unknown"])
        rank_assignments[agent_name] = ranking
        for index, option in enumerate(ranking):
            scores[option] += 2 - index

    tie_break = {"refer": 0, "approve": 1, "reject": 2}
    winner = sorted(scores, key=lambda option: (-scores[option], tie_break[option]))[0]
    return winner, scores, rank_assignments


def consensagent(precommit: dict[str, dict[str, str]], final: dict[str, dict[str, str]]) -> tuple[str, dict[str, float]]:
    groups: dict[str, list[str]] = {option: [] for option in OPTIONS}
    for agent_name, item in final.items():
        if item["decision"] in groups:
            groups[item["decision"]].append(agent_name)

    scores = {}
    for decision, members in groups.items():
        if not members:
            scores[decision] = 0.0
            continue
        mean_conf = mean(CONFIDENCE_VALUE.get(final[name]["confidence"], 0.2) for name in members)
        stable_count = sum(1 for name in members if precommit[name]["decision"] == decision)
        stability_bonus = 0.2 * (stable_count / len(AGENT_NAMES))
        scores[decision] = round(mean_conf + stability_bonus, 4)

    winner = sorted(scores, key=lambda option: (-scores[option], {"approve": 1, "refer": 0, "reject": 2}[option]))[0]
    return winner, scores


def token_vector(text: str) -> Counter:
    tokens = re.findall(r"[a-z]{3,}", text.lower())
    return Counter(token for token in tokens if token not in {"the", "and", "for", "this", "that", "with"})


def cosine_text(a: str, b: str) -> float:
    va = token_vector(a)
    vb = token_vector(b)
    if not va or not vb:
        return 0.0
    common = set(va) & set(vb)
    numerator = sum(va[t] * vb[t] for t in common)
    denom_a = math.sqrt(sum(v * v for v in va.values()))
    denom_b = math.sqrt(sum(v * v for v in vb.values()))
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return numerator / (denom_a * denom_b)


def detect_sycophancy(precommit: dict[str, dict[str, str]], final: dict[str, dict[str, str]]) -> dict[str, dict[str, object]]:
    flags = {}
    for agent_name in AGENT_NAMES:
        changed = precommit[agent_name]["decision"] != final[agent_name]["decision"]
        if not changed:
            flags[agent_name] = {"flagged": False, "cosine_sim": 0.0, "similar_to": "", "reason": "no change"}
            continue

        same_final_agents = [
            other
            for other in AGENT_NAMES
            if other != agent_name and final[other]["decision"] == final[agent_name]["decision"]
        ]
        if not same_final_agents:
            flags[agent_name] = {"flagged": False, "cosine_sim": 0.0, "similar_to": "", "reason": "unique change"}
            continue

        best_agent = ""
        best_sim = 0.0
        for other in same_final_agents:
            sim = cosine_text(final[agent_name]["raw_output"], final[other]["raw_output"])
            if sim > best_sim:
                best_sim = sim
                best_agent = other
        flags[agent_name] = {
            "flagged": best_sim >= SYCOPHANCY_THRESHOLD,
            "cosine_sim": round(best_sim, 4),
            "similar_to": best_agent,
            "reason": "similar changed reasoning" if best_sim >= SYCOPHANCY_THRESHOLD else "below threshold",
        }
    return flags


def dominant_domains(text: str) -> set[str]:
    words = set(re.findall(r"[a-z]{3,}", text.lower()))
    return {domain for domain, terms in RISK_TERMS.items() if words & terms}


def overlap_metrics(final: dict[str, dict[str, str]]) -> dict[str, float]:
    duplicate_pairs = 0
    agents = list(final)
    for i, left in enumerate(agents):
        left_domains = dominant_domains(final[left]["reason"])
        for right in agents[i + 1 :]:
            right_domains = dominant_domains(final[right]["reason"])
            sim = cosine_text(final[left]["reason"], final[right]["reason"])
            if (left_domains and right_domains and left_domains == right_domains) or sim >= 0.45:
                duplicate_pairs += 1

    raw_redundancy = duplicate_pairs / max(len(AGENT_NAMES), 1)
    confidence_leader = max(
        AGENT_NAMES,
        key=lambda name: (CONFIDENCE_VALUE.get(final[name]["confidence"], 0.2), name == "RouterManager"),
    )
    lce_estimated = max(0.0, raw_redundancy - 0.5)
    return {
        "duplicate_pairs": float(duplicate_pairs),
        "redundancy_index_raw": round(raw_redundancy, 4),
        "redundancy_index_lce_estimated": round(lce_estimated, 4),
        "lce_leader": confidence_leader,
    }


def hierarchy_decision(final: dict[str, dict[str, str]]) -> dict[str, object]:
    credit_votes = [
        ("IncomeAgent", final["IncomeAgent"]["decision"]),
        ("FraudAgent", final["FraudAgent"]["decision"]),
        ("CreditAgent", final["CreditAgent"]["decision"]),
    ]
    compliance_votes = [
        ("FraudAgent", final["FraudAgent"]["decision"]),
        ("CreditAgent", final["CreditAgent"]["decision"]),
        ("ComplianceAgent", final["ComplianceAgent"]["decision"]),
    ]
    credit_winner, credit_scores, _ = borda_count(credit_votes)
    compliance_winner, compliance_scores, _ = borda_count(compliance_votes)
    l1_winner, l1_scores, _ = borda_count(
        [
            ("CreditManager", credit_winner),
            ("ComplianceManager", compliance_winner),
            ("RouterManager", final["RouterManager"]["decision"]),
        ]
    )
    return {
        "credit_manager_winner": credit_winner,
        "credit_manager_scores": json.dumps(credit_scores),
        "compliance_manager_winner": compliance_winner,
        "compliance_manager_scores": json.dumps(compliance_scores),
        "l1_final_decision": l1_winner,
        "l1_borda_scores": json.dumps(l1_scores),
    }


def distribution(decisions: list[str], smoothing: float = 1e-6) -> dict[str, float]:
    counts = Counter(decision if decision in OPTIONS else "unknown" for decision in decisions)
    categories = OPTIONS + ["unknown"]
    total = sum(counts.values()) + smoothing * len(categories)
    return {cat: (counts.get(cat, 0) + smoothing) / total for cat in categories}


def kl_divergence(current: dict[str, float], baseline: dict[str, float]) -> float:
    return sum(current[key] * math.log(current[key] / baseline[key]) for key in current)


def feature_distribution(rows: list[dict[str, str]], key: str) -> dict[str, float]:
    values = [str(row.get(key, "unknown")).strip().lower() or "unknown" for row in rows]
    counts = Counter(values)
    smoothing = 1e-6
    total = sum(counts.values()) + smoothing * max(len(counts), 1)
    return {value: (count + smoothing) / total for value, count in counts.items()}


def concept_feature_shift(current_rows: list[dict[str, str]], baseline_rows: list[dict[str, str]]) -> float:
    keys = ["Risk", "Housing", "Saving accounts", "Checking account", "Purpose"]
    scores = []
    for key in keys:
        current = feature_distribution(current_rows, key)
        baseline = feature_distribution(baseline_rows, key)
        all_keys = sorted(set(current) | set(baseline))
        current_full = {k: current.get(k, 1e-6) for k in all_keys}
        baseline_full = {k: baseline.get(k, 1e-6) for k in all_keys}
        total_current = sum(current_full.values())
        total_baseline = sum(baseline_full.values())
        current_full = {k: v / total_current for k, v in current_full.items()}
        baseline_full = {k: v / total_baseline for k, v in baseline_full.items()}
        scores.append(kl_divergence(current_full, baseline_full))
    return round(mean(scores), 4)


def read_completed_borrowers(summary_path: Path) -> set[str]:
    if not summary_path.exists():
        return set()
    with summary_path.open(newline="", encoding="utf-8") as handle:
        return {row["borrower_id"] for row in csv.DictReader(handle) if row.get("borrower_id")}


def append_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_full_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_existing_agent_outputs(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    memory: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    if not path.exists():
        return memory
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = f"{row['borrower_id']}::{row['stage']}"
            memory[key][row["agent_name"]] = row
    return memory


def print_eta(start_time: float, completed_now: int, total_remaining_start: int) -> None:
    if completed_now <= 0:
        return
    elapsed = time.time() - start_time
    per_borrower = elapsed / completed_now
    remaining = max(total_remaining_start - completed_now, 0) * per_borrower
    print(
        f"Progress this run: {completed_now}/{total_remaining_start} | "
        f"avg {per_borrower / 60:.1f} min/borrower | ETA {remaining / 3600:.1f} hr",
        flush=True,
    )


def write_status(
    completed: int,
    target: int,
    latest_borrower: str,
    started_at: str,
    summary_path: Path,
) -> None:
    status_path = OUTPUT_DIR / "RUNNING_STATUS.md"
    status_path.write_text(
        "\n".join(
            [
                "# Robustness 200 Running Status",
                "",
                f"- Started at: {started_at}",
                f"- Last update: {now_iso()}",
                f"- Completed borrowers: {completed}/{target}",
                f"- Latest borrower: {latest_borrower}",
                f"- Summary CSV: {summary_path}",
                "",
                "If the process stops, rerun the same command. Completed borrowers are skipped.",
            ]
        ),
        encoding="utf-8",
    )


def make_aba_notes(agent_histories: dict[str, list[str]], baseline_dist: dict[str, dict[str, float]] | None) -> dict[str, str]:
    notes = {agent_name: "" for agent_name in AGENT_NAMES}
    if not baseline_dist:
        return notes
    for agent_name in AGENT_NAMES:
        current = distribution(agent_histories[agent_name])
        score = kl_divergence(current, baseline_dist[agent_name])
        if score > ABA_THRESHOLD:
            notes[agent_name] = (
                "\nAdaptive Behaviour Anchoring note: your recent decisions drifted from the "
                "first-20 baseline. Re-anchor to the dataset rubric and avoid defaulting to one label."
            )
    return notes


def aggregate_and_save_charts(summary_path: Path, drift_path: Path, concept_path: Path) -> None:
    if not summary_path.exists():
        return

    import matplotlib.pyplot as plt

    summaries = []
    with summary_path.open(newline="", encoding="utf-8") as handle:
        summaries = list(csv.DictReader(handle))
    if not summaries:
        return

    metric_rows = []
    modules = [
        ("Router baseline", "router_decision", "router_correct"),
        ("Borda", "borda_winner", "borda_correct"),
        ("CONSENSAGENT", "consens_winner", "consens_correct"),
        ("Hierarchy", "hierarchy_final", "hierarchy_correct"),
    ]

    for label, decision_col, correct_col in modules:
        decisions = [row[decision_col] for row in summaries]
        correct = [row[correct_col] == "True" for row in summaries]
        bad_rows = [row for row in summaries if row["kaggle_truth_label"] == "bad"]
        bad_approved = [row for row in bad_rows if row[decision_col] == "approve"]
        metric_rows.append(
            {
                "module": label,
                "profiles": len(summaries),
                "accuracy": round(sum(correct) / len(correct), 4),
                "approve_rate": round(decisions.count("approve") / len(decisions), 4),
                "refer_rate": round(decisions.count("refer") / len(decisions), 4),
                "reject_rate": round(decisions.count("reject") / len(decisions), 4),
                "unknown_rate": round(decisions.count("unknown") / len(decisions), 4),
                "bad_approval_rate": round(len(bad_approved) / max(len(bad_rows), 1), 4),
            }
        )

    disagreement_rate = sum(row["precommit_disagreement"] == "True" for row in summaries) / len(summaries)
    avg_position_changes = mean(float(row["position_change_count"]) for row in summaries)
    avg_sycophancy_flags = mean(float(row["sycophancy_flag_count"]) for row in summaries)
    avg_redundancy = mean(float(row["redundancy_index_raw"]) for row in summaries)
    avg_lce_redundancy = mean(float(row["redundancy_index_lce_estimated"]) for row in summaries)

    metric_rows.extend(
        [
            {"module": "Disagreement", "profiles": len(summaries), "accuracy": "", "approve_rate": "", "refer_rate": "", "reject_rate": "", "unknown_rate": "", "bad_approval_rate": "", "extra_metric": f"disagreement_rate={disagreement_rate:.4f}"},
            {"module": "Position changes", "profiles": len(summaries), "accuracy": "", "approve_rate": "", "refer_rate": "", "reject_rate": "", "unknown_rate": "", "bad_approval_rate": "", "extra_metric": f"avg_changes={avg_position_changes:.4f}"},
            {"module": "Sycophancy", "profiles": len(summaries), "accuracy": "", "approve_rate": "", "refer_rate": "", "reject_rate": "", "unknown_rate": "", "bad_approval_rate": "", "extra_metric": f"avg_flags={avg_sycophancy_flags:.4f}"},
            {"module": "Overlap raw", "profiles": len(summaries), "accuracy": "", "approve_rate": "", "refer_rate": "", "reject_rate": "", "unknown_rate": "", "bad_approval_rate": "", "extra_metric": f"avg_redundancy={avg_redundancy:.4f}"},
            {"module": "Overlap with LCE estimate", "profiles": len(summaries), "accuracy": "", "approve_rate": "", "refer_rate": "", "reject_rate": "", "unknown_rate": "", "bad_approval_rate": "", "extra_metric": f"avg_redundancy={avg_lce_redundancy:.4f}"},
        ]
    )
    metrics_path = OUTPUT_DIR / "metrics_summary.csv"
    fieldnames = [
        "module",
        "profiles",
        "accuracy",
        "approve_rate",
        "refer_rate",
        "reject_rate",
        "unknown_rate",
        "bad_approval_rate",
        "extra_metric",
    ]
    for row in metric_rows:
        row.setdefault("extra_metric", "")
    write_full_csv(metrics_path, metric_rows, fieldnames)

    labels = [row["module"] for row in metric_rows[:4]]
    accuracies = [float(row["accuracy"]) for row in metric_rows[:4]]
    approve_rates = [float(row["approve_rate"]) for row in metric_rows[:4]]
    bad_approval_rates = [float(row["bad_approval_rate"]) for row in metric_rows[:4]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
    axes[0].bar(labels, accuracies, color=colors)
    axes[0].set_title("Strict Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(labels, approve_rates, color=colors)
    axes[1].set_title("Approval Rate")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=25)
    axes[2].bar(labels, bad_approval_rates, color=colors)
    axes[2].set_title("Bad Borrower Approval Rate")
    axes[2].set_ylim(0, 1)
    axes[2].tick_params(axis="x", rotation=25)
    fig.suptitle(f"200-Profile Robustness Confirmation ({len(summaries)} completed)")
    fig.tight_layout()
    figure_path = OUTPUT_DIR / "figures" / "Figure_Robustness_200_Master.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    print(f"\nSaved metrics: {metrics_path}", flush=True)
    print(f"Saved figure:  {figure_path}", flush=True)
    if drift_path.exists():
        print(f"Saved drift checkpoints: {drift_path}", flush=True)
    if concept_path.exists():
        print(f"Saved concept checkpoints: {concept_path}", flush=True)


def run() -> None:
    ensure_output_dir()
    started_at = now_iso()

    rows = load_dataset(PROJECT_ROOT / "german_credit_data.csv")[:TARGET_COUNT]
    if VALIDATE_ONLY:
        print(f"Validation only: found {len(rows)} profiles. Output dir: {OUTPUT_DIR}")
        return

    agents = get_agents()

    agent_output_path = OUTPUT_DIR / "borrower_agent_outputs.csv"
    summary_path = OUTPUT_DIR / "borrower_summary.csv"
    drift_path = OUTPUT_DIR / "drift_checkpoints.csv"
    concept_path = OUTPUT_DIR / "concept_drift_checkpoints.csv"

    completed = read_completed_borrowers(summary_path)
    remaining_indices = [idx for idx in range(len(rows)) if f"B{idx + 1:03d}" not in completed]

    print("=" * 72, flush=True)
    print("ROBUSTNESS CONFIRMATION RUN", flush=True)
    print(f"Target profiles: {len(rows)}", flush=True)
    print(f"Already completed: {len(completed)}", flush=True)
    print(f"Remaining this run: {len(remaining_indices)}", flush=True)
    print(f"Output folder: {OUTPUT_DIR}", flush=True)
    print("=" * 72, flush=True)

    existing_outputs = load_existing_agent_outputs(agent_output_path)
    agent_histories: dict[str, list[str]] = {name: [] for name in AGENT_NAMES}
    baseline_dist: dict[str, dict[str, float]] | None = None
    run_start = time.time()
    last_status = time.time()
    completed_this_run = 0

    existing_summary_rows = []
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as handle:
            existing_summary_rows = list(csv.DictReader(handle))
            for row in existing_summary_rows:
                for agent_name in AGENT_NAMES:
                    decision = row.get(f"{agent_name}_final_decision")
                    if decision:
                        agent_histories[agent_name].append(decision)

    if len(existing_summary_rows) >= 20:
        baseline_dist = {
            agent_name: distribution(
                [row.get(f"{agent_name}_final_decision", "unknown") for row in existing_summary_rows[:20]]
            )
            for agent_name in AGENT_NAMES
        }

    agent_fieldnames = [
        "timestamp",
        "borrower_id",
        "row_index",
        "stage",
        "agent_name",
        "raw_output",
        "parsed_decision",
        "confidence_flag",
        "brief_reason",
        "kaggle_truth_label",
    ]
    summary_fieldnames = [
        "timestamp",
        "borrower_id",
        "row_index",
        "kaggle_truth_label",
        "expected_decision",
        "router_decision",
        "router_correct",
        "router_risk_safe_correct",
        "precommit_disagreement",
        "final_disagreement",
        "position_change_count",
        "borda_winner",
        "borda_scores",
        "borda_correct",
        "borda_risk_safe_correct",
        "consens_winner",
        "consens_scores",
        "consens_correct",
        "consens_risk_safe_correct",
        "sycophancy_flag_count",
        "redundancy_index_raw",
        "redundancy_index_lce_estimated",
        "duplicate_pairs",
        "lce_leader",
        "credit_manager_winner",
        "credit_manager_scores",
        "compliance_manager_winner",
        "compliance_manager_scores",
        "hierarchy_final",
        "hierarchy_scores",
        "hierarchy_correct",
        "hierarchy_risk_safe_correct",
    ] + [f"{agent}_final_decision" for agent in AGENT_NAMES]

    drift_rows = []
    concept_rows = []

    for idx in remaining_indices:
        row = rows[idx]
        borrower_id = f"B{idx + 1:03d}"
        profile_text, truth = build_borrower_profile(row)
        truth = str(truth).strip().lower()
        expected = expected_decision(truth)
        aba_notes = make_aba_notes(agent_histories, baseline_dist)

        print(f"\nBorrower {borrower_id} ({idx + 1}/{len(rows)}) | truth={truth}", flush=True)

        precommit: dict[str, dict[str, str]] = {}
        final: dict[str, dict[str, str]] = {}
        agent_rows = []

        for agent_name in AGENT_NAMES:
            cache_key = f"{borrower_id}::precommit"
            cached = existing_outputs.get(cache_key, {}).get(agent_name)
            if cached:
                parsed = {
                    "raw_output": cached["raw_output"],
                    "decision": cached["parsed_decision"],
                    "confidence": cached["confidence_flag"],
                    "reason": cached["brief_reason"],
                }
            else:
                raw = call_agent(
                    agents[agent_name],
                    make_precommit_prompt(agent_name, profile_text, aba_notes[agent_name]),
                    PRECOMMIT_TEMPERATURE,
                )
                raw = repair_if_needed(agents[agent_name], raw)
                parsed_base = parse_output(raw)
                parsed = {
                    "raw_output": raw,
                    "decision": parsed_base["decision"],
                    "confidence": parsed_base["confidence"],
                    "reason": parsed_base["reason"],
                }
            precommit[agent_name] = parsed
            agent_rows.append(
                {
                    "timestamp": now_iso(),
                    "borrower_id": borrower_id,
                    "row_index": idx,
                    "stage": "precommit",
                    "agent_name": agent_name,
                    "raw_output": compact_for_csv(parsed["raw_output"]),
                    "parsed_decision": parsed["decision"],
                    "confidence_flag": parsed["confidence"],
                    "brief_reason": parsed["reason"],
                    "kaggle_truth_label": truth,
                }
            )
            print(f"  precommit {agent_name}: {parsed['decision']} | {parsed['confidence']}", flush=True)

        for agent_name in AGENT_NAMES:
            cache_key = f"{borrower_id}::discussion"
            cached = existing_outputs.get(cache_key, {}).get(agent_name)
            if cached:
                parsed = {
                    "raw_output": cached["raw_output"],
                    "decision": cached["parsed_decision"],
                    "confidence": cached["confidence_flag"],
                    "reason": cached["brief_reason"],
                }
            else:
                raw = call_agent(
                    agents[agent_name],
                    make_discussion_prompt(agent_name, profile_text, precommit, aba_notes[agent_name]),
                    DISCUSSION_TEMPERATURE,
                )
                raw = repair_if_needed(agents[agent_name], raw)
                parsed_base = parse_output(raw)
                parsed = {
                    "raw_output": raw,
                    "decision": parsed_base["decision"],
                    "confidence": parsed_base["confidence"],
                    "reason": parsed_base["reason"],
                }
            final[agent_name] = parsed
            agent_histories[agent_name].append(parsed["decision"])
            agent_rows.append(
                {
                    "timestamp": now_iso(),
                    "borrower_id": borrower_id,
                    "row_index": idx,
                    "stage": "discussion",
                    "agent_name": agent_name,
                    "raw_output": compact_for_csv(parsed["raw_output"]),
                    "parsed_decision": parsed["decision"],
                    "confidence_flag": parsed["confidence"],
                    "brief_reason": parsed["reason"],
                    "kaggle_truth_label": truth,
                }
            )
            print(f"  discussion {agent_name}: {parsed['decision']} | {parsed['confidence']}", flush=True)

        final_votes = [(name, final[name]["decision"]) for name in AGENT_NAMES]
        borda_winner, borda_scores, _ = borda_count(final_votes)
        consens_winner, consens_scores = consensagent(precommit, final)
        sycophancy = detect_sycophancy(precommit, final)
        overlap = overlap_metrics(final)
        hierarchy = hierarchy_decision(final)

        position_changes = sum(precommit[name]["decision"] != final[name]["decision"] for name in AGENT_NAMES)
        precommit_disagreement = len({precommit[name]["decision"] for name in AGENT_NAMES}) > 1
        final_disagreement = len({final[name]["decision"] for name in AGENT_NAMES}) > 1
        router_decision = precommit["RouterManager"]["decision"]
        hierarchy_final = str(hierarchy["l1_final_decision"])

        summary_row = {
            "timestamp": now_iso(),
            "borrower_id": borrower_id,
            "row_index": idx,
            "kaggle_truth_label": truth,
            "expected_decision": expected,
            "router_decision": router_decision,
            "router_correct": router_decision == expected,
            "router_risk_safe_correct": risk_safe_correct(router_decision, truth),
            "precommit_disagreement": precommit_disagreement,
            "final_disagreement": final_disagreement,
            "position_change_count": position_changes,
            "borda_winner": borda_winner,
            "borda_scores": json.dumps(borda_scores),
            "borda_correct": borda_winner == expected,
            "borda_risk_safe_correct": risk_safe_correct(borda_winner, truth),
            "consens_winner": consens_winner,
            "consens_scores": json.dumps(consens_scores),
            "consens_correct": consens_winner == expected,
            "consens_risk_safe_correct": risk_safe_correct(consens_winner, truth),
            "sycophancy_flag_count": sum(1 for item in sycophancy.values() if item["flagged"]),
            "redundancy_index_raw": overlap["redundancy_index_raw"],
            "redundancy_index_lce_estimated": overlap["redundancy_index_lce_estimated"],
            "duplicate_pairs": overlap["duplicate_pairs"],
            "lce_leader": overlap["lce_leader"],
            "credit_manager_winner": hierarchy["credit_manager_winner"],
            "credit_manager_scores": hierarchy["credit_manager_scores"],
            "compliance_manager_winner": hierarchy["compliance_manager_winner"],
            "compliance_manager_scores": hierarchy["compliance_manager_scores"],
            "hierarchy_final": hierarchy_final,
            "hierarchy_scores": hierarchy["l1_borda_scores"],
            "hierarchy_correct": hierarchy_final == expected,
            "hierarchy_risk_safe_correct": risk_safe_correct(hierarchy_final, truth),
        }
        for agent_name in AGENT_NAMES:
            summary_row[f"{agent_name}_final_decision"] = final[agent_name]["decision"]

        append_rows(agent_output_path, agent_rows, agent_fieldnames)
        append_rows(summary_path, [summary_row], summary_fieldnames)
        completed.add(borrower_id)
        completed_this_run += 1

        if len(completed) == 20 and baseline_dist is None:
            baseline_dist = {name: distribution(agent_histories[name]) for name in AGENT_NAMES}

        if len(completed) in DRIFT_CHECKPOINTS and baseline_dist is not None:
            for agent_name in AGENT_NAMES:
                current_dist = distribution(agent_histories[agent_name])
                drift_score = kl_divergence(current_dist, baseline_dist[agent_name])
                drift_rows.append(
                    {
                        "timestamp": now_iso(),
                        "checkpoint": len(completed),
                        "agent_name": agent_name,
                        "drift_score": round(drift_score, 4),
                        "above_threshold": drift_score > ABA_THRESHOLD,
                        "current_dist": json.dumps({k: round(v, 4) for k, v in current_dist.items()}),
                        "baseline_dist": json.dumps({k: round(v, 4) for k, v in baseline_dist[agent_name].items()}),
                    }
                )
            write_full_csv(
                drift_path,
                drift_rows,
                [
                    "timestamp",
                    "checkpoint",
                    "agent_name",
                    "drift_score",
                    "above_threshold",
                    "current_dist",
                    "baseline_dist",
                ],
            )

            baseline_rows = rows[: min(50, len(rows))]
            current_rows = rows[: len(completed)]
            concept_rows.append(
                {
                    "timestamp": now_iso(),
                    "checkpoint": len(completed),
                    "feature_kl_shift": concept_feature_shift(current_rows, baseline_rows),
                    "baseline_window": min(50, len(rows)),
                    "current_window": len(completed),
                }
            )
            write_full_csv(
                concept_path,
                concept_rows,
                ["timestamp", "checkpoint", "feature_kl_shift", "baseline_window", "current_window"],
            )

        print(
            f"  result router={router_decision} borda={borda_winner} "
            f"consens={consens_winner} hierarchy={hierarchy_final} expected={expected}",
            flush=True,
        )

        if time.time() - last_status > STATUS_EVERY_SECONDS:
            write_status(len(completed), len(rows), borrower_id, started_at, summary_path)
            print_eta(run_start, completed_this_run, len(remaining_indices))
            aggregate_and_save_charts(summary_path, drift_path, concept_path)
            last_status = time.time()

    write_status(len(completed), len(rows), "complete", started_at, summary_path)
    aggregate_and_save_charts(summary_path, drift_path, concept_path)
    print("\nRobustness run complete or fully resumed.", flush=True)


if __name__ == "__main__":
    run()
