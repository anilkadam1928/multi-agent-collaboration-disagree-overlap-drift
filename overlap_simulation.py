from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

from config import AGENT_NAMES, BASELINE_SAMPLE_SIZE, OVERLAP_RESULTS_PATH
from data_loader import get_profiles, load_dataset
from output_parser import parse_all
from profile_builder import build_borrower_profile


# This module models asynchronous serial overlap in a staged loan-review pipeline.
# Agents are not acting in simultaneous parallel mode. Instead, each stage happens
# in sequence, and the active leader decides which workers should act next.

OVERLAP_PROFILE_COUNT = int(os.getenv("OVERLAP_COUNT", "3"))
FAST_NUM_PREDICT = int(os.getenv("OVERLAP_NUM_PREDICT", "160"))

FORMAT_RULE = (
    "Return the answer using exactly this structure:\n"
    "Decision: Approve, Reject, or Refer\n"
    "Confidence: high, medium, or low\n"
    "Brief reason: one short sentence."
)

OUTPUT_COLUMNS = [
    "run_id",
    "borrower_id",
    "kaggle_truth_label",
    "active_leader",
    "leadership_scores_json",
    "initial_decisions_json",
    "initial_confidences_json",
    "task_assignments_json",
]

SPECIALIST_TASKS = {
    "RouterManager": "Coordinate the review order and decide which worker acts next.",
    "IncomeAgent": "Assess repayment capacity from the available financial proxy fields only.",
    "FraudAgent": "Check for anomaly or suspicious pattern signals from structured fields only.",
    "CreditAgent": "Assess lending risk and repayment burden from the available credit fields.",
    "ComplianceAgent": "Check policy consistency and rule-fit from the available structured fields.",
    "SummariserAgent": "Prepare a concise decision memo after specialists complete their work.",
    "WeakModelAgent": "Provide a lightweight comparison review to expose blind spots.",
}


def get_agents():
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


def clean_output(text: str) -> str:
    text = str(text or "").strip()
    if "Final Answer:" in text:
        return text.split("Final Answer:", 1)[-1].strip()
    return text


def explicit_confidence(text: str) -> str:
    match = re.search(r"confidence:\s*(high|medium|low)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    parsed = parse_all(text).get("confidence_flag", "unknown")
    return parsed if parsed in {"high", "medium", "low"} else "unknown"


def confidence_to_score(confidence_flag: str) -> float:
    return {
        "high": 1.0,
        "medium": 0.6,
        "low": 0.3,
        "unknown": 0.1,
    }.get((confidence_flag or "unknown").lower(), 0.1)


def compact_for_log(text: str, limit: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + " ..."


def run_agent_once(agent_name: str, agent, profile_text: str) -> dict:
    task_prompt = (
        f"You are the {agent_name}. "
        "This is the bootstrap stage for LCE leader assignment.\n\n"
        f"{profile_text}\n\n"
        "Give your specialist initial assessment using only your own domain.\n"
        f"{FORMAT_RULE}"
    )

    direct_prompt = (
        f"Role: {agent.role}\n"
        f"Goal: {agent.goal}\n"
        "Act exactly in this specialist role. Use only the dataset fields provided.\n"
        "Do not output a thinking process or hidden reasoning.\n"
        "Start immediately with `Decision:` and follow the required format.\n\n"
        f"{task_prompt}"
    )

    if hasattr(agent.llm, "_call"):
        raw_output = clean_output(
            agent.llm._call(
                direct_prompt,
                num_predict=FAST_NUM_PREDICT,
                temperature=0.0,
            )
        )
    else:
        raw_output = clean_output(agent.llm.invoke(direct_prompt))

    parsed = parse_all(raw_output)
    confidence_flag = explicit_confidence(raw_output)
    if parsed["confidence_flag"] == "unknown":
        parsed["confidence_flag"] = confidence_flag

    return {
        "agent_name": agent_name,
        "raw_output": raw_output,
        "parsed_decision": parsed["parsed_decision"],
        "confidence_flag": confidence_flag,
    }


def bootstrap_last_outputs(agents: dict[str, object], profile_text: str) -> dict[str, dict]:
    outputs: dict[str, dict] = {}
    for agent_name in AGENT_NAMES:
        print(f"  Bootstrapping {agent_name}...", flush=True)
        outputs[agent_name] = run_agent_once(agent_name, agents[agent_name], profile_text)
        print(
            f"    {agent_name}: {outputs[agent_name]['parsed_decision']} | "
            f"{outputs[agent_name]['confidence_flag']}",
            flush=True,
        )
        print(f"    Raw: {compact_for_log(outputs[agent_name]['raw_output'])}", flush=True)
    return outputs


def compute_leadership_scores(last_outputs: dict[str, dict]) -> dict[str, float]:
    return {
        agent_name: confidence_to_score(last_outputs[agent_name]["confidence_flag"])
        for agent_name in AGENT_NAMES
    }


def select_active_leader(last_outputs: dict[str, dict]) -> tuple[str, dict[str, float]]:
    leadership_scores = compute_leadership_scores(last_outputs)
    active_leader = max(AGENT_NAMES, key=lambda agent_name: leadership_scores[agent_name])
    return active_leader, leadership_scores


def build_leader_task_assignments(active_leader: str, borrower_id: str) -> dict[str, str]:
    assignments = {}
    for agent_name in AGENT_NAMES:
        if agent_name == active_leader:
            assignments[agent_name] = (
                f"Lead Borrower {borrower_id}: assign the next worker task and prevent "
                "workers from independently initiating overlapping checks."
            )
        else:
            assignments[agent_name] = SPECIALIST_TASKS[agent_name]
    return assignments


def load_overlap_profiles(sample_size: int) -> list[dict]:
    rows = load_dataset()
    selected_rows = get_profiles(rows, BASELINE_SAMPLE_SIZE)[:sample_size]

    profiles = []
    for index, row in enumerate(selected_rows, start=1):
        profile_text, truth = build_borrower_profile(row)
        profiles.append(
            {
                "borrower_id": f"B{index:03d}",
                "profile_text": profile_text,
                "kaggle_truth_label": truth,
            }
        )
    return profiles


def save_overlap_rows(file_path: str, rows: list[dict]) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {file_path}", flush=True)


def run_lce_overlap_bootstrap(sample_size: int = OVERLAP_PROFILE_COUNT) -> list[dict]:
    print("Starting overlap_simulation.py (Hour 1: LCE leader assignment)...", flush=True)
    agents = get_agents()
    profiles = load_overlap_profiles(sample_size)
    print(f"Loaded {len(profiles)} overlap profiles.", flush=True)

    rows: list[dict] = []

    for profile in profiles:
        borrower_id = profile["borrower_id"]
        print(f"\nBorrower {borrower_id}", flush=True)
        print(profile["profile_text"], flush=True)

        last_outputs = bootstrap_last_outputs(agents, profile["profile_text"])
        active_leader, leadership_scores = select_active_leader(last_outputs)
        task_assignments = build_leader_task_assignments(active_leader, borrower_id)

        print(f"  Leadership scores: {leadership_scores}", flush=True)
        print(f"  Active leader: {active_leader}", flush=True)
        print(f"  Leader task board: {task_assignments}", flush=True)

        rows.append(
            {
                "run_id": f"overlap_{borrower_id}_lce_bootstrap",
                "borrower_id": borrower_id,
                "kaggle_truth_label": profile["kaggle_truth_label"],
                "active_leader": active_leader,
                "leadership_scores_json": json.dumps(leadership_scores),
                "initial_decisions_json": json.dumps(
                    {agent_name: last_outputs[agent_name]["parsed_decision"] for agent_name in AGENT_NAMES}
                ),
                "initial_confidences_json": json.dumps(
                    {agent_name: last_outputs[agent_name]["confidence_flag"] for agent_name in AGENT_NAMES}
                ),
                "task_assignments_json": json.dumps(task_assignments),
            }
        )

    save_overlap_rows(OVERLAP_RESULTS_PATH, rows)
    return rows


if __name__ == "__main__":
    run_lce_overlap_bootstrap()
