from __future__ import annotations

"""Week 4 - Scenario 3: Agent Drift Module (Day 1 scaffold).

This file is intentionally self-contained and only depends on the repo's
existing Week 3 modules. It does not modify any shared simulation code.

By default it runs a 1-profile smoke test so you can confirm the drift
pipeline starts correctly. Full EMC and drift-comparison runs are available
through environment variables.
"""

import csv
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

from agents import (
    compliance_agent,
    credit_agent,
    fraud_agent,
    income_agent,
    router_manager,
    summariser_agent,
    weak_model_agent,
)
from config import BASELINE_RESULTS_PATH, DRIFT_KL_THRESHOLD
from data_loader import get_profiles, load_dataset
from output_parser import parse_confidence, parse_decision
from profile_builder import build_borrower_profile


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BASELINE_CSV = (PROJECT_ROOT / BASELINE_RESULTS_PATH).resolve()
EMC_RESULTS_PATH = PROJECT_ROOT / "data" / "emc_test_results.csv"
DRIFT_NO_EMC_PATH = PROJECT_ROOT / "data" / "drift_no_emc.csv"
DRIFT_WITH_EMC_PATH = PROJECT_ROOT / "data" / "drift_with_emc.csv"

FAST_NUM_PREDICT = int(os.getenv("DRIFT_NUM_PREDICT", "180"))
DEFAULT_PROFILE_COUNT = int(os.getenv("DRIFT_PROFILE_COUNT", "1"))
DEFAULT_N_RUNS = int(os.getenv("DRIFT_RUNS", "1"))
EMC_TRIGGER_EVERY = int(os.getenv("DRIFT_EMC_TRIGGER", "20"))
CHECKPOINTS = tuple(
    int(part.strip())
    for part in os.getenv("DRIFT_CHECKPOINTS", "20,50,100").split(",")
    if part.strip()
)

FORMAT_RULE = (
    "Return the answer using exactly this structure:\n"
    "Decision: Approve, Reject, or Refer\n"
    "Confidence: high, medium, or low\n"
    "Brief reason: one short sentence."
)

DECISION_RE = re.compile(r"decision:\s*(approve|reject|refer)", re.IGNORECASE)
CONFIDENCE_RE = re.compile(r"confidence:\s*(high|medium|low)", re.IGNORECASE)
REASON_RE = re.compile(r"brief reason:\s*(.+)", re.IGNORECASE | re.DOTALL)

run_counter = {"count": 0}
agent_memory: dict[str, list[str]] = {}


def get_all_agents():
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
    cleaned = str(text or "").strip()
    if "Final Answer:" in cleaned:
        cleaned = cleaned.split("Final Answer:", 1)[-1].strip()
    return cleaned


def call_agent_llm(agent, prompt: str, temperature: float = 0.0, num_predict: int = FAST_NUM_PREDICT) -> str:
    if hasattr(agent.llm, "_call"):
        return clean_output(
            agent.llm._call(
                prompt,
                num_predict=num_predict,
                temperature=temperature,
            )
        )
    return clean_output(agent.llm.invoke(prompt))


def extract_structured_fields(text: str) -> dict[str, str]:
    cleaned = clean_output(text)
    decision_match = DECISION_RE.search(cleaned)
    confidence_match = CONFIDENCE_RE.search(cleaned)
    reason_match = REASON_RE.search(cleaned)

    decision = decision_match.group(1).lower() if decision_match else parse_decision(cleaned)
    confidence = confidence_match.group(1).lower() if confidence_match else parse_confidence(cleaned)

    if reason_match:
        reason = re.sub(r"\s+", " ", reason_match.group(1)).strip()
    else:
        reason = re.sub(r"\s+", " ", cleaned).strip()

    if len(reason) > 220:
        reason = reason[:220].rstrip() + " ..."

    return {
        "decision": decision,
        "confidence": confidence,
        "reason": reason or "No clear reason provided",
    }


def repair_output_if_needed(agent_name: str, agent, raw_output: str) -> str:
    fields = extract_structured_fields(raw_output)
    if fields["decision"] != "unknown" and fields["confidence"] != "unknown":
        return raw_output

    repair_prompt = (
        f"You are reformatting the {agent_name} draft below.\n"
        "Do not add new analysis. Infer the closest structured answer from the draft.\n"
        f"{FORMAT_RULE}\n\n"
        f"DRAFT:\n{raw_output}"
    )
    repaired = call_agent_llm(agent, repair_prompt, temperature=0.0, num_predict=120)
    repaired_fields = extract_structured_fields(repaired)
    if repaired_fields["decision"] != "unknown":
        return repaired
    return raw_output


def record_to_memory(agent_name: str, borrower_id: str, decision_text: str):
    if agent_name not in agent_memory:
        agent_memory[agent_name] = []
    agent_memory[agent_name].append(f"{borrower_id}: {decision_text}")


def memory_text_for(agent_name: str) -> str:
    history = agent_memory.get(agent_name, [])
    if not history:
        return ""
    recent = history[-8:]
    return "Recent memory:\n" + "\n".join(f"- {item}" for item in recent)


def run_agent_once(agent_name: str, agent, borrower_id: str, profile_text: str, memory_text: str = "") -> dict[str, str]:
    prompt_parts = [
        f"You are the {agent_name}.",
        "This is a sequential drift simulation on the German Credit research dataset.",
        "Use only the structured fields provided. Do not ask for external documents.",
        "Do not output a thinking process or hidden reasoning.",
    ]
    if memory_text:
        prompt_parts.append(memory_text)
    prompt_parts.extend(
        [
            f"Borrower ID: {borrower_id}",
            profile_text,
            FORMAT_RULE,
        ]
    )
    prompt = "\n\n".join(prompt_parts)

    raw_output = call_agent_llm(agent, prompt, temperature=0.0)
    raw_output = repair_output_if_needed(agent_name, agent, raw_output)
    fields = extract_structured_fields(raw_output)

    return {
        "raw_output": raw_output,
        "parsed_decision": fields["decision"],
        "confidence_flag": fields["confidence"],
        "brief_reason": fields["reason"],
    }


def load_drift_profiles(n: int = 50) -> list[dict[str, str]]:
    rows = load_dataset()
    return get_profiles(rows, n)


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_sequential_pipeline(profiles: list[dict[str, str]], n_runs: int = 1, use_memory: bool = True) -> list[dict]:
    """Run all 7 agents on each profile for n_runs iterations."""
    agents = get_all_agents()
    all_results: list[dict] = []

    for run_number in range(1, n_runs + 1):
        for profile_index, row in enumerate(profiles, start=1):
            borrower_id = f"B{profile_index:03d}"
            profile_text, truth = build_borrower_profile(row)

            for agent_name, agent in agents.items():
                run_counter["count"] += 1
                result = run_agent_once(
                    agent_name,
                    agent,
                    borrower_id,
                    profile_text,
                    memory_text_for(agent_name) if use_memory else "",
                )

                record_to_memory(
                    agent_name,
                    borrower_id,
                    f"decision={result['parsed_decision']} | confidence={result['confidence_flag']} | reason={result['brief_reason']}",
                )

                row_result = {
                    "run_id": run_counter["count"],
                    "run_number": run_number,
                    "borrower_id": borrower_id,
                    "kaggle_truth_label": truth.lower(),
                    "agent_name": agent_name,
                    "raw_output": result["raw_output"],
                    "parsed_decision": result["parsed_decision"],
                    "confidence_flag": result["confidence_flag"],
                    "brief_reason": result["brief_reason"],
                }
                all_results.append(row_result)
                print(
                    f"  Run {run_counter['count']} | Borrower {borrower_id} | "
                    f"{agent_name} -> {row_result['parsed_decision']} | {row_result['confidence_flag']}"
                )

    return all_results


def summarize_history_with_emc(agent_name: str, history: list[str]) -> str:
    if not history:
        return ""

    summarizer_prompt = (
        "You are compressing agent memory for drift control.\n"
        "Summarize the following history into exactly 3 short sentences.\n"
        "Keep decision tendencies, confidence pattern, and any persistent risk themes.\n"
        "Do not include bullet points.\n\n"
        f"Agent: {agent_name}\n"
        "History:\n"
        + "\n".join(f"- {item}" for item in history[-20:])
    )

    try:
        summary = call_agent_llm(summariser_agent, summarizer_prompt, temperature=0.0, num_predict=140)
        summary = re.sub(r"\s+", " ", summary).strip()
        if summary:
            return summary
    except Exception:
        pass

    counts = Counter()
    for item in history[-20:]:
        lowered = item.lower()
        if "approve" in lowered:
            counts["approve"] += 1
        elif "reject" in lowered:
            counts["reject"] += 1
        elif "refer" in lowered:
            counts["refer"] += 1
        else:
            counts["unknown"] += 1

    return (
        f"{agent_name} processed {len(history[-20:])} recent cases. "
        f"Approvals={counts['approve']}, refers={counts['refer']}, rejects={counts['reject']}, unknown={counts['unknown']}. "
        "This condensed summary replaces the longer raw history for future prompts."
    )


def emc_consolidation(trigger_run: int) -> list[dict]:
    """Compress each agent's memory to a concise summary."""
    print(f"\n>>> EMC triggered after borrower {trigger_run}.")
    rows: list[dict] = []

    for agent_name, history in list(agent_memory.items()):
        if not history:
            continue

        words_before = sum(len(item.split()) for item in history)
        summary = summarize_history_with_emc(agent_name, history)
        words_after = len(summary.split())
        compression_ratio = round(words_before / max(words_after, 1), 2)

        agent_memory[agent_name] = [summary]
        rows.append(
            {
                "run_id": trigger_run,
                "agent_name": agent_name,
                "words_before": words_before,
                "words_after": words_after,
                "compression_ratio": compression_ratio,
                "summary": summary,
            }
        )
        print(f"   {agent_name}: {words_before}w -> {words_after}w (ratio {compression_ratio}x)")

    return rows


def run_emc_test(profiles: list[dict[str, str]], output_csv: Path = EMC_RESULTS_PATH) -> list[dict]:
    agents = get_all_agents()
    agent_memory.clear()
    emc_rows: list[dict] = []

    for profile_index, row in enumerate(profiles, start=1):
        borrower_id = f"B{profile_index:03d}"
        profile_text, _truth = build_borrower_profile(row)
        print(f"  EMC borrower {borrower_id} ({profile_index}/{len(profiles)})")

        for agent_name, agent in agents.items():
            result = run_agent_once(
                agent_name,
                agent,
                borrower_id,
                profile_text,
                memory_text_for(agent_name),
            )
            record_to_memory(
                agent_name,
                borrower_id,
                f"decision={result['parsed_decision']} | confidence={result['confidence_flag']} | reason={result['brief_reason']}",
            )
            print(
                f"    {agent_name} -> {result['parsed_decision']} | {result['confidence_flag']}"
            )

        if profile_index % EMC_TRIGGER_EVERY == 0:
            emc_rows.extend(emc_consolidation(trigger_run=profile_index))

    write_rows(
        output_csv,
        emc_rows,
        ["run_id", "agent_name", "words_before", "words_after", "compression_ratio", "summary"],
    )
    print(f"\n✓ EMC test complete. Results saved to {output_csv}")
    return emc_rows


def get_baseline_distribution(agent_name: str, baseline_csv: Path = DEFAULT_BASELINE_CSV) -> dict[str, float]:
    counts = Counter({"approve": 0, "refer": 0, "reject": 0})
    total = 0

    with baseline_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("agent_name") != agent_name:
                continue
            counts[row.get("parsed_decision", "unknown").lower()] += 1
            total += 1

    if total == 0:
        total = 1

    return {
        label: (counts[label] + 1e-6) / (total + 3e-6)
        for label in ("approve", "refer", "reject")
    }


def calculate_drift_score(agent_name: str, current_results: list[dict], baseline_csv: Path = DEFAULT_BASELINE_CSV) -> float:
    baseline_dist = get_baseline_distribution(agent_name, baseline_csv)
    agent_rows = [row for row in current_results if row["agent_name"] == agent_name]
    counts = Counter(row["parsed_decision"] for row in agent_rows)
    total = max(len(agent_rows), 1)

    current_dist = {
        label: (counts.get(label, 0) + 1e-6) / (total + 3e-6)
        for label in ("approve", "refer", "reject")
    }

    kl = 0.0
    for label in ("approve", "refer", "reject"):
        kl += current_dist[label] * math.log(current_dist[label] / baseline_dist[label])

    return round(kl, 4)


def run_drift_comparison(profiles: list[dict[str, str]], output_prefix: str = "drift") -> dict[str, Path]:
    results_map = {
        "no_emc": DRIFT_NO_EMC_PATH,
        "with_emc": DRIFT_WITH_EMC_PATH,
    }
    agents = list(get_all_agents().keys())

    for mode in ("no_emc", "with_emc"):
        print(f"\n{'=' * 56}")
        print(f"MODE: {mode.upper()}")
        print(f"{'=' * 56}")

        agent_memory.clear()
        current_results: list[dict] = []
        drift_rows: list[dict] = []
        agents_obj = get_all_agents()

        for profile_index, row in enumerate(profiles, start=1):
            borrower_id = f"B{profile_index:03d}"
            profile_text, truth = build_borrower_profile(row)
            print(f"  {mode} borrower {borrower_id} ({profile_index}/{len(profiles)})")

            for agent_name, agent in agents_obj.items():
                result = run_agent_once(
                    agent_name,
                    agent,
                    borrower_id,
                    profile_text,
                    memory_text_for(agent_name),
                )
                current_results.append(
                    {
                        "borrower_id": borrower_id,
                        "kaggle_truth_label": truth.lower(),
                        "agent_name": agent_name,
                        "parsed_decision": result["parsed_decision"],
                        "confidence_flag": result["confidence_flag"],
                    }
                )
                record_to_memory(
                    agent_name,
                    borrower_id,
                    f"decision={result['parsed_decision']} | confidence={result['confidence_flag']} | reason={result['brief_reason']}",
                )
                print(
                    f"    {agent_name} -> {result['parsed_decision']} | {result['confidence_flag']}"
                )

            if mode == "with_emc" and profile_index % EMC_TRIGGER_EVERY == 0:
                emc_consolidation(trigger_run=profile_index)

            if profile_index in CHECKPOINTS:
                print(f"\n--- Drift check at borrower {profile_index} ---")
                for agent_name in agents:
                    score = calculate_drift_score(agent_name, current_results)
                    drift_rows.append(
                        {
                            "mode": mode,
                            "checkpoint": profile_index,
                            "agent_name": agent_name,
                            "drift_score": score,
                        }
                    )
                    status = "ABOVE THRESHOLD" if score > DRIFT_KL_THRESHOLD else "below threshold"
                    print(f"  {agent_name}: {score} ({status})")

        write_rows(results_map[mode], drift_rows, ["mode", "checkpoint", "agent_name", "drift_score"])
        print(f"\n✓ Saved {results_map[mode]}")

    return results_map


def print_mean_drift_summary():
    if not DRIFT_NO_EMC_PATH.exists() or not DRIFT_WITH_EMC_PATH.exists():
        return

    def mean_for_latest_checkpoint(path: Path) -> tuple[int | None, float]:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return None, 0.0
        latest = max(int(row["checkpoint"]) for row in rows)
        scores = [float(row["drift_score"]) for row in rows if int(row["checkpoint"]) == latest]
        return latest, round(sum(scores) / max(len(scores), 1), 4)

    no_checkpoint, no_mean = mean_for_latest_checkpoint(DRIFT_NO_EMC_PATH)
    with_checkpoint, with_mean = mean_for_latest_checkpoint(DRIFT_WITH_EMC_PATH)

    if no_checkpoint is None or with_checkpoint is None:
        return

    reduction = round((1 - (with_mean / max(no_mean, 0.001))) * 100, 1)
    print(f"\n{'=' * 56}")
    print(f"Without EMC — mean drift at checkpoint {no_checkpoint}: {no_mean}")
    print(f"With EMC    — mean drift at checkpoint {with_checkpoint}: {with_mean}")
    print(f"EMC reduced drift by {reduction}%")
    print(f"{'=' * 56}")


if __name__ == "__main__":
    if os.getenv("DRIFT_SKIP_SCAFFOLD") != "1":
        print("=== DRIFT SIMULATION — SCAFFOLD TEST ===")
        profiles = load_drift_profiles(DEFAULT_PROFILE_COUNT)
        results = run_sequential_pipeline(profiles, n_runs=DEFAULT_N_RUNS, use_memory=True)
        print(f"\n✓ Scaffold working. {len(results)} agent outputs captured.")
        for row in results:
            print(f"  {row['borrower_id']} | {row['agent_name']}: {row['parsed_decision']} ({row['confidence_flag']})")

    if os.getenv("DRIFT_RUN_EMC_TEST") == "1":
        print("\n=== EMC TEST ===")
        emc_profiles = load_drift_profiles(int(os.getenv("DRIFT_EMC_PROFILE_COUNT", "20")))
        run_emc_test(emc_profiles)

    if os.getenv("DRIFT_RUN_COMPARISON") == "1":
        print("\n=== DRIFT COMPARISON ===")
        comparison_profiles = load_drift_profiles(int(os.getenv("DRIFT_COMPARISON_PROFILE_COUNT", "50")))
        run_drift_comparison(comparison_profiles)
        print_mean_drift_summary()
