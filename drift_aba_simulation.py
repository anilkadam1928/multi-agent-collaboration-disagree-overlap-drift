from __future__ import annotations

"""Week 4 Tuesday - ABA + Drift-Aware Routing runner.

This is a separate Tuesday file. It imports Monday's drift helpers but does
not edit them. If this experiment misbehaves, your Week 3 and Monday files
remain untouched.

Implements:
- Adaptive Behaviour Anchoring (ABA) when KL drift is above threshold.
- EMC memory compaction during the same run.
- Drift-aware routing logs using the currently least-drifted agent.
- CSV outputs and a comparison chart.
"""

import csv
import os
from collections import Counter
from pathlib import Path

from anchor_examples import get_anchor_examples
from config import DRIFT_KL_THRESHOLD
from drift_simulation import (
    DEFAULT_BASELINE_CSV,
    DRIFT_NO_EMC_PATH,
    DRIFT_WITH_EMC_PATH,
    FORMAT_RULE,
    PROJECT_ROOT,
    agent_memory,
    calculate_drift_score,
    call_agent_llm,
    emc_consolidation,
    extract_structured_fields,
    get_all_agents,
    load_drift_profiles,
    memory_text_for,
    record_to_memory,
    repair_output_if_needed,
    write_rows,
)
from profile_builder import build_borrower_profile


DRIFT_WITH_ABA_PATH = PROJECT_ROOT / "data" / "drift_with_aba.csv"
DRIFT_WITH_ABA_EVENTS_PATH = PROJECT_ROOT / "data" / "drift_with_aba_events.csv"
ABA_FIGURE_PATH = PROJECT_ROOT / "results" / "Figure_3_Drift_Score_With_ABA.png"

ABA_PROFILE_COUNT = int(os.getenv("DRIFT_ABA_PROFILE_COUNT", "40"))
ABA_ANCHOR_LIMIT = int(os.getenv("DRIFT_ABA_ANCHOR_LIMIT", "6"))
ABA_THRESHOLD = float(os.getenv("DRIFT_ABA_THRESHOLD", str(DRIFT_KL_THRESHOLD)))
ABA_EMC_TRIGGER = int(os.getenv("DRIFT_ABA_EMC_TRIGGER", os.getenv("DRIFT_EMC_TRIGGER", "20")))
ABA_PROGRESS_EVERY = int(os.getenv("DRIFT_ABA_PROGRESS_EVERY", "1"))

ABA_CHECKPOINTS = tuple(
    int(part.strip())
    for part in os.getenv("DRIFT_ABA_CHECKPOINTS", "20,40").split(",")
    if part.strip()
)


def format_anchor_examples(limit: int = ABA_ANCHOR_LIMIT) -> str:
    """Keep the injected anchors short so prompts do not become huge."""
    approve_examples = get_anchor_examples(limit=max(1, limit // 2), decision="approve")
    reject_examples = get_anchor_examples(limit=max(1, limit - len(approve_examples)), decision="reject")
    selected = approve_examples + reject_examples

    lines = ["Anchor calibration examples:"]
    for item in selected[:limit]:
        keywords = ", ".join(str(keyword) for keyword in item["reasoning_keywords"])
        lines.append(
            f"- {item['borrower_id']}: correct_decision={item['correct_decision']}; "
            f"truth={item['kaggle_truth_label']}; keywords={keywords}"
        )
    return "\n".join(lines)


def build_aba_block(agent_name: str, drift_score: float) -> str:
    return (
        "ADAPTIVE BEHAVIOUR ANCHORING ACTIVE.\n"
        f"Agent: {agent_name}\n"
        f"Recent KL drift score: {drift_score:.4f}; threshold: {ABA_THRESHOLD:.4f}.\n"
        "Your recent decision distribution is drifting from the frozen baseline.\n"
        "Use these anchor examples to recalibrate your decision standard. Do not copy them.\n"
        "Apply the same standard to the current borrower only.\n\n"
        f"{format_anchor_examples()}"
    )


def build_agent_prompt(
    agent_name: str,
    borrower_id: str,
    profile_text: str,
    routed_to_agent: str,
    drift_score: float,
    aba_triggered: bool,
) -> str:
    prompt_parts = [
        f"You are the {agent_name}.",
        "This is Week 4 Tuesday Scenario 3: Adaptive Behaviour Anchoring.",
        "Use only the structured German Credit dataset fields provided.",
        "Do not ask for external documents. Do not output hidden thinking.",
        f"Current drift-aware route leader: {routed_to_agent}.",
    ]

    memory_text = memory_text_for(agent_name)
    if memory_text:
        prompt_parts.append(memory_text)

    if aba_triggered:
        prompt_parts.append(build_aba_block(agent_name, drift_score))

    prompt_parts.extend(
        [
            f"Borrower ID: {borrower_id}",
            profile_text,
            FORMAT_RULE,
        ]
    )
    return "\n\n".join(prompt_parts)


def choose_least_drifted_agent(drift_leaderboard: dict[str, float]) -> str:
    return min(drift_leaderboard, key=lambda name: (drift_leaderboard[name], name))


def run_agent_with_aba(
    agent_name: str,
    agent,
    borrower_id: str,
    profile_text: str,
    routed_to_agent: str,
    drift_score: float,
) -> dict[str, object]:
    aba_triggered = drift_score > ABA_THRESHOLD
    prompt = build_agent_prompt(
        agent_name=agent_name,
        borrower_id=borrower_id,
        profile_text=profile_text,
        routed_to_agent=routed_to_agent,
        drift_score=drift_score,
        aba_triggered=aba_triggered,
    )

    raw_output = call_agent_llm(agent, prompt, temperature=0.0)
    raw_output = repair_output_if_needed(agent_name, agent, raw_output)
    fields = extract_structured_fields(raw_output)

    return {
        "raw_output": raw_output,
        "parsed_decision": fields["decision"],
        "confidence_flag": fields["confidence"],
        "brief_reason": fields["reason"],
        "aba_triggered": aba_triggered,
        "pre_call_drift_score": drift_score,
    }


def read_mean_drift_by_checkpoint(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}

    grouped: dict[int, list[float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            checkpoint = int(row.get("checkpoint") or row.get("run_batch") or 0)
            if checkpoint <= 0:
                continue
            grouped.setdefault(checkpoint, []).append(float(row["drift_score"]))

    return {
        checkpoint: round(sum(scores) / max(len(scores), 1), 4)
        for checkpoint, scores in sorted(grouped.items())
    }


def generate_aba_chart() -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Could not create chart because matplotlib is unavailable: {exc}")
        return

    series = {
        "Without EMC": read_mean_drift_by_checkpoint(DRIFT_NO_EMC_PATH),
        "With EMC": read_mean_drift_by_checkpoint(DRIFT_WITH_EMC_PATH),
        "With EMC + ABA": read_mean_drift_by_checkpoint(DRIFT_WITH_ABA_PATH),
    }

    if not any(series.values()):
        print("No drift CSV data available yet, so chart was not created.")
        return

    ABA_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, 5.0))

    for label, data in series.items():
        if not data:
            continue
        checkpoints = list(data.keys())
        values = list(data.values())
        plt.plot(checkpoints, values, marker="o", linewidth=2.2, label=label)

    plt.axhline(ABA_THRESHOLD, color="crimson", linestyle="--", linewidth=1.5, label="KL threshold")
    plt.title("Week 4 Drift Score Comparison")
    plt.xlabel("Borrower checkpoint")
    plt.ylabel("Mean KL drift score")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ABA_FIGURE_PATH, dpi=200)
    plt.close()
    print(f"Saved chart: {ABA_FIGURE_PATH}")


def print_latest_summary() -> None:
    no_emc = read_mean_drift_by_checkpoint(DRIFT_NO_EMC_PATH)
    with_emc = read_mean_drift_by_checkpoint(DRIFT_WITH_EMC_PATH)
    with_aba = read_mean_drift_by_checkpoint(DRIFT_WITH_ABA_PATH)

    if not with_aba:
        return

    latest = max(with_aba)
    print("\n" + "=" * 64)
    print("TUESDAY ABA DRIFT SUMMARY")
    if latest in no_emc:
        print(f"Without EMC       at checkpoint {latest}: {no_emc[latest]:.4f}")
    if latest in with_emc:
        print(f"With EMC          at checkpoint {latest}: {with_emc[latest]:.4f}")
    print(f"With EMC + ABA    at checkpoint {latest}: {with_aba[latest]:.4f}")

    if latest in with_emc and with_emc[latest] > 0:
        reduction = (1 - (with_aba[latest] / with_emc[latest])) * 100
        print(f"ABA change vs EMC-only: {reduction:.1f}%")
    print("=" * 64)


def run_aba_simulation(profiles: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    agents = get_all_agents()
    agent_names = list(agents.keys())
    checkpoint_set = set(ABA_CHECKPOINTS)

    agent_memory.clear()
    current_results: list[dict] = []
    event_rows: list[dict] = []
    drift_rows: list[dict] = []
    drift_leaderboard = {agent_name: 0.0 for agent_name in agent_names}
    aba_counts: Counter[str] = Counter()
    emc_triggered_at: set[int] = set()

    print("=== ABA DRIFT SIMULATION ===")
    print(f"Profiles: {len(profiles)}")
    print(f"Checkpoints: {ABA_CHECKPOINTS}")
    print(f"ABA threshold: {ABA_THRESHOLD}")
    print(f"EMC trigger: every {ABA_EMC_TRIGGER} borrowers")

    for profile_index, row in enumerate(profiles, start=1):
        borrower_id = f"B{profile_index:03d}"
        profile_text, truth = build_borrower_profile(row)
        routed_to_agent = choose_least_drifted_agent(drift_leaderboard)

        if profile_index == 1 or profile_index % ABA_PROGRESS_EVERY == 0:
            print(
                f"\nABA borrower {borrower_id} ({profile_index}/{len(profiles)}) | "
                f"route={routed_to_agent} | route_drift={drift_leaderboard[routed_to_agent]:.4f}"
            )

        for agent_name, agent in agents.items():
            result = run_agent_with_aba(
                agent_name=agent_name,
                agent=agent,
                borrower_id=borrower_id,
                profile_text=profile_text,
                routed_to_agent=routed_to_agent,
                drift_score=drift_leaderboard[agent_name],
            )

            if result["aba_triggered"]:
                aba_counts[agent_name] += 1

            current_results.append(
                {
                    "borrower_id": borrower_id,
                    "kaggle_truth_label": str(truth).lower(),
                    "agent_name": agent_name,
                    "parsed_decision": result["parsed_decision"],
                    "confidence_flag": result["confidence_flag"],
                }
            )
            record_to_memory(
                agent_name,
                borrower_id,
                (
                    f"decision={result['parsed_decision']} | "
                    f"confidence={result['confidence_flag']} | "
                    f"reason={result['brief_reason']}"
                ),
            )

            event_rows.append(
                {
                    "borrower_id": borrower_id,
                    "kaggle_truth_label": str(truth).lower(),
                    "agent_name": agent_name,
                    "routed_to_agent": routed_to_agent,
                    "pre_call_drift_score": result["pre_call_drift_score"],
                    "aba_triggered": result["aba_triggered"],
                    "parsed_decision": result["parsed_decision"],
                    "confidence_flag": result["confidence_flag"],
                    "brief_reason": result["brief_reason"],
                }
            )

            print(
                f"  {agent_name} -> {result['parsed_decision']} | "
                f"{result['confidence_flag']} | ABA={result['aba_triggered']}"
            )

        if profile_index % ABA_EMC_TRIGGER == 0:
            emc_consolidation(trigger_run=profile_index)
            emc_triggered_at.add(profile_index)

        if profile_index in checkpoint_set:
            print(f"\n--- ABA drift check at borrower {profile_index} ---")
            for agent_name in agent_names:
                score = calculate_drift_score(agent_name, current_results)
                drift_leaderboard[agent_name] = score
                status = "ABOVE THRESHOLD" if score > ABA_THRESHOLD else "below threshold"
                drift_rows.append(
                    {
                        "run_batch": profile_index,
                        "checkpoint": profile_index,
                        "agent_name": agent_name,
                        "drift_score": score,
                        "aba_triggered": aba_counts[agent_name] > 0,
                        "aba_trigger_count": aba_counts[agent_name],
                        "emc_triggered": profile_index in emc_triggered_at,
                        "routed_to_agent": choose_least_drifted_agent(drift_leaderboard),
                        "threshold": ABA_THRESHOLD,
                    }
                )
                print(f"  {agent_name}: {score} ({status}); ABA triggers={aba_counts[agent_name]}")

            next_route = choose_least_drifted_agent(drift_leaderboard)
            print(f"  Next drift-aware route leader: {next_route}")

    write_rows(
        DRIFT_WITH_ABA_PATH,
        drift_rows,
        [
            "run_batch",
            "checkpoint",
            "agent_name",
            "drift_score",
            "aba_triggered",
            "aba_trigger_count",
            "emc_triggered",
            "routed_to_agent",
            "threshold",
        ],
    )
    write_rows(
        DRIFT_WITH_ABA_EVENTS_PATH,
        event_rows,
        [
            "borrower_id",
            "kaggle_truth_label",
            "agent_name",
            "routed_to_agent",
            "pre_call_drift_score",
            "aba_triggered",
            "parsed_decision",
            "confidence_flag",
            "brief_reason",
        ],
    )

    print(f"\nSaved drift CSV: {DRIFT_WITH_ABA_PATH}")
    print(f"Saved event CSV: {DRIFT_WITH_ABA_EVENTS_PATH}")
    generate_aba_chart()
    print_latest_summary()
    return drift_rows, event_rows


if __name__ == "__main__":
    selected_profiles = load_drift_profiles(ABA_PROFILE_COUNT)
    run_aba_simulation(selected_profiles)
