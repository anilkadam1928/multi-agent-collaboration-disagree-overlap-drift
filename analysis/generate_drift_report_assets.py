from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

DRIFT_NO_EMC = DATA_DIR / "drift_no_emc.csv"
DRIFT_WITH_EMC = DATA_DIR / "drift_with_emc.csv"
EMC_RESULTS = DATA_DIR / "emc_test_results.csv"

DRIFT_LINE_FIGURE = RESULTS_DIR / "Figure_3_Drift_Score_Over_Time.png"
AGENT_BAR_FIGURE = RESULTS_DIR / "Figure_3_Agent_Drift_At_Checkpoint_40.png"
COMPRESSION_FIGURE = RESULTS_DIR / "Figure_3_EMC_Compression_Ratio.png"
SUMMARY_MD = ANALYSIS_DIR / "week4_day1_drift_summary.md"
SUMMARY_CSV = ANALYSIS_DIR / "week4_day1_drift_summary.csv"

AGENT_LABELS = {
    "RouterManager": "Router",
    "IncomeAgent": "Income",
    "FraudAgent": "Fraud",
    "CreditAgent": "Credit",
    "ComplianceAgent": "Compliance",
    "SummariserAgent": "Summary",
    "WeakModelAgent": "Weak",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_drift() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    missing = [str(path) for path in (DRIFT_NO_EMC, DRIFT_WITH_EMC, EMC_RESULTS) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required CSV(s): " + ", ".join(missing))

    return read_csv(DRIFT_NO_EMC), read_csv(DRIFT_WITH_EMC)


def group_mean_drift(rows: list[dict[str, str]]) -> dict[int, float]:
    by_checkpoint: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_checkpoint[int(row["checkpoint"])].append(float(row["drift_score"]))
    return {checkpoint: mean(scores) for checkpoint, scores in by_checkpoint.items()}


def agent_drift_at(rows: list[dict[str, str]], checkpoint: int) -> dict[str, float]:
    return {
        row["agent_name"]: float(row["drift_score"])
        for row in rows
        if int(row["checkpoint"]) == checkpoint
    }


def compression_by_run() -> dict[int, dict[str, float]]:
    grouped: dict[int, dict[str, float]] = defaultdict(dict)
    for row in read_csv(EMC_RESULTS):
        grouped[int(row["run_id"])][row["agent_name"]] = float(row["compression_ratio"])
    return dict(grouped)


def setup_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def save_drift_line_chart(no_means: dict[int, float], with_means: dict[int, float]):
    checkpoints = sorted(set(no_means) | set(with_means))
    no_values = [no_means.get(checkpoint, 0.0) for checkpoint in checkpoints]
    with_values = [with_means.get(checkpoint, 0.0) for checkpoint in checkpoints]

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    ax.plot(checkpoints, no_values, marker="o", linewidth=2.5, color="#9B2C2C", label="Without EMC")
    ax.plot(checkpoints, with_values, marker="o", linewidth=2.5, color="#1F6F5B", label="With EMC")
    ax.axhline(0.15, color="#505A64", linewidth=1.2, linestyle="--", label="Drift threshold 0.15")
    ax.set_title("KL Drift Score Over Sequential Loan Processing", fontsize=13, pad=12)
    ax.set_xlabel("Borrower checkpoint")
    ax.set_ylabel("Mean KL drift score")
    ax.set_xticks(checkpoints)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(DRIFT_LINE_FIGURE)
    plt.close(fig)


def save_agent_bar_chart(no_rows: list[dict[str, str]], with_rows: list[dict[str, str]], checkpoint: int):
    no_scores = agent_drift_at(no_rows, checkpoint)
    with_scores = agent_drift_at(with_rows, checkpoint)
    agents = [agent for agent in AGENT_LABELS if agent in no_scores or agent in with_scores]
    x = list(range(len(agents)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9.6, 5.0), dpi=180)
    ax.bar([i - width / 2 for i in x], [no_scores.get(agent, 0.0) for agent in agents], width, color="#A94442", label="Without EMC")
    ax.bar([i + width / 2 for i in x], [with_scores.get(agent, 0.0) for agent in agents], width, color="#287D6A", label="With EMC")
    ax.axhline(0.15, color="#505A64", linewidth=1.2, linestyle="--")
    ax.set_title(f"Agent-Level KL Drift at Checkpoint {checkpoint}", fontsize=13, pad=12)
    ax.set_ylabel("KL drift score")
    ax.set_xticks(x)
    ax.set_xticklabels([AGENT_LABELS.get(agent, agent) for agent in agents], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(AGENT_BAR_FIGURE)
    plt.close(fig)


def save_compression_chart(compression: dict[int, dict[str, float]]):
    run_ids = sorted(compression)
    agents = [agent for agent in AGENT_LABELS if any(agent in compression[run_id] for run_id in run_ids)]
    x = list(range(len(agents)))
    width = 0.72 / max(len(run_ids), 1)
    palette = ["#345C8C", "#C27C2C", "#4F7A48"]

    fig, ax = plt.subplots(figsize=(9.8, 5.0), dpi=180)
    for idx, run_id in enumerate(run_ids):
        offset = (idx - (len(run_ids) - 1) / 2) * width
        ax.bar(
            [i + offset for i in x],
            [compression[run_id].get(agent, 0.0) for agent in agents],
            width,
            color=palette[idx % len(palette)],
            label=f"EMC at borrower {run_id}",
        )

    ax.set_title("EMC Memory Compression Ratio by Agent", fontsize=13, pad=12)
    ax.set_ylabel("Compression ratio (words before / words after)")
    ax.set_xticks(x)
    ax.set_xticklabels([AGENT_LABELS.get(agent, agent) for agent in agents], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(COMPRESSION_FIGURE)
    plt.close(fig)


def write_summary(no_rows: list[dict[str, str]], with_rows: list[dict[str, str]], no_means: dict[int, float], with_means: dict[int, float]):
    shared_checkpoints = sorted(set(no_means) & set(with_means))
    latest = shared_checkpoints[-1]
    no_latest = no_means[latest]
    with_latest = with_means[latest]
    reduction = (1 - with_latest / max(no_latest, 0.001)) * 100

    compression = compression_by_run()
    latest_compression_run = max(compression)
    latest_compression_values = list(compression[latest_compression_run].values())
    avg_compression = mean(latest_compression_values)

    no_agent = agent_drift_at(no_rows, latest)
    with_agent = agent_drift_at(with_rows, latest)
    improved_agents = [
        agent for agent in AGENT_LABELS
        if agent in no_agent and agent in with_agent and with_agent[agent] < no_agent[agent]
    ]

    summary_rows = [
        {"metric": "latest_checkpoint", "value": latest},
        {"metric": "mean_drift_without_emc", "value": round(no_latest, 4)},
        {"metric": "mean_drift_with_emc", "value": round(with_latest, 4)},
        {"metric": "drift_reduction_percent", "value": round(reduction, 1)},
        {"metric": "latest_emc_checkpoint", "value": latest_compression_run},
        {"metric": "average_compression_ratio_latest_emc", "value": round(avg_compression, 2)},
        {"metric": "agents_with_reduced_drift", "value": len(improved_agents)},
    ]

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(summary_rows)

    agent_lines = []
    for agent in AGENT_LABELS:
        if agent not in no_agent or agent not in with_agent:
            continue
        delta = with_agent[agent] - no_agent[agent]
        agent_lines.append(
            f"| {agent} | {no_agent[agent]:.4f} | {with_agent[agent]:.4f} | {delta:+.4f} |"
        )

    SUMMARY_MD.write_text(
        "\n".join(
            [
                "# Week 4 Day 1 Drift Summary",
                "",
                f"At checkpoint {latest}, mean KL drift decreased from {no_latest:.4f} without EMC to {with_latest:.4f} with EMC.",
                f"This is an {reduction:.1f}% drift reduction.",
                "",
                f"The latest EMC trigger at borrower {latest_compression_run} achieved an average compression ratio of {avg_compression:.2f}x across agents.",
                "",
                "## Agent-Level Drift at Latest Checkpoint",
                "",
                "| Agent | Without EMC | With EMC | Change |",
                "|---|---:|---:|---:|",
                *agent_lines,
                "",
                "## Report Wording",
                "",
                (
                    f"In the Week 4 Day 1 drift simulation, EMC reduced mean KL drift from {no_latest:.4f} "
                    f"to {with_latest:.4f} at checkpoint {latest}, corresponding to an {reduction:.1f}% reduction. "
                    f"The memory compaction mechanism also compressed agent histories by {avg_compression:.2f}x on average "
                    f"at the latest EMC trigger, confirming that context growth was materially reduced."
                ),
                "",
                "## Generated Figures",
                "",
                f"- {DRIFT_LINE_FIGURE}",
                f"- {AGENT_BAR_FIGURE}",
                f"- {COMPRESSION_FIGURE}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main():
    setup_dirs()
    no_rows, with_rows = load_drift()
    no_means = group_mean_drift(no_rows)
    with_means = group_mean_drift(with_rows)
    shared_checkpoints = sorted(set(no_means) & set(with_means))
    if not shared_checkpoints:
        raise ValueError("No shared checkpoints found between drift CSVs.")
    latest_checkpoint = shared_checkpoints[-1]

    save_drift_line_chart(no_means, with_means)
    save_agent_bar_chart(no_rows, with_rows, latest_checkpoint)
    save_compression_chart(compression_by_run())
    write_summary(no_rows, with_rows, no_means, with_means)

    print(f"Saved {DRIFT_LINE_FIGURE}")
    print(f"Saved {AGENT_BAR_FIGURE}")
    print(f"Saved {COMPRESSION_FIGURE}")
    print(f"Saved {SUMMARY_MD}")
    print(f"Saved {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
