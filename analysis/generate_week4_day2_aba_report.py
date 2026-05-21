from __future__ import annotations

"""Generate Week 4 Day 2 ABA charts and a short report.

This script is read-only with respect to experiment data. It does not edit
any simulation file or prior CSV. It only creates new report assets.
"""

import csv
import math
import subprocess
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

BASELINE_PATH = DATA_DIR / "baseline_results.csv"
DRIFT_NO_EMC_PATH = DATA_DIR / "drift_no_emc.csv"
DRIFT_WITH_EMC_PATH = DATA_DIR / "drift_with_emc.csv"
DRIFT_WITH_ABA_PATH = DATA_DIR / "drift_with_aba.csv"
ABA_EVENTS_PATH = DATA_DIR / "drift_with_aba_events.csv"

SUMMARY_CSV = ANALYSIS_DIR / "week4_day2_aba_summary.csv"
CORRECTED_KL_CSV = ANALYSIS_DIR / "week4_day2_aba_corrected_kl.csv"
REPORT_MD = ANALYSIS_DIR / "Week4_Day2_ABA_Report.md"
REPORT_HTML = ANALYSIS_DIR / "Week4_Day2_ABA_Report.html"
REPORT_DOCX = ANALYSIS_DIR / "Week4_Day2_ABA_Report.docx"

FIG_DRIFT = RESULTS_DIR / "Figure_4A_ABA_Drift_Comparison.png"
FIG_DECISIONS = RESULTS_DIR / "Figure_4B_ABA_Decision_Mix.png"
FIG_TRIGGERS = RESULTS_DIR / "Figure_4C_ABA_Trigger_Counts.png"
FIG_KL_UNKNOWN = RESULTS_DIR / "Figure_4D_ABA_Corrected_KL_With_Unknown.png"
FIG_ROUTING = RESULTS_DIR / "Figure_4E_ABA_Routing_Leader_Frequency.png"

DECISION_LABELS_3 = ["approve", "refer", "reject"]
DECISION_LABELS_4 = ["approve", "refer", "reject", "unknown"]
AGENT_ORDER = [
    "RouterManager",
    "IncomeAgent",
    "FraudAgent",
    "CreditAgent",
    "ComplianceAgent",
    "SummariserAgent",
    "WeakModelAgent",
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def mean_drift_by_checkpoint(path: Path) -> pd.Series:
    df = read_csv(path)
    grouped = df.groupby("checkpoint")["drift_score"].mean().sort_index()
    return grouped


def percent_change(new_value: float, old_value: float) -> float:
    if abs(old_value) < 1e-9:
        return 0.0
    return (1 - (new_value / old_value)) * 100


def smoothed_distribution(values: pd.Series, labels: list[str]) -> dict[str, float]:
    counts = Counter(str(value).lower() for value in values.fillna("unknown"))
    total = sum(counts.get(label, 0) for label in labels)
    return {
        label: (counts.get(label, 0) + 1e-6) / (total + len(labels) * 1e-6)
        for label in labels
    }


def kl_divergence(current: dict[str, float], baseline: dict[str, float], labels: list[str]) -> float:
    return sum(current[label] * math.log(current[label] / baseline[label]) for label in labels)


def corrected_kl_with_unknown(baseline: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    events = events.copy()
    events["borrower_num"] = events["borrower_id"].astype(str).str.extract(r"(\d+)").astype(int)

    checkpoints = sorted(events["borrower_num"].unique())
    checkpoints = [point for point in checkpoints if point in {20, 40}]
    if not checkpoints:
        checkpoints = [int(events["borrower_num"].max())]

    for checkpoint in checkpoints:
        checkpoint_events = events[events["borrower_num"] <= checkpoint]
        for agent_name in AGENT_ORDER:
            baseline_agent = baseline[baseline["agent_name"] == agent_name]
            current_agent = checkpoint_events[checkpoint_events["agent_name"] == agent_name]

            baseline_dist = smoothed_distribution(baseline_agent["parsed_decision"], DECISION_LABELS_4)
            current_dist = smoothed_distribution(current_agent["parsed_decision"], DECISION_LABELS_4)
            score = kl_divergence(current_dist, baseline_dist, DECISION_LABELS_4)

            known_rate = current_agent["parsed_decision"].isin(DECISION_LABELS_3).mean()
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "agent_name": agent_name,
                    "corrected_kl_including_unknown": round(score, 4),
                    "known_decision_rate": round(float(known_rate), 4),
                    "approve_count": int((current_agent["parsed_decision"] == "approve").sum()),
                    "refer_count": int((current_agent["parsed_decision"] == "refer").sum()),
                    "reject_count": int((current_agent["parsed_decision"] == "reject").sum()),
                    "unknown_count": int((current_agent["parsed_decision"] == "unknown").sum()),
                }
            )

    return pd.DataFrame(rows)


def save_summary(no_emc: pd.Series, with_emc: pd.Series, with_aba: pd.Series, corrected: pd.DataFrame) -> dict[str, float]:
    latest = int(with_aba.index.max())
    no_value = float(no_emc.loc[latest]) if latest in no_emc.index else float("nan")
    emc_value = float(with_emc.loc[latest]) if latest in with_emc.index else float("nan")
    aba_value = float(with_aba.loc[latest])
    corrected_latest = corrected[corrected["checkpoint"] == latest]
    corrected_mean = float(corrected_latest["corrected_kl_including_unknown"].mean())
    known_rate = float(corrected_latest["known_decision_rate"].mean())

    metrics = {
        "latest_checkpoint": latest,
        "without_emc_mean_kl_3class": round(no_value, 4),
        "with_emc_mean_kl_3class": round(emc_value, 4),
        "with_emc_aba_mean_kl_3class": round(aba_value, 4),
        "aba_reduction_vs_emc_percent_3class": round(percent_change(aba_value, emc_value), 1),
        "corrected_mean_kl_including_unknown": round(corrected_mean, 4),
        "mean_known_decision_rate": round(known_rate, 4),
    }

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    return metrics


def plot_drift_comparison(no_emc: pd.Series, with_emc: pd.Series, with_aba: pd.Series) -> None:
    plt.figure(figsize=(8.8, 5.2))
    for label, series, color in [
        ("Without EMC", no_emc, "#8C3B2E"),
        ("With EMC", with_emc, "#2E6F95"),
        ("With EMC + ABA", with_aba, "#2E8B57"),
    ]:
        plt.plot(series.index, series.values, marker="o", linewidth=2.6, label=label, color=color)

    plt.axhline(0.15, linestyle="--", color="#CC3333", linewidth=1.5, label="KL threshold 0.15")
    plt.title("Mean Drift Score Over Time")
    plt.xlabel("Borrower checkpoint")
    plt.ylabel("Mean KL drift score")
    plt.xticks(sorted(set(no_emc.index) | set(with_emc.index) | set(with_aba.index)))
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DRIFT, dpi=220)
    plt.close()


def plot_decision_mix(events: pd.DataFrame) -> None:
    counts = (
        events.groupby(["agent_name", "parsed_decision"])
        .size()
        .unstack(fill_value=0)
        .reindex(AGENT_ORDER)
        .reindex(columns=DECISION_LABELS_4, fill_value=0)
    )
    shares = counts.div(counts.sum(axis=1), axis=0)

    colors = {
        "approve": "#2E8B57",
        "refer": "#D9A441",
        "reject": "#B84A4A",
        "unknown": "#777777",
    }

    ax = shares.plot(
        kind="bar",
        stacked=True,
        figsize=(10.0, 5.4),
        color=[colors[label] for label in DECISION_LABELS_4],
        width=0.75,
    )
    ax.set_title("ABA Run Decision Mix by Agent")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Share of 40 borrower decisions")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(title="Decision", loc="upper right")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DECISIONS, dpi=220)
    plt.close()


def plot_aba_triggers(events: pd.DataFrame) -> None:
    trigger_counts = (
        events.groupby("agent_name")["aba_triggered"]
        .sum()
        .reindex(AGENT_ORDER)
        .fillna(0)
        .astype(int)
    )

    plt.figure(figsize=(9.4, 5.0))
    bars = plt.bar(trigger_counts.index, trigger_counts.values, color="#3A6EA5")
    plt.title("ABA Trigger Count by Agent")
    plt.xlabel("Agent")
    plt.ylabel("Number of borrower calls with ABA active")
    plt.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=30, ha="right")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.4, str(int(height)), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIG_TRIGGERS, dpi=220)
    plt.close()


def plot_corrected_kl(corrected: pd.DataFrame) -> None:
    latest = corrected["checkpoint"].max()
    latest_df = corrected[corrected["checkpoint"] == latest].set_index("agent_name").reindex(AGENT_ORDER)

    plt.figure(figsize=(9.6, 5.0))
    values = latest_df["corrected_kl_including_unknown"]
    colors = ["#B84A4A" if value > 0.15 else "#2E8B57" for value in values]
    bars = plt.bar(values.index, values.values, color=colors)
    plt.axhline(0.15, linestyle="--", color="#CC3333", linewidth=1.5, label="KL threshold 0.15")
    plt.title(f"Corrected KL Drift Including Unknown Outputs at Checkpoint {latest}")
    plt.xlabel("Agent")
    plt.ylabel("KL drift including unknown")
    plt.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.04, f"{height:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_KL_UNKNOWN, dpi=220)
    plt.close()


def plot_routing(events: pd.DataFrame) -> None:
    routes = events[["borrower_id", "routed_to_agent"]].drop_duplicates()
    route_counts = routes["routed_to_agent"].value_counts().reindex(AGENT_ORDER).fillna(0).astype(int)

    plt.figure(figsize=(9.4, 5.0))
    bars = plt.bar(route_counts.index, route_counts.values, color="#5B7C99")
    plt.title("Drift-Aware Routing Leader Frequency")
    plt.xlabel("Selected route leader")
    plt.ylabel("Number of borrowers")
    plt.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=30, ha="right")
    for bar in bars:
        height = bar.get_height()
        if height:
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.15, str(int(height)), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIG_ROUTING, dpi=220)
    plt.close()


def markdown_image(path: Path) -> str:
    return f"![{path.stem}]({path})"


def write_report(metrics: dict[str, float]) -> None:
    text = f"""# Week 4 Day 2: Adaptive Behaviour Anchoring Report

## Objective

The Tuesday experiment tested Adaptive Behaviour Anchoring (ABA) on top of the Monday EMC drift module. The purpose was to detect agents whose decision distribution drifted away from the frozen Week 3 baseline and inject a small set of anchor examples when drift exceeded the KL threshold of 0.15.

## Setup

- Baseline file: `{BASELINE_PATH}`
- EMC-only drift file: `{DRIFT_WITH_EMC_PATH}`
- ABA event file: `{ABA_EVENTS_PATH}`
- Profiles run: 40
- Checkpoints: 20 and 40 borrowers
- ABA trigger threshold: KL drift > 0.15

## Headline Result

Using the original three-class decision space (approve, refer, reject), the mean drift score decreased from EMC-only `{metrics['with_emc_mean_kl_3class']}` to EMC + ABA `{metrics['with_emc_aba_mean_kl_3class']}` at checkpoint `{metrics['latest_checkpoint']}`. This is a `{metrics['aba_reduction_vs_emc_percent_3class']}%` reduction relative to EMC-only.

{markdown_image(FIG_DRIFT)}

Interpretation: ABA substantially reduced the measured three-class drift score after the checkpoint trigger. This suggests the anchor examples helped pull the agents closer to the frozen baseline decision pattern.

## ABA Trigger Pattern

{markdown_image(FIG_TRIGGERS)}

Interpretation: ABA activated only for the agents whose checkpoint drift crossed the threshold. In this run, RouterManager, CreditAgent, ComplianceAgent, and SummariserAgent were repeatedly anchored after the first checkpoint.

## Decision Mix and Output Quality

{markdown_image(FIG_DECISIONS)}

Interpretation: the run still produced a large number of `unknown` outputs for some agents. This is important because `unknown` is not a business decision, but it is a useful reliability signal from the local model. The mean known-decision rate at checkpoint `{metrics['latest_checkpoint']}` was `{metrics['mean_known_decision_rate']}`.

## Corrected Drift Audit Including Unknown

{markdown_image(FIG_KL_UNKNOWN)}

Interpretation: when `unknown` is included as a fourth decision class, drift remains high for several agents. This does not invalidate the ABA mechanism. Instead, it shows that there are two separate effects:

1. ABA improved alignment within the parsed approve/refer/reject decision space.
2. The local model still had output-format instability, visible through unknown decisions.

## Drift-Aware Routing

{markdown_image(FIG_ROUTING)}

Interpretation: the routing tracker selected the currently least-drifted agent as route leader for each borrower. This creates an audit trail showing how the system would prefer lower-drift agents for later decisions.

## Report-Ready Conclusion

The Week 4 Day 2 ABA module produced a strong three-class drift reduction compared with EMC-only, reducing mean KL drift by `{metrics['aba_reduction_vs_emc_percent_3class']}%` at checkpoint `{metrics['latest_checkpoint']}`. However, the corrected audit including `unknown` outputs shows that formatting instability remained for some agents. Therefore, the result should be reported as partial success: ABA improved decision-distribution calibration, but output-format reliability still requires stricter parsing or stronger response constraints in the next iteration.
"""
    REPORT_MD.write_text(text, encoding="utf-8")

    html = text
    html = html.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    lines: list[str] = []
    for raw_line in html.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append("")
        elif line.startswith("# "):
            lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("- "):
            lines.append(f"<p>{line}</p>")
        elif line.startswith("!["):
            alt = line.split("]", 1)[0][2:]
            src = line.split("(", 1)[1].rstrip(")")
            lines.append(f'<p><img src="{src}" alt="{alt}" style="max-width: 900px; width: 100%;"></p>')
        else:
            lines.append(f"<p>{line}</p>")

    REPORT_HTML.write_text(
        "<html><head><meta charset='utf-8'><style>"
        "body{font-family:Arial, sans-serif; max-width:980px; margin:32px auto; line-height:1.45;}"
        "h1,h2{color:#1f4e79;} code{background:#f2f2f2; padding:2px 4px;}"
        "img{border:1px solid #ddd; margin:8px 0 20px 0;}"
        "</style></head><body>"
        + "\n".join(lines)
        + "</body></html>",
        encoding="utf-8",
    )


def try_convert_html_to_docx() -> bool:
    try:
        subprocess.run(
            ["textutil", "-convert", "docx", str(REPORT_HTML), "-output", str(REPORT_DOCX)],
            check=True,
            capture_output=True,
            text=True,
        )
        return REPORT_DOCX.exists()
    except Exception:
        return False


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    baseline = read_csv(BASELINE_PATH)
    events = read_csv(ABA_EVENTS_PATH)
    no_emc = mean_drift_by_checkpoint(DRIFT_NO_EMC_PATH)
    with_emc = mean_drift_by_checkpoint(DRIFT_WITH_EMC_PATH)
    with_aba = mean_drift_by_checkpoint(DRIFT_WITH_ABA_PATH)

    corrected = corrected_kl_with_unknown(baseline, events)
    corrected.to_csv(CORRECTED_KL_CSV, index=False)

    metrics = save_summary(no_emc, with_emc, with_aba, corrected)

    plot_drift_comparison(no_emc, with_emc, with_aba)
    plot_decision_mix(events)
    plot_aba_triggers(events)
    plot_corrected_kl(corrected)
    plot_routing(events)
    write_report(metrics)
    converted = try_convert_html_to_docx()

    print("Generated Week 4 Day 2 ABA report assets:")
    for path in [
        SUMMARY_CSV,
        CORRECTED_KL_CSV,
        FIG_DRIFT,
        FIG_DECISIONS,
        FIG_TRIGGERS,
        FIG_KL_UNKNOWN,
        FIG_ROUTING,
        REPORT_MD,
        REPORT_HTML,
        REPORT_DOCX if converted else None,
    ]:
        if path:
            print(f"- {path}")


if __name__ == "__main__":
    main()
