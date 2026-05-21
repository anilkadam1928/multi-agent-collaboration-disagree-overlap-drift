"""
verify_200_borrower_run.py
==========================

Recomputes the report's findings from a 100/200-borrower audit log instead of
trusting a final chart.

Run after the integrated experiment:

    python verify_200_borrower_run.py \
        --input results/borrower_audit_200.csv \
        --out-dir results/robustness_200

Create the required CSV header/template:

    python verify_200_borrower_run.py \
        --write-template results/borrower_audit_200_template.csv

The verifier is intentionally strict.  If a claim cannot be supported because
raw fields are missing, it writes that finding as NEEDS_DATA instead of
silently treating the run as robust.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from experiment_audit import (
    AUDIT_FIELDNAMES,
    REQUIRED_FIELDS_BY_FINDING,
    bool_value,
    expected_decision_for_truth,
    float_value,
    int_value,
    normalise_decision,
    normalise_truth,
    write_template,
)


DECISION_ORDER = ("approve", "refer", "reject")
FINAL_STAGE_HINTS = {
    "final",
    "aggregate",
    "aggregated",
    "resolution",
    "resolved",
    "post_rl",
    "hierarchy_final",
}


def is_present(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def mean(values: Iterable[float]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Audit input not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    rows.append(json.loads(text))
        return rows

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def unique_borrowers(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row.get("borrower_id", "")).strip() for row in rows if is_present(row.get("borrower_id"))})


def available_columns(rows: list[dict[str, Any]]) -> set[str]:
    columns: set[str] = set()
    for row in rows:
        columns.update(row.keys())
    return columns


def non_empty_columns(rows: list[dict[str, Any]]) -> set[str]:
    columns: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if is_present(value):
                columns.add(key)
    return columns


def stage_rank(row: dict[str, Any]) -> int:
    stage = str(row.get("stage", "")).strip().lower()
    agent = str(row.get("agent_name", "")).strip().lower()
    if stage in FINAL_STAGE_HINTS:
        return 4
    if agent in {"routermanager", "router-manager", "strategicoverseer", "strategic overseer"}:
        return 3
    if not agent:
        return 2
    return 1


def final_rows_by_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select one final decision row per method and borrower.

    Audit logs often repeat the group decision on each agent row.  This helper
    de-duplicates those rows so accuracy and approval rates are not inflated.
    """
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    selected_rank: dict[tuple[str, str], int] = {}

    for row in rows:
        borrower_id = str(row.get("borrower_id", "")).strip()
        method = str(row.get("method", "")).strip() or str(row.get("scenario", "")).strip()
        decision = normalise_decision(row.get("final_decision"))
        truth = normalise_truth(row.get("truth_label"))
        if not borrower_id or not method or decision not in DECISION_ORDER or truth not in {"good", "bad"}:
            continue

        key = (method, borrower_id)
        rank = stage_rank(row)
        if key not in selected or rank > selected_rank[key]:
            selected[key] = row
            selected_rank[key] = rank

    return list(selected.values())


def compute_method_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in final_rows_by_method(rows):
        method = str(row.get("method", "")).strip() or str(row.get("scenario", "")).strip()
        grouped[method].append(row)

    for method, method_rows in grouped.items():
        counts = Counter()
        correct = 0
        risk_safe_correct = 0
        bad_total = 0
        good_total = 0
        bad_approved = 0
        bad_detected = 0
        good_approved = 0

        for row in method_rows:
            truth = normalise_truth(row.get("truth_label"))
            decision = normalise_decision(row.get("final_decision"))
            expected = normalise_decision(row.get("expected_decision")) or expected_decision_for_truth(truth)
            counts[f"{truth}_{decision}"] += 1
            counts[f"decision_{decision}"] += 1

            if decision == expected:
                correct += 1

            if truth == "bad":
                bad_total += 1
                if decision == "approve":
                    bad_approved += 1
                if decision in {"refer", "reject"}:
                    bad_detected += 1
                    risk_safe_correct += 1
            elif truth == "good":
                good_total += 1
                if decision == "approve":
                    good_approved += 1
                    risk_safe_correct += 1

        total = len(method_rows)
        metrics[method] = {
            "profiles": total,
            "strict_accuracy": safe_div(correct, total),
            "risk_safe_accuracy": safe_div(risk_safe_correct, total),
            "approval_rate": safe_div(counts["decision_approve"], total),
            "refer_rate": safe_div(counts["decision_refer"], total),
            "reject_rate": safe_div(counts["decision_reject"], total),
            "bad_borrower_approval_rate": safe_div(bad_approved, bad_total),
            "bad_borrower_detection_rate": safe_div(bad_detected, bad_total),
            "good_borrower_approval_rate": safe_div(good_approved, good_total),
            "good_borrower_nonapproval_rate": safe_div(good_total - good_approved, good_total),
            "confusion_counts": dict(counts),
        }

    return metrics


def compute_disagreement_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def grouped_disagreement(decision_field: str) -> dict[str, Any]:
        groups: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        for row in rows:
            borrower_id = str(row.get("borrower_id", "")).strip()
            method = str(row.get("method", "")).strip() or str(row.get("scenario", "")).strip()
            stage = str(row.get("stage", "")).strip() or decision_field
            agent = str(row.get("agent_name", "")).strip()
            decision = normalise_decision(row.get(decision_field))
            if borrower_id and method and agent and decision in DECISION_ORDER:
                groups[(method, borrower_id, stage)].add(decision)

        eligible = [decisions for decisions in groups.values() if len(decisions) >= 1]
        disagreed = [decisions for decisions in eligible if len(decisions) > 1]
        return {
            "groups": len(eligible),
            "disagreement_count": len(disagreed),
            "disagreement_rate": safe_div(len(disagreed), len(eligible)),
        }

    borda_rows: dict[str, dict[str, Any]] = {}
    consens_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        borrower_id = str(row.get("borrower_id", "")).strip()
        if borrower_id and is_present(row.get("borda_winner")):
            borda_rows[borrower_id] = row
        if borrower_id and is_present(row.get("consensagent_winner")):
            consens_rows[borrower_id] = row

    def resolution_accuracy(source: dict[str, dict[str, Any]], field: str) -> float | None:
        if not source:
            return None
        correct = 0
        total = 0
        for row in source.values():
            truth = normalise_truth(row.get("truth_label"))
            expected = expected_decision_for_truth(truth)
            decision = normalise_decision(row.get(field))
            if expected and decision in DECISION_ORDER:
                total += 1
                if decision == expected:
                    correct += 1
        return safe_div(correct, total)

    return {
        "precommit": grouped_disagreement("precommit_decision"),
        "post_discussion": grouped_disagreement("post_discussion_decision"),
        "final_agent_decision": grouped_disagreement("agent_decision"),
        "borda_accuracy": resolution_accuracy(borda_rows, "borda_winner"),
        "consensagent_accuracy": resolution_accuracy(consens_rows, "consensagent_winner"),
    }


def compute_sycophancy_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_changes = 0
    total_flags = 0
    flagged_similarities: list[float] = []
    unflagged_similarities: list[float] = []

    for row in rows:
        pre = normalise_decision(row.get("precommit_decision"))
        post = normalise_decision(row.get("final_decision")) or normalise_decision(row.get("post_discussion_decision"))
        changed = bool_value(row.get("changed_after_discussion"))
        if changed is None and pre in DECISION_ORDER and post in DECISION_ORDER:
            changed = pre != post

        if not changed:
            continue

        total_changes += 1
        flagged = bool_value(row.get("sycophancy_flagged")) is True
        similarity = float_value(row.get("similarity_score"))
        if flagged:
            total_flags += 1
            if similarity is not None:
                flagged_similarities.append(similarity)
        elif similarity is not None:
            unflagged_similarities.append(similarity)

    return {
        "total_changes": total_changes,
        "total_flags": total_flags,
        "sycophancy_rate": safe_div(total_flags, total_changes),
        "mean_similarity_flagged": mean(flagged_similarities),
        "mean_similarity_unflagged": mean(unflagged_similarities),
    }


def borrower_numeric_aggregates(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        borrower_id = str(row.get("borrower_id", "")).strip()
        value = float_value(row.get(field))
        if borrower_id and value is not None:
            values[borrower_id].append(value)
    # Use max so repeated per-agent rows do not multiply the same borrower-level count.
    return {borrower_id: max(items) for borrower_id, items in values.items()}


def compute_overlap_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_duplicates = borrower_numeric_aggregates(rows, "duplicate_invocations_raw")
    after_lce = borrower_numeric_aggregates(rows, "duplicate_invocations_after_lce")
    added_by_resolution = borrower_numeric_aggregates(rows, "additional_resolution_invocations")
    removed = borrower_numeric_aggregates(rows, "lce_removed_invocations")
    raw_ri = borrower_numeric_aggregates(rows, "raw_redundancy_index")
    controlled_ri = borrower_numeric_aggregates(rows, "controlled_redundancy_index")

    leaders_by_borrower: dict[str, tuple[int, str]] = {}
    for row in rows:
        borrower_id = str(row.get("borrower_id", "")).strip()
        leader = str(row.get("leader_agent", "")).strip()
        if not borrower_id or not leader:
            continue
        profile_index = int_value(row.get("profile_index")) or 0
        leaders_by_borrower[borrower_id] = (profile_index, leader)

    ordered_leaders = [
        leader
        for _borrower_id, (_idx, leader) in sorted(
            leaders_by_borrower.items(),
            key=lambda item: (item[1][0], item[0]),
        )
    ]
    leader_transitions = sum(
        1
        for previous, current in zip(ordered_leaders, ordered_leaders[1:])
        if previous != current
    )

    return {
        "raw_duplicate_invocations_total": sum(raw_duplicates.values()) if raw_duplicates else None,
        "post_lce_duplicate_invocations_total": sum(after_lce.values()) if after_lce else None,
        "avg_additional_resolution_invocations": mean(added_by_resolution.values()),
        "avg_lce_removed_invocations": mean(removed.values()),
        "raw_redundancy_index": mean(raw_ri.values()),
        "controlled_redundancy_index": mean(controlled_ri.values()),
        "redundancy_reduction_rate": (
            safe_div(mean(raw_ri.values()) - mean(controlled_ri.values()), mean(raw_ri.values()))
            if mean(raw_ri.values()) is not None and mean(controlled_ri.values()) is not None
            else None
        ),
        "leader_transitions": leader_transitions if ordered_leaders else None,
        "leader_distribution": dict(Counter(ordered_leaders)),
    }


def compute_drift_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition_checkpoint: dict[tuple[str, int], list[float]] = defaultdict(list)
    exceeded = Counter()
    aba_events = 0
    aba_rows = 0
    skipped_for_drift = 0

    for row in rows:
        score = float_value(row.get("drift_score"))
        condition = str(row.get("drift_condition", "")).strip() or str(row.get("method", "")).strip()
        checkpoint = int_value(row.get("checkpoint"))
        if condition and checkpoint is not None and score is not None:
            by_condition_checkpoint[(condition, checkpoint)].append(score)

        if bool_value(row.get("drift_exceeded")) is True:
            exceeded[str(row.get("agent_name", "")).strip() or "unknown"] += 1
        if bool_value(row.get("aba_applied")) is True:
            aba_rows += 1
        event_count = int_value(row.get("aba_event_count"))
        if event_count is not None:
            aba_events += event_count
        if bool_value(row.get("agent_skipped_for_drift")) is True:
            skipped_for_drift += 1

    condition_rows = []
    for (condition, checkpoint), scores in sorted(by_condition_checkpoint.items()):
        condition_rows.append(
            {
                "condition": condition,
                "checkpoint": checkpoint,
                "mean_drift_score": mean(scores),
                "max_drift_score": max(scores),
                "rows": len(scores),
            }
        )

    return {
        "by_condition_checkpoint": condition_rows,
        "drift_exceeded_by_agent": dict(exceeded),
        "aba_event_count": aba_events,
        "aba_rows": aba_rows,
        "agent_skipped_for_drift_count": skipped_for_drift,
    }


def compute_causal_chain_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    borrower_flags: dict[str, dict[str, bool]] = defaultdict(lambda: {"overlap": False, "disagreement": False})
    for row in rows:
        borrower_id = str(row.get("borrower_id", "")).strip()
        if not borrower_id:
            continue
        if bool_value(row.get("overlap_detected")) is True:
            borrower_flags[borrower_id]["overlap"] = True
        if bool_value(row.get("disagreement_detected")) is True:
            borrower_flags[borrower_id]["disagreement"] = True

    eligible = list(borrower_flags.values())
    cooccurred = [flags for flags in eligible if flags["overlap"] and flags["disagreement"]]

    grouped_drift: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        group = str(row.get("interaction_group", "")).strip().lower()
        score = float_value(row.get("group_kl_drift_score"))
        if group and score is not None:
            grouped_drift[group].append(score)

    high = mean(grouped_drift.get("high", []))
    low = mean(grouped_drift.get("low", []))

    return {
        "cooccurrence_profiles": len(cooccurred),
        "profiles_with_chain_flags": len(eligible),
        "overlap_disagreement_cooccurrence_rate": safe_div(len(cooccurred), len(eligible)),
        "high_group_kl_drift": high,
        "low_group_kl_drift": low,
        "high_low_drift_ratio": safe_div(high, low) if high is not None and low is not None else None,
    }


def compute_concept_drift_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_batch: dict[str, list[float]] = defaultdict(list)
    flags = Counter()
    stress_rows = 0
    for row in rows:
        batch_id = str(row.get("concept_batch_id", "")).strip()
        corr = float_value(row.get("concept_correlation"))
        if batch_id and corr is not None:
            by_batch[batch_id].append(corr)
        if bool_value(row.get("concept_drift_flag")) is True:
            flags[batch_id or "unknown"] += 1
        if bool_value(row.get("stress_test")) is True:
            stress_rows += 1

    batch_rows = [
        {"concept_batch_id": batch_id, "mean_correlation": mean(values), "rows": len(values)}
        for batch_id, values in sorted(by_batch.items())
    ]
    correlations = [row["mean_correlation"] for row in batch_rows if row["mean_correlation"] is not None]

    return {
        "batches": batch_rows,
        "min_natural_correlation": min(correlations) if correlations else None,
        "concept_drift_flags": dict(flags),
        "stress_test_rows": stress_rows,
    }


def compute_reward_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reward_values = [float_value(row.get("reward_value")) for row in rows]
    reward_values = [value for value in reward_values if value is not None]
    route_deltas = [float_value(row.get("route_weight_delta")) for row in rows]
    route_deltas = [value for value in route_deltas if value is not None]
    correctness = [bool_value(row.get("reward_correct")) for row in rows if bool_value(row.get("reward_correct")) is not None]

    return {
        "reward_events": len(reward_values),
        "total_reward": sum(reward_values) if reward_values else None,
        "mean_reward": mean(reward_values),
        "mean_route_weight_delta": mean(route_deltas),
        "reward_correct_rate": safe_div(sum(1 for value in correctness if value), len(correctness)) if correctness else None,
    }


def compute_hierarchy_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    manager_flags = [bool_value(row.get("manager_disagreement")) for row in rows]
    manager_flags = [flag for flag in manager_flags if flag is not None]
    l1_triggers = sum(1 for row in rows if bool_value(row.get("l1_triggered")) is True)
    resets = sum(1 for row in rows if bool_value(row.get("reset_issued")) is True)

    reductions = []
    for row in rows:
        baseline = float_value(row.get("baseline_drift_score"))
        post = float_value(row.get("post_reset_drift_score"))
        if baseline is not None and post is not None and baseline != 0:
            reductions.append((baseline - post) / baseline)

    return {
        "manager_disagreement_rate": safe_div(sum(1 for flag in manager_flags if flag), len(manager_flags)) if manager_flags else None,
        "l1_trigger_count": l1_triggers,
        "reset_count": resets,
        "mean_drift_reduction_after_reset": mean(reductions),
    }


def compute_claim_readiness(rows: list[dict[str, Any]], min_borrowers: int) -> list[dict[str, Any]]:
    columns = available_columns(rows)
    non_empty = non_empty_columns(rows)
    borrower_count = len(unique_borrowers(rows))
    readiness = []

    for finding, required_fields in REQUIRED_FIELDS_BY_FINDING.items():
        missing_columns = [field for field in required_fields if field not in columns]
        empty_columns = [field for field in required_fields if field in columns and field not in non_empty]
        has_required_fields = not missing_columns and not empty_columns
        has_enough_borrowers = borrower_count >= min_borrowers

        if has_required_fields and has_enough_borrowers:
            status = "READY"
        elif has_required_fields:
            status = "PARTIAL_SAMPLE"
        else:
            status = "NEEDS_DATA"

        readiness.append(
            {
                "finding": finding,
                "status": status,
                "unique_borrowers": borrower_count,
                "min_borrowers": min_borrowers,
                "missing_columns": ",".join(missing_columns),
                "empty_columns": ",".join(empty_columns),
                "required_fields": ",".join(required_fields),
            }
        )

    return readiness


def flatten_method_metrics(method_metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method, metrics in sorted(method_metrics.items()):
        flat = {"method": method}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    flat[f"{key}.{nested_key}"] = nested_value
            else:
                flat[key] = value
        rows.append(flat)
    return rows


def build_summary(rows: list[dict[str, Any]], min_borrowers: int) -> dict[str, Any]:
    borrowers = unique_borrowers(rows)
    method_metrics = compute_method_metrics(rows)
    readiness = compute_claim_readiness(rows, min_borrowers)

    return {
        "input_rows": len(rows),
        "unique_borrowers": len(borrowers),
        "borrower_ids_sample": borrowers[:10],
        "min_borrowers_required": min_borrowers,
        "methods": sorted(method_metrics.keys()),
        "claim_readiness": readiness,
        "method_metrics": method_metrics,
        "disagreement": compute_disagreement_metrics(rows),
        "sycophancy": compute_sycophancy_metrics(rows),
        "overlap": compute_overlap_metrics(rows),
        "drift": compute_drift_metrics(rows),
        "causal_chain": compute_causal_chain_metrics(rows),
        "concept_drift": compute_concept_drift_metrics(rows),
        "reward_loop": compute_reward_metrics(rows),
        "hierarchy": compute_hierarchy_metrics(rows),
    }


def write_metric_tables(summary: dict[str, Any], out_dir: Path) -> None:
    write_csv(out_dir / "method_metrics.csv", flatten_method_metrics(summary["method_metrics"]))
    write_csv(out_dir / "claim_readiness.csv", summary["claim_readiness"])

    missing_rows = [
        row
        for row in summary["claim_readiness"]
        if row["missing_columns"] or row["empty_columns"] or row["status"] != "READY"
    ]
    write_csv(out_dir / "missing_or_incomplete_evidence.csv", missing_rows)

    drift_rows = summary["drift"]["by_condition_checkpoint"]
    if drift_rows:
        write_csv(out_dir / "drift_by_condition_checkpoint.csv", drift_rows)

    concept_rows = summary["concept_drift"]["batches"]
    if concept_rows:
        write_csv(out_dir / "concept_drift_batches.csv", concept_rows)


def plot_summary(summary: dict[str, Any], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional visual only
        print(f"Skipping plot because matplotlib is unavailable: {exc}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("200-Borrower Robustness Reverification Audit", fontsize=14, fontweight="bold")

    method_metrics = summary["method_metrics"]
    methods = list(method_metrics.keys())[:10]

    ax = axes[0, 0]
    if methods:
        values = [method_metrics[m].get("strict_accuracy") or 0 for m in methods]
        ax.bar(methods, values, color="#1f77b4")
        ax.set_ylim(0, 1)
        ax.set_title("Strict Accuracy")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No method metrics", ha="center", va="center")
        ax.axis("off")

    ax = axes[0, 1]
    if methods:
        values = [method_metrics[m].get("bad_borrower_approval_rate") or 0 for m in methods]
        ax.bar(methods, values, color="#d62728")
        ax.set_ylim(0, 1)
        ax.set_title("Bad Borrower Approval Rate")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No safety metrics", ha="center", va="center")
        ax.axis("off")

    ax = axes[1, 0]
    readiness_counts = Counter(row["status"] for row in summary["claim_readiness"])
    labels = ["READY", "PARTIAL_SAMPLE", "NEEDS_DATA"]
    values = [readiness_counts[label] for label in labels]
    ax.bar(labels, values, color=["#2ca02c", "#ff7f0e", "#7f7f7f"])
    ax.set_title("Finding Reverification Readiness")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    drift_rows = summary["drift"]["by_condition_checkpoint"]
    latest_by_condition: dict[str, dict[str, Any]] = {}
    for row in drift_rows:
        condition = row["condition"]
        if condition not in latest_by_condition or row["checkpoint"] > latest_by_condition[condition]["checkpoint"]:
            latest_by_condition[condition] = row

    if latest_by_condition:
        labels = list(latest_by_condition.keys())[:10]
        values = [latest_by_condition[label]["mean_drift_score"] or 0 for label in labels]
        ax.bar(labels, values, color="#9467bd")
        ax.set_title("Latest Mean Drift Score")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    else:
        syco = summary["sycophancy"]
        labels = ["changes", "flags"]
        values = [syco["total_changes"], syco["total_flags"]]
        ax.bar(labels, values, color=["#7f7f7f", "#ff7f0e"])
        ax.set_title("Sycophancy Evidence")
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def print_console_summary(summary: dict[str, Any], out_dir: Path) -> None:
    readiness = Counter(row["status"] for row in summary["claim_readiness"])
    print("\n200-borrower audit summary")
    print(f"  Rows read:          {summary['input_rows']}")
    print(f"  Unique borrowers:   {summary['unique_borrowers']} / {summary['min_borrowers_required']}")
    print(f"  Methods found:      {', '.join(summary['methods']) if summary['methods'] else 'none'}")
    print(
        "  Claim readiness:   "
        f"READY={readiness['READY']}, "
        f"PARTIAL_SAMPLE={readiness['PARTIAL_SAMPLE']}, "
        f"NEEDS_DATA={readiness['NEEDS_DATA']}"
    )

    syco = summary["sycophancy"]
    if syco["total_changes"]:
        print(
            "  Sycophancy:        "
            f"{syco['total_flags']} flags / {syco['total_changes']} changes "
            f"({syco['sycophancy_rate']:.3f})"
        )

    overlap = summary["overlap"]
    if overlap["redundancy_reduction_rate"] is not None:
        print(f"  Redundancy drop:   {overlap['redundancy_reduction_rate']:.3f}")

    hierarchy = summary["hierarchy"]
    if hierarchy["mean_drift_reduction_after_reset"] is not None:
        print(f"  Hierarchy drift:   {hierarchy['mean_drift_reduction_after_reset']:.3f} mean reset reduction")

    print(f"\nWrote audit outputs to: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reverify HDFC multi-agent findings from borrower audit logs.")
    parser.add_argument("--input", default="results/borrower_audit_200.csv", help="CSV or JSONL audit log.")
    parser.add_argument("--out-dir", default="results/robustness_200", help="Directory for summary outputs.")
    parser.add_argument("--min-borrowers", type=int, default=200, help="Minimum unique borrowers required for READY status.")
    parser.add_argument("--write-template", help="Write an empty audit CSV template and exit if --input does not exist.")
    parser.add_argument("--no-plot", action="store_true", help="Skip PNG plot generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.write_template:
        template_path = write_template(args.write_template)
        print(f"Wrote audit template: {template_path}")
        if not Path(args.input).exists():
            return

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"\nAudit input not found: {input_path}")
        print(
            "\nThis verifier runs after the 200-borrower experiment. "
            "It does not generate borrower results by itself."
        )
        print("\nNext steps:")
        print("  1. Run your 200-borrower experiment and make it write:")
        print(f"     {input_path}")
        print("  2. Then run this verifier again with the same command.")
        print("\nIf you only wanted the empty logging template, run:")
        print("  python3 verify_200_borrower_run.py --write-template results/borrower_audit_200_template.csv")
        return

    rows = load_rows(input_path)
    if not rows:
        raise ValueError(f"No audit rows found in {args.input}")

    out_dir = Path(args.out_dir)
    summary = build_summary(rows, args.min_borrowers)
    write_json(out_dir / "robustness_summary.json", summary)
    write_metric_tables(summary, out_dir)
    if not args.no_plot:
        plot_summary(summary, out_dir / "Figure_200_Robustness_Audit.png")
    print_console_summary(summary, out_dir)


if __name__ == "__main__":
    main()
