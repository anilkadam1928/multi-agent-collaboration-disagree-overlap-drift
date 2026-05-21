"""
experiment_audit.py
===================

Audit schema and lightweight CSV logger for the HDFC multi-agent credit-risk
experiments.

The important design choice is one audit row per:

    borrower_id + method + stage + agent_name

Aggregate decisions can use agent_name="" and stage="final".  The verifier can
then recompute accuracy, safety, disagreement, sycophancy, overlap, drift,
concept-drift, reward-loop, and hierarchy findings from the saved evidence.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


AUDIT_FIELDNAMES = [
    # Run/profile identity
    "run_id",
    "logged_at_utc",
    "borrower_id",
    "profile_index",
    "dataset_row_id",
    "scenario",
    "method",
    "stage",
    # Agent identity
    "agent_name",
    "agent_role",
    # Final decision quality and safety
    "truth_label",
    "expected_decision",
    "final_decision",
    "agent_decision",
    "confidence",
    "refer_reason",
    # Disagreement and resolution
    "precommit_decision",
    "post_discussion_decision",
    "changed_after_discussion",
    "resolution_method",
    "borda_winner",
    "consensagent_winner",
    "borda_score_approve",
    "borda_score_refer",
    "borda_score_reject",
    "consensagent_score_approve",
    "consensagent_score_refer",
    "consensagent_score_reject",
    # Sycophancy detection
    "sycophancy_flagged",
    "similarity_score",
    "similar_to_agent",
    # Overlap and causal-chain evidence
    "overlap_detected",
    "disagreement_detected",
    "task_intent",
    "task_scope",
    "overlap_cluster",
    "duplicate_invocations_raw",
    "additional_resolution_invocations",
    "duplicate_invocations_after_lce",
    "lce_removed_invocations",
    "raw_redundancy_index",
    "controlled_redundancy_index",
    "leader_agent",
    "leader_changed",
    "interaction_group",
    "group_kl_drift_score",
    # Behavioural drift, memory compaction, and ABA
    "drift_condition",
    "checkpoint",
    "drift_score",
    "baseline_drift_score",
    "post_reset_drift_score",
    "drift_threshold",
    "drift_exceeded",
    "aba_applied",
    "aba_event_count",
    "aba_anchor_id",
    "memory_tokens_before",
    "memory_tokens_after",
    "agent_skipped_for_drift",
    # Concept drift
    "concept_batch_id",
    "concept_correlation",
    "concept_drift_threshold",
    "concept_drift_flag",
    "stress_test",
    # Reward loop
    "reward_policy",
    "reward_value",
    "reward_correct",
    "route_weight_before",
    "route_weight_after",
    "route_weight_delta",
    # Hierarchy governance
    "hierarchy_level",
    "manager_name",
    "manager_decision",
    "manager_disagreement",
    "l1_triggered",
    "reset_issued",
    "reset_agents",
    # Parser/data-quality guardrails
    "parser_valid",
    "parser_error",
    "reasoning_summary",
    "extra_json",
]


REQUIRED_FIELDS_BY_FINDING = {
    "sample_coverage": [
        "borrower_id",
        "truth_label",
    ],
    "decision_quality_and_safety": [
        "borrower_id",
        "method",
        "truth_label",
        "final_decision",
    ],
    "disagreement_resolution": [
        "borrower_id",
        "method",
        "agent_name",
        "precommit_decision",
        "final_decision",
        "borda_winner",
        "consensagent_winner",
    ],
    "sycophancy_reduction": [
        "borrower_id",
        "agent_name",
        "precommit_decision",
        "final_decision",
        "sycophancy_flagged",
        "similarity_score",
    ],
    "overlap_reduction": [
        "borrower_id",
        "duplicate_invocations_raw",
        "duplicate_invocations_after_lce",
        "raw_redundancy_index",
        "controlled_redundancy_index",
        "leader_agent",
    ],
    "behavioural_drift_and_aba": [
        "borrower_id",
        "agent_name",
        "drift_condition",
        "checkpoint",
        "drift_score",
        "drift_threshold",
        "aba_applied",
        "aba_event_count",
    ],
    "causal_chain": [
        "borrower_id",
        "overlap_detected",
        "disagreement_detected",
        "additional_resolution_invocations",
        "duplicate_invocations_after_lce",
        "lce_removed_invocations",
        "interaction_group",
        "group_kl_drift_score",
    ],
    "concept_drift_layer": [
        "concept_batch_id",
        "concept_correlation",
        "concept_drift_threshold",
        "concept_drift_flag",
    ],
    "reward_loop": [
        "borrower_id",
        "method",
        "truth_label",
        "final_decision",
        "reward_policy",
        "reward_value",
        "route_weight_before",
        "route_weight_after",
    ],
    "hierarchy_governance": [
        "borrower_id",
        "method",
        "hierarchy_level",
        "manager_disagreement",
        "l1_triggered",
        "reset_issued",
        "baseline_drift_score",
        "post_reset_drift_score",
    ],
}


TRUE_VALUES = {"1", "true", "yes", "y", "t"}
FALSE_VALUES = {"0", "false", "no", "n", "f"}


def normalise_decision(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"approve", "approved", "accept", "accepted", "good"}:
        return "approve"
    if text in {"reject", "rejected", "decline", "declined", "bad"}:
        return "reject"
    if text in {"refer", "referred", "review", "manual_review", "manual review", "escalate"}:
        return "refer"
    return text


def normalise_truth(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"good", "approve", "approved", "1", "positive"}:
        return "good"
    if text in {"bad", "reject", "rejected", "0", "negative"}:
        return "bad"
    return text


def expected_decision_for_truth(value: Any) -> str:
    truth = normalise_truth(value)
    if truth == "good":
        return "approve"
    if truth == "bad":
        return "reject"
    return ""


def bool_value(value: Any) -> bool | None:
    text = str(value or "").strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    return None


def float_value(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_value(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def write_template(path: str | Path) -> Path:
    """Create an empty CSV with the complete audit header."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDNAMES)
        writer.writeheader()
    return output


class AuditLogger:
    """Small append-only CSV logger for experiment runners.

    Example:

        logger = AuditLogger("results/borrower_audit_200.csv")
        logger.log(
            run_id="week6_200",
            borrower_id="B037",
            method="CONSENSAGENT",
            stage="final",
            truth_label="bad",
            final_decision="refer",
        )
        logger.close()
    """

    def __init__(self, path: str | Path, append: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open(
            "a" if append else "w",
            newline="",
            encoding="utf-8",
        )
        self._writer = csv.DictWriter(
            self._handle,
            fieldnames=AUDIT_FIELDNAMES,
            extrasaction="ignore",
        )
        if not append or self.path.stat().st_size == 0:
            self._writer.writeheader()

    def log(self, **event: Any) -> None:
        row = {field: "" for field in AUDIT_FIELDNAMES}
        row.update({key: value for key, value in event.items() if key in row})

        if not row["logged_at_utc"]:
            row["logged_at_utc"] = datetime.now(timezone.utc).isoformat()
        if not row["expected_decision"] and row["truth_label"]:
            row["expected_decision"] = expected_decision_for_truth(row["truth_label"])

        extras = {key: value for key, value in event.items() if key not in row}
        if extras:
            row["extra_json"] = json.dumps(extras, sort_keys=True, default=str)

        self._writer.writerow(row)
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "AuditLogger":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def audit_row_from_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a schema-aligned row without writing it."""
    row = {field: "" for field in AUDIT_FIELDNAMES}
    row.update({key: value for key, value in data.items() if key in row})
    if not row["expected_decision"] and row["truth_label"]:
        row["expected_decision"] = expected_decision_for_truth(row["truth_label"])
    extras = {key: value for key, value in data.items() if key not in row}
    if extras:
        row["extra_json"] = json.dumps(extras, sort_keys=True, default=str)
    return row
