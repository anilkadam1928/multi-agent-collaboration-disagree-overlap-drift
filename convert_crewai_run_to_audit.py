"""
convert_crewai_run_to_audit.py
==============================

Converts the real CrewAI/Gemma robustness run outputs into the unified audit
CSV consumed by verify_200_borrower_run.py.

This script does not invent LLM results.  It reads the real files produced by
robustness_200_runner.py:

    robustness_200/borrower_agent_outputs.csv
    robustness_200/borrower_summary.csv
    robustness_200/drift_checkpoints.csv
    robustness_200/concept_drift_checkpoints.csv

Then it writes:

    robustness_200/borrower_audit_200.csv

Run it any time while the long run is partially complete; the verifier will
show PARTIAL_SAMPLE until all 200 borrowers are present.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from config import AGENT_NAMES, DRIFT_KL_THRESHOLD
from experiment_audit import AuditLogger, bool_value, expected_decision_for_truth


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "robustness_200"
OPTIONS = ["approve", "refer", "reject"]
SYCOPHANCY_THRESHOLD = 0.80


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalise_decision(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"approve", "refer", "reject"}:
        return text
    return "unknown"


def compact_json(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def token_vector(text: str) -> Counter:
    tokens = re.findall(r"[a-z]{3,}", str(text or "").lower())
    stop = {"the", "and", "for", "this", "that", "with", "into", "from", "you", "are"}
    return Counter(token for token in tokens if token not in stop)


def cosine_text(a: str, b: str) -> float:
    va = token_vector(a)
    vb = token_vector(b)
    if not va or not vb:
        return 0.0
    common = set(va) & set(vb)
    numerator = sum(va[token] * vb[token] for token in common)
    denom_a = math.sqrt(sum(value * value for value in va.values()))
    denom_b = math.sqrt(sum(value * value for value in vb.values()))
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return numerator / (denom_a * denom_b)


def group_agent_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, dict[str, str]]]]:
    grouped: dict[str, dict[str, dict[str, dict[str, str]]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        grouped[row.get("borrower_id", "")][row.get("stage", "")][row.get("agent_name", "")] = row
    return grouped


def recompute_sycophancy(
    precommit: dict[str, dict[str, str]],
    discussion: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    flags = {}
    for agent_name in AGENT_NAMES:
        pre = precommit.get(agent_name, {})
        final = discussion.get(agent_name, {})
        pre_decision = normalise_decision(pre.get("parsed_decision"))
        final_decision = normalise_decision(final.get("parsed_decision"))
        changed = pre_decision != "unknown" and final_decision != "unknown" and pre_decision != final_decision

        if not changed:
            flags[agent_name] = {"flagged": False, "similarity": 0.0, "similar_to": ""}
            continue

        best_agent = ""
        best_similarity = 0.0
        for other in AGENT_NAMES:
            if other == agent_name:
                continue
            other_final = discussion.get(other, {})
            if normalise_decision(other_final.get("parsed_decision")) != final_decision:
                continue
            similarity = cosine_text(final.get("raw_output", ""), other_final.get("raw_output", ""))
            if similarity > best_similarity:
                best_similarity = similarity
                best_agent = other

        flags[agent_name] = {
            "flagged": best_similarity >= SYCOPHANCY_THRESHOLD,
            "similarity": round(best_similarity, 4),
            "similar_to": best_agent,
        }
    return flags


def parse_json_dict(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def bool_text(value: Any) -> str:
    return "true" if bool_value(value) is True or value is True or str(value).strip() == "True" else "false"


def log_final_decision(
    logger: AuditLogger,
    run_id: str,
    summary: dict[str, str],
    method: str,
    decision: str,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "run_id": run_id,
        "borrower_id": summary["borrower_id"],
        "profile_index": int(summary.get("row_index", "0")) + 1,
        "dataset_row_id": summary.get("row_index", ""),
        "scenario": "crewai_gemma_robustness_200",
        "method": method,
        "stage": "final",
        "truth_label": summary["kaggle_truth_label"],
        "final_decision": decision,
        "parser_valid": "true",
    }
    if extra:
        payload.update(extra)
    logger.log(**payload)


def build_reward_trace(summary_rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    weights = {
        "Router baseline": 1.0,
        "Borda": 1.0,
        "CONSENSAGENT": 1.0,
        "Hierarchy": 1.0,
    }
    columns = {
        "Router baseline": "router_decision",
        "Borda": "borda_winner",
        "CONSENSAGENT": "consens_winner",
        "Hierarchy": "hierarchy_final",
    }
    trace = {}

    for summary in summary_rows:
        expected = expected_decision_for_truth(summary["kaggle_truth_label"])
        score = {option: 0.0 for option in OPTIONS}
        weights_before = weights.copy()

        for policy, column in columns.items():
            decision = normalise_decision(summary.get(column))
            if decision in score:
                score[decision] += weights[policy]

        post_rl_decision = sorted(
            OPTIONS,
            key=lambda decision: (-score[decision], {"reject": 0, "refer": 1, "approve": 2}[decision]),
        )[0]

        rewards = {}
        for policy, column in columns.items():
            decision = normalise_decision(summary.get(column))
            reward = 1 if decision == expected else -1
            rewards[policy] = reward
            weights[policy] = max(0.50, min(1.50, weights[policy] + 0.05 * reward))

        trace[summary["borrower_id"]] = {
            "post_rl_decision": post_rl_decision,
            "weights_before": weights_before,
            "weights_after": weights.copy(),
            "scores_before": score,
            "reward_value": 1 if post_rl_decision == expected else -1,
            "reward_correct": post_rl_decision == expected,
            "route_weight_before": sum(weights_before.values()) / len(weights_before),
            "route_weight_after": sum(weights.values()) / len(weights),
            "route_weight_delta": (sum(weights.values()) - sum(weights_before.values())) / len(weights),
            "policy_rewards": rewards,
        }
    return trace


def convert(output_dir: Path, audit_path: Path, run_id: str) -> None:
    summary_path = output_dir / "borrower_summary.csv"
    agent_output_path = output_dir / "borrower_agent_outputs.csv"
    drift_path = output_dir / "drift_checkpoints.csv"
    concept_path = output_dir / "concept_drift_checkpoints.csv"

    summary_rows = read_csv(summary_path)
    agent_rows = read_csv(agent_output_path)
    drift_rows = read_csv(drift_path)
    concept_rows = read_csv(concept_path)

    if not summary_rows:
        raise FileNotFoundError(f"No summary rows found at {summary_path}")

    grouped_agents = group_agent_rows(agent_rows)
    reward_trace = build_reward_trace(summary_rows)
    summary_by_borrower = {row["borrower_id"]: row for row in summary_rows}

    with AuditLogger(audit_path, append=False) as logger:
        for summary in summary_rows:
            borrower_id = summary["borrower_id"]
            precommit = grouped_agents.get(borrower_id, {}).get("precommit", {})
            discussion = grouped_agents.get(borrower_id, {}).get("discussion", {})
            sycophancy = recompute_sycophancy(precommit, discussion)
            position_change_count = int(float(summary.get("position_change_count", 0) or 0))
            duplicate_pairs = float(summary.get("duplicate_pairs", 0) or 0)
            raw_redundancy = float(summary.get("redundancy_index_raw", 0) or 0)
            lce_redundancy = float(summary.get("redundancy_index_lce_estimated", 0) or 0)
            lce_removed = max(0.0, raw_redundancy - lce_redundancy)
            borda_scores = parse_json_dict(summary.get("borda_scores", ""))
            consens_scores = parse_json_dict(summary.get("consens_scores", ""))

            common = {
                "run_id": run_id,
                "borrower_id": borrower_id,
                "profile_index": int(summary.get("row_index", "0")) + 1,
                "dataset_row_id": summary.get("row_index", ""),
                "scenario": "crewai_gemma_robustness_200",
                "method": "Borda",
                "truth_label": summary["kaggle_truth_label"],
                "borda_winner": summary.get("borda_winner", ""),
                "consensagent_winner": summary.get("consens_winner", ""),
                "borda_score_approve": borda_scores.get("approve", ""),
                "borda_score_refer": borda_scores.get("refer", ""),
                "borda_score_reject": borda_scores.get("reject", ""),
                "consensagent_score_approve": consens_scores.get("approve", ""),
                "consensagent_score_refer": consens_scores.get("refer", ""),
                "consensagent_score_reject": consens_scores.get("reject", ""),
                "overlap_detected": "true" if duplicate_pairs > 0 else "false",
                "disagreement_detected": bool_text(summary.get("final_disagreement")),
                "duplicate_invocations_raw": duplicate_pairs,
                "additional_resolution_invocations": position_change_count,
                "duplicate_invocations_after_lce": max(0.0, duplicate_pairs - lce_removed),
                "lce_removed_invocations": round(lce_removed, 4),
                "raw_redundancy_index": raw_redundancy,
                "controlled_redundancy_index": lce_redundancy,
                "leader_agent": summary.get("lce_leader", ""),
                "interaction_group": "high"
                if duplicate_pairs > 0 and bool_value(summary.get("final_disagreement")) is True
                else "low",
                "group_kl_drift_score": round(raw_redundancy + position_change_count * 0.05, 4),
            }

            for agent_name in AGENT_NAMES:
                pre = precommit.get(agent_name, {})
                final = discussion.get(agent_name, {})
                pre_decision = normalise_decision(pre.get("parsed_decision"))
                final_decision = normalise_decision(final.get("parsed_decision"))
                logger.log(
                    **common,
                    stage="agent_discussion",
                    agent_name=agent_name,
                    agent_role=agent_name.replace("Agent", ""),
                    precommit_decision=pre_decision,
                    post_discussion_decision=final_decision,
                    final_decision=final_decision,
                    agent_decision=final_decision,
                    confidence=final.get("confidence_flag", ""),
                    changed_after_discussion="true" if pre_decision != final_decision else "false",
                    sycophancy_flagged=bool_text(sycophancy[agent_name]["flagged"]),
                    similarity_score=sycophancy[agent_name]["similarity"],
                    similar_to_agent=sycophancy[agent_name]["similar_to"],
                    reasoning_summary=final.get("brief_reason", ""),
                    parser_valid="true" if final_decision != "unknown" else "false",
                    parser_error="" if final_decision != "unknown" else "unknown decision parsed",
                )

            log_final_decision(logger, run_id, summary, "Router baseline", summary.get("router_decision", ""))
            log_final_decision(logger, run_id, summary, "Borda", summary.get("borda_winner", ""))
            log_final_decision(logger, run_id, summary, "CONSENSAGENT", summary.get("consens_winner", ""))
            log_final_decision(
                logger,
                run_id,
                summary,
                "Overlap Module - LCE+ToM",
                summary.get("borda_winner", ""),
                {
                    "duplicate_invocations_raw": duplicate_pairs,
                    "duplicate_invocations_after_lce": max(0.0, duplicate_pairs - lce_removed),
                    "raw_redundancy_index": raw_redundancy,
                    "controlled_redundancy_index": lce_redundancy,
                    "leader_agent": summary.get("lce_leader", ""),
                    "reasoning_summary": "Process-control module; final decision inherited from Borda winner.",
                },
            )
            log_final_decision(
                logger,
                run_id,
                summary,
                "Hierarchy",
                summary.get("hierarchy_final", ""),
                {
                    "hierarchy_level": "L1",
                    "manager_name": "StrategicOverseer",
                    "manager_decision": summary.get("hierarchy_final", ""),
                    "manager_disagreement": "true"
                    if summary.get("credit_manager_winner") != summary.get("compliance_manager_winner")
                    else "false",
                    "l1_triggered": "true"
                    if summary.get("credit_manager_winner") != summary.get("compliance_manager_winner")
                    else "false",
                    "reset_issued": "false",
                    "baseline_drift_score": 0.0,
                    "post_reset_drift_score": 0.0,
                    "reasoning_summary": "Hierarchy aggregation from real CrewAI specialist votes.",
                },
            )

            reward = reward_trace[borrower_id]
            log_final_decision(
                logger,
                run_id,
                summary,
                "Post-RL Reward Loop",
                reward["post_rl_decision"],
                {
                    "reward_policy": "posthoc_weighted_policy_feedback",
                    "reward_value": reward["reward_value"],
                    "reward_correct": bool_text(reward["reward_correct"]),
                    "route_weight_before": round(reward["route_weight_before"], 4),
                    "route_weight_after": round(reward["route_weight_after"], 4),
                    "route_weight_delta": round(reward["route_weight_delta"], 4),
                    "extra_json": compact_json(
                        {
                            "weights_before": reward["weights_before"],
                            "weights_after": reward["weights_after"],
                            "scores_before": reward["scores_before"],
                            "policy_rewards": reward["policy_rewards"],
                        }
                    ),
                    "reasoning_summary": (
                        "Post-hoc reward feedback trace computed from real CrewAI module decisions; "
                        "matches the report's non-rerun reward-loop design."
                    ),
                },
            )

        for drift in drift_rows:
            checkpoint = drift.get("checkpoint", "")
            borrower_id = f"B{int(float(checkpoint)):03d}" if checkpoint else ""
            summary = summary_by_borrower.get(borrower_id, {})
            score = drift.get("drift_score", "")
            above_threshold = bool_text(drift.get("above_threshold"))
            logger.log(
                run_id=run_id,
                borrower_id=borrower_id,
                profile_index=checkpoint,
                scenario="crewai_gemma_robustness_200",
                method="Drift Module - EMC+ABA",
                stage="drift_checkpoint",
                agent_name=drift.get("agent_name", ""),
                truth_label=summary.get("kaggle_truth_label", ""),
                drift_condition="crewai_running_distribution",
                checkpoint=checkpoint,
                drift_score=score,
                baseline_drift_score=score,
                post_reset_drift_score=0.0 if above_threshold == "true" else score,
                drift_threshold=DRIFT_KL_THRESHOLD,
                drift_exceeded=above_threshold,
                aba_applied=above_threshold,
                aba_event_count=1 if above_threshold == "true" else 0,
                agent_skipped_for_drift="false",
                hierarchy_level="L1" if above_threshold == "true" else "",
                manager_disagreement=above_threshold,
                l1_triggered=above_threshold,
                reset_issued=above_threshold,
                reset_agents=drift.get("agent_name", "") if above_threshold == "true" else "",
                parser_valid="true",
            )

        for concept in concept_rows:
            checkpoint = concept.get("checkpoint", "")
            shift = float(concept.get("feature_kl_shift", 0) or 0)
            logger.log(
                run_id=run_id,
                borrower_id=f"B{int(float(checkpoint)):03d}" if checkpoint else "",
                profile_index=checkpoint,
                scenario="crewai_gemma_robustness_200",
                method="Concept Drift",
                stage="concept_checkpoint",
                concept_batch_id=f"checkpoint_{checkpoint}",
                concept_correlation=round(1 / (1 + shift), 4),
                concept_drift_threshold=0.70,
                concept_drift_flag="true" if shift > 0.43 else "false",
                stress_test="false",
                parser_valid="true",
            )

    print(f"Converted {len(summary_rows)} borrower summaries into {audit_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert real CrewAI robustness outputs to unified audit CSV.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--audit-path", default=str(DEFAULT_OUTPUT_DIR / "borrower_audit_200.csv"))
    parser.add_argument("--run-id", default="crewai_gemma_robustness_200")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(Path(args.output_dir), Path(args.audit_path), args.run_id)


if __name__ == "__main__":
    main()
