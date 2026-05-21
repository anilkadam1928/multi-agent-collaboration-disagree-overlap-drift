"""Week 5 Tuesday: close the RL reward loop without rerunning LLM agents.

This script uses the already-frozen Scenario 5 combined outputs. Each candidate
decision policy receives +1 / -1 feedback against the Kaggle label, and its
future routing weight is adjusted by +/- 0.05.

It intentionally does not modify the older simulation files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
COMBINED_PATH = ROOT / "data" / "combined_results.csv"
POST_ACCURACY_PATH = ROOT / "data" / "rl_post_accuracy.csv"
WEIGHT_TRACE_PATH = ROOT / "data" / "rl_weight_trace.csv"
REWARDS_EXTENDED_PATH = ROOT / "data" / "rl_rewards_extended.csv"

POLICIES = {
    "BaselineRouter": "baseline_router_decision",
    "Borda": "borda_winner",
    "ConsensAgent": "consensagent_winner",
    "OverlapControlled": "controlled_overlap_final_decision",
    "DriftRouter": "drift_router_decision",
}

INITIAL_WEIGHT = 1.0
REWARD_STEP = 0.05
MIN_WEIGHT = 0.50
MAX_WEIGHT = 1.50


def expected_decision(kaggle_truth_label: str) -> str:
    """Map German Credit labels to the strict final-decision target."""
    return "approve" if str(kaggle_truth_label).lower() == "good" else "reject"


def score_policy_votes(row: pd.Series, weights: dict[str, float]) -> dict[str, float]:
    scores = {"approve": 0.0, "reject": 0.0, "refer": 0.0, "unknown": 0.0}
    for policy_name, column in POLICIES.items():
        decision = str(row[column]).lower()
        if decision not in scores:
            decision = "unknown"
        scores[decision] += weights[policy_name]
    return scores


def choose_weighted_decision(scores: dict[str, float]) -> str:
    # Conservative tie-break: reject, then refer, then approve, then unknown.
    order = {"reject": 0, "refer": 1, "approve": 2, "unknown": 3}
    return sorted(scores, key=lambda d: (-scores[d], order[d]))[0]


def update_weights(row: pd.Series, expected: str, weights: dict[str, float]) -> dict[str, int]:
    rewards: dict[str, int] = {}
    for policy_name, column in POLICIES.items():
        decision = str(row[column]).lower()
        reward = 1 if decision == expected else -1
        rewards[policy_name] = reward
        weights[policy_name] = max(
            MIN_WEIGHT,
            min(MAX_WEIGHT, weights[policy_name] + REWARD_STEP * reward),
        )
    return rewards


def run_rl_reward_loop() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combined = pd.read_csv(COMBINED_PATH)
    weights = {policy_name: INITIAL_WEIGHT for policy_name in POLICIES}
    trace_rows = []
    reward_rows = []

    for idx, row in combined.iterrows():
        expected = expected_decision(row["kaggle_truth_label"])
        scores_before = score_policy_votes(row, weights)
        post_rl_decision = choose_weighted_decision(scores_before)
        post_rl_correct = post_rl_decision == expected
        weights_before = weights.copy()
        rewards = update_weights(row, expected, weights)

        trace_rows.append(
            {
                "run_id": row["run_id"],
                "borrower_id": row["borrower_id"],
                "kaggle_truth_label": row["kaggle_truth_label"],
                "expected_decision": expected,
                "post_rl_decision": post_rl_decision,
                "post_rl_correct": post_rl_correct,
                "scores_before_json": json.dumps(scores_before),
                "weights_before_json": json.dumps(weights_before),
                "weights_after_json": json.dumps(weights),
            }
        )

        for policy_name, column in POLICIES.items():
            decision = str(row[column]).lower()
            reward_rows.append(
                {
                    "run_id": row["run_id"],
                    "borrower_id": row["borrower_id"],
                    "policy_name": policy_name,
                    "policy_decision": decision,
                    "expected_decision": expected,
                    "reward": rewards[policy_name],
                    "weight_after": weights[policy_name],
                }
            )

        print(
            f"RL {idx + 1:02d}/{len(combined)} | {row['borrower_id']} | "
            f"expected={expected} post_rl={post_rl_decision} "
            f"correct={post_rl_correct}",
            flush=True,
        )

    trace_df = pd.DataFrame(trace_rows)
    reward_df = pd.DataFrame(reward_rows)

    pre_rl_accuracy = float(combined["baseline_correct"].mean())
    combined_accuracy = float(combined["combined_final_correct"].mean())
    post_rl_accuracy = float(trace_df["post_rl_correct"].mean())

    summary_df = pd.DataFrame(
        [
            {
                "run_id": "week5_tuesday_rl_loop",
                "profiles_run": int(len(combined)),
                "pre_rl_accuracy": round(pre_rl_accuracy, 4),
                "combined_pre_rl_accuracy": round(combined_accuracy, 4),
                "post_rl_accuracy": round(post_rl_accuracy, 4),
                "delta_vs_baseline": round(post_rl_accuracy - pre_rl_accuracy, 4),
                "delta_vs_combined": round(post_rl_accuracy - combined_accuracy, 4),
                "reward_step": REWARD_STEP,
                "final_weights_json": json.dumps(weights),
                "note": (
                    "RL adjusted policy weights using +1/-1 Kaggle feedback. "
                    "The post-RL score is compared against both the frozen "
                    "baseline and the pre-RL combined pipeline."
                ),
            }
        ]
    )

    POST_ACCURACY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(POST_ACCURACY_PATH, index=False)
    trace_df.to_csv(WEIGHT_TRACE_PATH, index=False)
    reward_df.to_csv(REWARDS_EXTENDED_PATH, index=False)
    return summary_df, trace_df, reward_df


if __name__ == "__main__":
    summary, _, _ = run_rl_reward_loop()
    row = summary.iloc[0]
    print("\nRL REWARD LOOP SUMMARY")
    print(f"Profiles: {int(row['profiles_run'])}")
    print(f"Pre-RL baseline accuracy: {row['pre_rl_accuracy']:.2%}")
    print(f"Pre-RL combined accuracy: {row['combined_pre_rl_accuracy']:.2%}")
    print(f"Post-RL accuracy: {row['post_rl_accuracy']:.2%}")
    print(f"Delta vs baseline: {row['delta_vs_baseline']:+.2%}")
    print(f"Delta vs combined: {row['delta_vs_combined']:+.2%}")
    print(f"Saved: {POST_ACCURACY_PATH}")
    print(f"Saved: {WEIGHT_TRACE_PATH}")
    print(f"Saved: {REWARDS_EXTENDED_PATH}")
