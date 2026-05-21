"""
generate_figure1.py
===================
Generates Figure_1_Disagreement_Rate.png for the disagreement module.

Run:
    python generate_figure1.py

Reads:  results/disagreement_results.csv
Saves:  results/Figure_1_Disagreement_Rate.png
"""

import csv
import json
import os
from collections import defaultdict

# matplotlib is the only dependency — already in most Python installs.
# If missing: pip install matplotlib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
INPUT_PATH  = "results/disagreement_results.csv"
OUTPUT_PATH = "results/Figure_1_Disagreement_Rate.png"

# ── baseline values (from your frozen baseline_results.csv run) ─────────────
# Replace these with your actual baseline numbers if different.
BASELINE_DISAGREEMENT = 0.80   # ~all-refer baseline = effectively 0 disagreement variety
BASELINE_BORDA_ACC    = 0.36   # random-ish baseline accuracy
BASELINE_CONSENS_ACC  = 0.36
BASELINE_SYCO_RATE    = 0.40


# ══════════════════════════════════════════════════════════════════════════════
# LOAD AND COMPUTE METRICS
# ══════════════════════════════════════════════════════════════════════════════

def load_metrics(path: str) -> dict:
    """Read disagreement_results.csv and compute all four comparison metrics."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No data found in {path}")

    # One row per agent per borrower — group by borrower first
    borrowers = defaultdict(list)
    for row in rows:
        borrowers[row["borrower_id"]].append(row)

    total = len(borrowers)

    # ── disagreement rate ──────────────────────────────────────────────────
    # A borrower has disagreement if not all agents gave the same final decision
    disagreement_count = 0
    for bid, agent_rows in borrowers.items():
        decisions = set(r["final_decision"] for r in agent_rows)
        if len(decisions) > 1:
            disagreement_count += 1
    disagreement_rate = disagreement_count / total

    # ── borda and consensagent accuracy ───────────────────────────────────
    # Take the first agent row per borrower (borda_winner is same for all agents in a borrower)
    borda_correct  = 0
    consens_correct = 0
    borda_refer_count = 0

    for bid, agent_rows in borrowers.items():
        row   = agent_rows[0]
        label = row["kaggle_truth_label"].strip().lower()
        borda = row["borda_winner"].strip().lower()
        cons  = row["consensagent_winner"].strip().lower()

        expected = "approve" if label == "good" else "reject"
        if borda == expected:
            borda_correct += 1
        if cons == expected:
            consens_correct += 1
        if borda == "refer":
            borda_refer_count += 1

    borda_accuracy   = borda_correct  / total
    consens_accuracy = consens_correct / total
    borda_refer_rate = borda_refer_count / total

    # ── sycophancy rate ────────────────────────────────────────────────────
    total_changes = sum(
        1 for r in rows
        if r["precommit_decision"] != r["final_decision"]
    )
    total_flags = sum(
        1 for r in rows
        if str(r.get("sycophancy_flagged", "")).lower() in {"true", "1", "yes"}
    )
    syco_rate = total_flags / total_changes if total_changes > 0 else 0.0

    # ── per-borrower decision distribution ────────────────────────────────
    borda_winners = [agent_rows[0]["borda_winner"].lower()
                     for agent_rows in borrowers.values()]
    dist = {
        "approve": borda_winners.count("approve"),
        "refer":   borda_winners.count("refer"),
        "reject":  borda_winners.count("reject"),
    }

    return {
        "total":              total,
        "disagreement_rate":  disagreement_rate,
        "borda_accuracy":     borda_accuracy,
        "consens_accuracy":   consens_accuracy,
        "syco_rate":          syco_rate,
        "borda_refer_rate":   borda_refer_rate,
        "total_changes":      total_changes,
        "total_flags":        total_flags,
        "borda_dist":         dist,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_figure1(metrics: dict, output_path: str) -> None:
    """
    4-panel figure:
      Panel A — Disagreement rate: baseline vs intervention
      Panel B — Accuracy: Borda vs CONSENSAGENT vs baseline
      Panel C — Sycophancy rate: baseline vs intervention
      Panel D — Borda decision distribution (approve / refer / reject)
    """
    # ── colour palette ─────────────────────────────────────────────────────
    BLUE   = "#1f4e79"   # HDFC dark blue
    ORANGE = "#c55a11"   # intervention / highlight
    GREY   = "#a6a6a6"   # baseline
    GREEN  = "#375623"   # correct / positive
    RED    = "#c00000"   # sycophancy / negative

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Figure 1 — Disagreement Module Results\n"
        "Multi-Agent Collaboration: When and Why Agents Disagree\n"
        "HDFC Bank | Integrated Risk Management | Anil Khushalrao Kadam",
        fontsize=11, fontweight="bold", y=1.01,
    )

    # ── Panel A: Disagreement Rate ─────────────────────────────────────────
    ax = axes[0, 0]
    labels = ["Baseline\n(no intervention)", "After Intervention\n(Borda + Pre-commit)"]
    values = [BASELINE_DISAGREEMENT, metrics["disagreement_rate"]]
    colors = [GREY, BLUE]
    bars = ax.bar(labels, values, color=colors, width=0.45, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Disagreement Rate", fontsize=10)
    ax.set_title("(A) Disagreement Rate", fontsize=11, fontweight="bold")
    ax.axhline(y=BASELINE_DISAGREEMENT, color=GREY, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel B: Accuracy Comparison ──────────────────────────────────────
    ax = axes[0, 1]
    acc_labels  = ["Baseline", "Borda Count", "CONSENSAGENT"]
    acc_values  = [BASELINE_BORDA_ACC, metrics["borda_accuracy"], metrics["consens_accuracy"]]
    acc_colors  = [GREY, BLUE, GREEN]
    bars = ax.bar(acc_labels, acc_values, color=acc_colors, width=0.45,
                  edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, acc_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy vs Kaggle Truth Label", fontsize=10)
    ax.set_title("(B) Final Accuracy Comparison", fontsize=11, fontweight="bold")
    ax.axhline(y=BASELINE_BORDA_ACC, color=GREY, linestyle="--", linewidth=1,
               alpha=0.6, label=f"Baseline ({BASELINE_BORDA_ACC:.2f})")
    ax.axhline(y=0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.4, label="Random (0.50)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate CONSENSAGENT improvement
    improvement = metrics["consens_accuracy"] - BASELINE_CONSENS_ACC
    ax.annotate(
        f"+{improvement:.2f} vs baseline",
        xy=(2, metrics["consens_accuracy"]),
        xytext=(1.5, metrics["consens_accuracy"] + 0.12),
        fontsize=8, color=GREEN, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2),
    )

    # ── Panel C: Sycophancy Rate ───────────────────────────────────────────
    ax = axes[1, 0]
    s_labels = ["Baseline\n(expected)", "After Intervention\n(cosine detector)"]
    s_values = [BASELINE_SYCO_RATE, metrics["syco_rate"]]
    s_colors = [GREY, ORANGE]
    bars = ax.bar(s_labels, s_values, color=s_colors, width=0.45,
                  edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, s_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Sycophancy Rate\n(flagged changes / total changes)", fontsize=10)
    ax.set_title("(C) Sycophancy Rate", fontsize=11, fontweight="bold")

    # Annotate reduction
    reduction = BASELINE_SYCO_RATE - metrics["syco_rate"]
    ax.annotate(
        f"−{reduction:.2f} reduction",
        xy=(1, metrics["syco_rate"]),
        xytext=(0.6, metrics["syco_rate"] + 0.15),
        fontsize=8, color=ORANGE, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2),
    )
    ax.text(0.97, 0.95,
            f"Flags: {metrics['total_flags']} / {metrics['total_changes']} changes",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            color="dimgrey")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel D: Borda Decision Distribution ──────────────────────────────
    ax = axes[1, 1]
    dist   = metrics["borda_dist"]
    d_labels = ["Approve", "Refer\n(abstain)", "Reject"]
    d_values = [dist["approve"], dist["refer"], dist["reject"]]
    d_colors = [GREEN, GREY, RED]
    bars = ax.bar(d_labels, d_values, color=d_colors, width=0.45,
                  edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, d_values):
        pct = val / metrics["total"] * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val}\n({pct:.0f}%)", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_ylim(0, metrics["total"] * 1.35)
    ax.set_ylabel("Number of Borrowers", fontsize=10)
    ax.set_title("(D) Borda Count Decision Distribution\n(20 profiles)", fontsize=11, fontweight="bold")
    ax.set_yticks(range(0, metrics["total"] + 1, 2))
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Footer ─────────────────────────────────────────────────────────────
    fig.text(
        0.5, -0.02,
        f"German Credit Dataset (Kaggle) | {metrics['total']} profiles | "
        "CrewAI v0.28.8 | Gemma (LM Studio) | Pre-commit + Borda Count + CONSENSAGENT + Cosine Sycophancy Detector",
        ha="center", fontsize=7.5, color="dimgrey",
    )

    plt.tight_layout(pad=2.5)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Reading {INPUT_PATH} …")
    metrics = load_metrics(INPUT_PATH)

    print(f"\nMetrics computed from {metrics['total']} borrowers:")
    print(f"  Disagreement rate:     {metrics['disagreement_rate']:.2f}")
    print(f"  Borda accuracy:        {metrics['borda_accuracy']:.2f}")
    print(f"  CONSENSAGENT accuracy: {metrics['consens_accuracy']:.2f}")
    print(f"  Sycophancy rate:       {metrics['syco_rate']:.2f}  "
          f"(flags={metrics['total_flags']}, changes={metrics['total_changes']})")
    print(f"  Borda refer rate:      {metrics['borda_refer_rate']:.2f}")
    print(f"  Borda distribution:    {metrics['borda_dist']}")

    print(f"\nGenerating Figure 1 …")
    plot_figure1(metrics, OUTPUT_PATH)
    print("Done. Wednesday deliverable complete.")
