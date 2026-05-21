# Evidence Package: Multi-Agent Credit-Risk Governance Simulation

This folder is the shareable evidence bundle for the final research report.
It is designed to be reviewed from this public repository.

## How to read this package

- `evidence_index.csv` maps report claims to the exact CSV files that support them.
- `csv/` contains copied evidence CSVs from the local project run.
- `figures/` contains copied final figures used or indexed in the report.
- `reproducibility_notes.md` explains the local model setup, dataset, and limitations.

## Main evidence files

- `baseline_results.csv`: frozen baseline agent decisions.
- `disagreement_results.csv`: pre-commit, discussion, Borda, CONSENSAGENT, and sycophancy fields.
- `overlap_results.csv`: TAP intent board, LCE leader, skipped agents, and redundancy index.
- `drift_no_emc.csv`, `drift_with_emc.csv`, `drift_with_aba.csv`: KL-style drift comparisons.
- `dual_results.csv`: causal-chain evidence linking overlap, disagreement, resolution, renewed overlap, and drift-risk proxy.
- `hierarchy_results.csv`: three-level governance hierarchy outputs.
- `metrics_summary.csv`: 200-profile robustness confirmation summary.

## Important note

The project uses public German Credit data and local simulated LLM-agent outputs. It does not use real bank customer files, KYC documents, salary slips, bank statements, or internal HDFC data.
