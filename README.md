# Multi-Agent Credit-Risk Governance

This repository contains the shareable code and evidence package for a research
simulation on multi-agent collaboration in a bank-style credit-risk workflow.

The project studies when specialist agents disagree, duplicate work, or drift
away from baseline behaviour during repeated borrower reviews. The experiments
use the public German Credit research dataset as a non-confidential proxy for
borrower review. No real HDFC customer records, KYC documents, salary slips,
bank statements, or internal bank underwriting data are included.

## What is included

- Core simulation and analysis scripts for disagreement, overlap, drift,
  reward feedback, hierarchy, concept drift, and 200-profile robustness checks.
- Public-data CSV outputs in `data/`, `results/`, and `robustness_200_clean/`.
- A curated `evidence_package/` with the CSVs, figures, evidence index, and
  reproducibility notes referenced by the final research report appendix.
- `requirements.txt` from the local Python environment used during the project.

## Evidence package

Start here when reviewing the report evidence:

- `evidence_package/README.md`
- `evidence_package/evidence_index.csv`
- `evidence_package/csv/`
- `evidence_package/figures/`
- `evidence_package/reproducibility_notes.md`

The evidence index maps report claims to the supporting CSV files and figures.
The 200-profile robustness confirmation is provided as supporting simulation
evidence, not as production validation.

## Reproducibility notes

The original experiments were run locally with CrewAI/Ollama-compatible agents.
Some report-generation helper scripts still reflect the original local project
layout, but the shareable evidence files are available in this repository using
the relative paths above.

To install the Python dependencies in a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The local model setup, dataset assumptions, and limitations are documented in
`evidence_package/reproducibility_notes.md`.

## Limitations

This is a research simulation using public proxy data and local LLM-agent
outputs. It is not a production credit model, does not make real lending
decisions, and should not be interpreted as validation on private bank data.
