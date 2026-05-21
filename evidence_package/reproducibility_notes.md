# Reproducibility Notes

## Dataset
The experiments use the public German Credit research dataset as a non-confidential proxy for borrower review. Borrower profiles are generated from structured fields such as age, job type, housing status, savings/checking status, credit amount, duration, and purpose.

## Local model stack
The agent pipeline was run locally using CrewAI/Ollama-compatible agents. The code records the primary local model candidate as `google/gemma-4-e4b` with environment-variable overrides such as `OLLAMA_MODEL` and `OLLAMA_WEAK_MODEL` where applicable. The report therefore describes the system as a local LLM simulation rather than a cloud deployment.

## Evidence discipline
The final report does not embed full raw CSV dumps. Instead, Appendix C and this evidence package map every major quantitative claim to a local CSV or figure path.

## Limitations
Because the dataset has no real documents, document-level checks such as salary-slip verification, bank-statement validation, KYC matching, and forged-document detection are approximated through structured fields and missing-evidence reasoning.
