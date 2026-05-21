from __future__ import annotations

import argparse
from textwrap import dedent

from crewai import Crew, Process, Task

from agents import (
    PRIMARY_MODEL_NAME,
    WEAK_MODEL_NAME,
    compliance_agent,
    credit_agent,
    fraud_agent,
    income_agent,
    router_manager,
    summariser_agent,
    weak_model_agent,
)


HELLO_WORLD_CASE = dedent(
    """
    Applicant: Amit Sharma
    Monthly Income: 50000
    Employment Type: Salaried
    Credit Score: 710
    Existing EMI: 12000
    Loan Amount Requested: 500000
    Documents Submitted: PAN, Aadhaar, salary slips, bank statements
    Concern: One month of bank statement looks slightly inconsistent
    """
).strip()

FULL_PIPELINE_CASE = dedent(
    """
    Applicant: Neha Verma
    Monthly Income: 82000
    Employment Type: Salaried
    Employer: RapidAxis Services Pvt Ltd
    Credit Score: 684
    Existing EMI: 26500
    Loan Amount Requested: 900000
    Residence: Pune
    Documents Submitted: PAN, Aadhaar, 6 months bank statements, 3 salary slips, Form 16
    Red Flags:
    - Salary credit amount does not match one payslip exactly.
    - One bank statement month shows an unexplained large cash deposit.
    - Current EMI burden is already moderate to high.
    - Employer name formatting is slightly inconsistent across documents.
    Requested Outcome: Recommend approve, reject, or send for manual review with reasons.
    """
).strip()


FORMAT_RULE = (
    "Formatting rule: keep the answer concise, do not include hidden reasoning, "
    "and end with `Final Answer:` followed by the actual answer."
)


def with_format_rule(body: str) -> str:
    return f"{body}\n\n{FORMAT_RULE}"


def build_hello_world_crew(loan_case: str) -> Crew:
    income_task = Task(
        description=with_format_rule(
            "Review this loan application and provide a short income verification "
            f"assessment:\n\n{loan_case}"
        ),
        agent=income_agent,
        expected_output=(
            "A short assessment of income stability, document sufficiency, and key risk flags."
        ),
    )

    credit_task = Task(
        description=with_format_rule(
            "Review this same loan application and provide a short credit assessment. "
            "Use the income review context if helpful.\n\n"
            f"{loan_case}"
        ),
        agent=credit_agent,
        context=[income_task],
        expected_output=(
            "A short assessment of creditworthiness, liabilities, and lending risk."
        ),
    )

    return Crew(
        agents=[income_agent, credit_agent],
        tasks=[income_task, credit_task],
        process=Process.sequential,
        verbose=True,
    )


def build_full_pipeline_crew(loan_case: str) -> Crew:
    router_task = Task(
        description=with_format_rule(
            "Create the routing plan for this loan file. Decide review order, specialist focus, "
            f"and the top three risks.\n\n{loan_case}"
        ),
        agent=router_manager,
        expected_output=(
            "A routing memo with review sequence, top risks, and the recommended path."
        ),
    )

    income_task = Task(
        description=with_format_rule(
            "Perform the income review. Verify earnings consistency, sufficiency of submitted "
            f"proof, and whether the income can support the requested loan.\n\n{loan_case}"
        ),
        agent=income_agent,
        context=[router_task],
        expected_output=(
            "An income-verification note with findings, anomalies, and a risk rating."
        ),
    )

    fraud_task = Task(
        description=with_format_rule(
            "Perform the Stage 2 fraud review. Focus on suspicious patterns, document mismatch, "
            f"and whether this case should be escalated for manual investigation.\n\n{loan_case}"
        ),
        agent=fraud_agent,
        context=[router_task, income_task],
        expected_output=(
            "A fraud-risk note with suspected issues, confidence level, and escalation advice."
        ),
    )

    credit_task = Task(
        description=with_format_rule(
            "Perform the Stage 3 credit review. Judge repayment ability, leverage, EMI burden, "
            f"and overall lending risk.\n\n{loan_case}"
        ),
        agent=credit_agent,
        context=[router_task, income_task, fraud_task],
        expected_output=(
            "A credit note with risk summary, repayment capacity, and lending recommendation."
        ),
    )

    compliance_task = Task(
        description=with_format_rule(
            "Perform the Stage 4 compliance review. Check policy fit, KYC readiness, and whether "
            f"the case is safe to move toward a final decision.\n\n{loan_case}"
        ),
        agent=compliance_agent,
        context=[router_task, income_task, fraud_task, credit_task],
        expected_output=(
            "A compliance note listing policy gaps, regulatory concerns, and readiness status."
        ),
    )

    weak_model_task = Task(
        description=with_format_rule(
            "Give a simpler lower-confidence second opinion on this loan file. Call out where you "
            f"disagree with the stronger specialists or where they may be overconfident.\n\n{loan_case}"
        ),
        agent=weak_model_agent,
        context=[router_task, income_task, fraud_task, credit_task, compliance_task],
        expected_output=(
            "A lightweight comparison opinion that highlights possible blind spots."
        ),
    )

    summariser_task = Task(
        description=with_format_rule(
            "Write the final decision memo. Merge the routing, income, fraud, credit, compliance, "
            "and weak-model views into one concise recommendation. End with APPROVE, REJECT, or "
            f"MANUAL REVIEW and list the main reasons.\n\n{loan_case}"
        ),
        agent=summariser_agent,
        context=[
            router_task,
            income_task,
            fraud_task,
            credit_task,
            compliance_task,
            weak_model_task,
        ],
        expected_output=(
            "A final decision memo with recommendation, rationale, and the most important risks."
        ),
    )

    return Crew(
        agents=[
            router_manager,
            income_agent,
            fraud_agent,
            credit_agent,
            compliance_agent,
            weak_model_agent,
            summariser_agent,
        ],
        tasks=[
            router_task,
            income_task,
            fraud_task,
            credit_task,
            compliance_task,
            weak_model_task,
            summariser_task,
        ],
        process=Process.sequential,
        verbose=True,
    )


def run(mode: str) -> str:
    if mode == "hello":
        crew = build_hello_world_crew(HELLO_WORLD_CASE)
    else:
        crew = build_full_pipeline_crew(FULL_PIPELINE_CASE)

    print(f"\nRunning mode: {mode}")
    print(f"Primary Ollama model: {PRIMARY_MODEL_NAME}")
    print(f"Weak-model Ollama model: {WEAK_MODEL_NAME}\n")

    return crew.kickoff()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the HDFC multi-agent CrewAI demo with local Ollama models."
    )
    parser.add_argument(
        "--mode",
        choices=("hello", "full"),
        default="hello",
        help="`hello` runs the 2-agent smoke test, `full` runs the 7-agent pipeline.",
    )
    args = parser.parse_args()

    result = run(args.mode)
    print("\nFINAL OUTPUT:\n")
    print(result)


if __name__ == "__main__":
    main()
