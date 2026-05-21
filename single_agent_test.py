# single_agent_test.py
# Run all 7 agents on 1 borrower.
# Check: does each agent stay in its own domain?

from data_loader import get_profiles, load_dataset
from output_parser import parse_all
from profile_builder import build_borrower_profile

FORMAT_RULE = (
    "Return the answer using exactly this structure:\n"
    "Decision: Approve, Reject, or Refer\n"
    "Confidence: high, medium, or low\n"
    "Brief reason: one short sentence."
)


def clean_output(text: str) -> str:
    if "Final Answer:" in text:
        return text.split("Final Answer:", 1)[-1].strip()
    return text.strip()


def get_agents():
    print("Importing agents...", flush=True)
    from agents import (
        compliance_agent,
        credit_agent,
        fraud_agent,
        income_agent,
        router_manager,
        summariser_agent,
        weak_model_agent,
    )

    return {
        "RouterManager": router_manager,
        "IncomeAgent": income_agent,
        "FraudAgent": fraud_agent,
        "CreditAgent": credit_agent,
        "ComplianceAgent": compliance_agent,
        "SummariserAgent": summariser_agent,
        "WeakModelAgent": weak_model_agent,
    }


def run_single_agent(agent, profile_text):
    try:
        from crewai import Crew, Process, Task

        task_description = (
            "Analyse this loan application only from your assigned specialist role.\n\n"
            f"{profile_text}\n\n"
            f"{FORMAT_RULE}"
        )

        for _ in range(2):
            task = Task(
                description=task_description,
                agent=agent,
                expected_output=(
                    "Decision: Approve, Reject, or Refer. Confidence: high, medium, or low. "
                    "Brief reason: one short sentence."
                ),
            )
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
            )
            raw_output = clean_output(str(crew.kickoff()))
            if raw_output != "Agent stopped due to iteration limit or time limit.":
                return raw_output

        fallback_prompt = (
            f"You are acting as {agent.role}.\n"
            "Review the borrower profile only from your assigned specialist role.\n"
            f"{FORMAT_RULE}\n\n"
            f"{profile_text}"
        )
        return clean_output(agent.llm.invoke(fallback_prompt))
    except Exception as e:
        return f"ERROR: {str(e)}"


if __name__ == "__main__":
    print("Loading dataset...", flush=True)
    rows = load_dataset()
    profiles = get_profiles(rows, 50)
    agents = get_agents()

    row = profiles[0]
    profile_text, truth_label = build_borrower_profile(row)

    print(f"Profile: {profile_text}")
    print(f"Ground truth: {truth_label}")
    print("=" * 60)

    for agent_name, agent in agents.items():
        print(f"\n--- {agent_name} ---", flush=True)
        raw = run_single_agent(agent, profile_text)
        parsed = parse_all(raw)
        print(f"Raw output: {raw[:500]}")
        print(
            f"Decision: {parsed['parsed_decision']} | "
            f"Confidence: {parsed['confidence_flag']}"
        )
