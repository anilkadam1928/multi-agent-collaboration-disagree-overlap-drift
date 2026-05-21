print("Starting test_10_borrowers.py...", flush=True)

from data_loader import get_profiles, load_dataset
from output_parser import parse_all
from profile_builder import build_borrower_profile
from results_logger import save_results

TEST_BORROWER_COUNT = 50
RETRY_ATTEMPTS = 2
RESULTS_PATH = f"test{TEST_BORROWER_COUNT}_results.csv"
RUN_PREFIX = f"test{TEST_BORROWER_COUNT}"

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


def compact_for_csv(text: str) -> str:
    return " | ".join(part.strip() for part in text.splitlines() if part.strip())


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

        for _ in range(RETRY_ATTEMPTS):
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
    agents = get_agents()

    profiles = get_profiles(rows, TEST_BORROWER_COUNT)
    print(f"Loaded {len(rows)} rows total.", flush=True)
    print(
        f"Running {len(profiles)} borrowers x {len(agents)} agents = "
        f"{len(profiles) * len(agents)} calls",
        flush=True,
    )

    all_results = []

    for idx, row in enumerate(profiles):
        profile_text, truth = build_borrower_profile(row)
        borrower_id = f"B{idx+1:03d}"
        print(f"\nBorrower {borrower_id} ({idx+1}/{len(profiles)})", flush=True)

        for agent_name, agent in agents.items():
            print(f"  Running {agent_name}...", flush=True)
            raw = run_single_agent(agent, profile_text)
            parsed = parse_all(raw)
            print(
                f"  Done {agent_name} -> {parsed['parsed_decision']} | "
                f"{parsed['confidence_flag']}",
                flush=True,
            )

            all_results.append(
                {
                    "run_id": f"{RUN_PREFIX}_{borrower_id}_{agent_name}",
                    "borrower_id": borrower_id,
                    "agent_name": agent_name,
                    "raw_output": compact_for_csv(raw)[:500],
                    "parsed_decision": parsed["parsed_decision"],
                    "confidence_flag": parsed["confidence_flag"],
                    "kaggle_truth_label": truth,
                }
            )

    save_results(RESULTS_PATH, all_results)
    print(f"\nDone. {len(all_results)} rows saved to {RESULTS_PATH}.", flush=True)
