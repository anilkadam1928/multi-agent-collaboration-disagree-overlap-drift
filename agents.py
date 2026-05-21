from __future__ import annotations

import os
import subprocess
from typing import Any

import requests
from crewai import Agent
import crewai.crew as crew_module
from crewai.tools import tool_usage as tool_usage_module
from langchain_core.language_models.llms import LLM


class LocalNoOpTelemetry:
    """Keep the demo fully local by disabling CrewAI's remote telemetry hooks."""

    def set_tracer(self) -> None:
        pass

    def crew_creation(self, crew) -> None:
        pass

    def crew_execution_span(self, crew):
        return None

    def end_crew(self, crew, output) -> None:
        pass

    def tool_repeated_usage(self, llm, tool_name: str, attempts: int) -> None:
        pass

    def tool_usage(self, llm, tool_name: str, attempts: int) -> None:
        pass

    def tool_usage_error(self, llm) -> None:
        pass


crew_module.Telemetry = LocalNoOpTelemetry
tool_usage_module.Telemetry = LocalNoOpTelemetry


class LocalOllamaLLM(LLM):
    """Minimal non-streaming Ollama adapter that works cleanly with CrewAI 0.28.x."""

    base_url: str = "http://127.0.0.1:1234"
    model: str
    fallback_model: str | None = None
    temperature: float = 0.0
    num_predict: int = 256
    timeout: int = 300

    @property
    def _llm_type(self) -> str:
        return "local-ollama-generate"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "fallback_model": self.fallback_model,
            "temperature": self.temperature,
            "num_predict": self.num_predict,
        }

    def _generate_once(
        self,
        model_name: str,
        prompt: str,
        stop: list[str] | None,
        **kwargs: Any,
    ) -> str:
        options = {
            "temperature": kwargs.get("temperature", self.temperature),
            "num_predict": kwargs.get("num_predict", self.num_predict),
        }
        if stop:
            options["stop"] = stop

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": options.get("temperature", self.temperature),
                "max_tokens": options.get("num_predict", self.num_predict),
                "stream": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]

        content = (message.get("content") or "").strip()
        if content:
            return content

        reasoning_content = (message.get("reasoning_content") or "").strip()
        if not reasoning_content:
            return ""

        recovery_prompt = (
            "You previously produced draft reasoning instead of the user-visible answer.\n"
            "Using the original prompt and the draft reasoning below, return only the final "
            "answer for the user.\n"
            "Rules:\n"
            "- Start with `Final Answer:`\n"
            "- After `Final Answer:`, provide the actual answer requested by the original prompt\n"
            "- If the original prompt requested a specific structure, preserve it after "
            "`Final Answer:`\n"
            "- Do not include hidden reasoning, analysis, or any extra preamble\n\n"
            f"Original prompt:\n{prompt}\n\n"
            f"Draft reasoning:\n{reasoning_content}"
        )

        recovery_response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": recovery_prompt}],
                "temperature": options.get("temperature", self.temperature),
                "max_tokens": min(max(options.get("num_predict", self.num_predict), 256), 512),
                "stream": False,
            },
            timeout=self.timeout,
        )
        recovery_response.raise_for_status()
        recovery_message = recovery_response.json()["choices"][0]["message"]

        recovered_content = (recovery_message.get("content") or "").strip()
        if recovered_content:
            return recovered_content

        return f"Final Answer: {reasoning_content}"

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> str:
        primary_response = self._generate_once(
            self.model,
            prompt,
            stop,
            **kwargs,
        )
        if primary_response or not self.fallback_model or self.fallback_model == self.model:
            return primary_response

        return self._generate_once(
            self.fallback_model,
            prompt,
            stop,
            **kwargs,
        )


def _installed_ollama_models() -> set[str]:
    """Read locally available Ollama models without depending on any cloud API."""
    try:
        completed = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return set()

    installed_models: set[str] = set()
    for line in completed.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            installed_models.add(parts[0])
    return installed_models


def _pick_model(*candidates: str | None) -> str:
    installed_models = _installed_ollama_models()
    cleaned_candidates = [candidate for candidate in candidates if candidate]

    if not cleaned_candidates:
        raise ValueError("At least one Ollama model candidate is required.")

    if not installed_models:
        return cleaned_candidates[0]

    for candidate in cleaned_candidates:
        if candidate in installed_models:
            return candidate

    return cleaned_candidates[0]


PRIMARY_MODEL_NAME = _pick_model(
    os.getenv("OLLAMA_MODEL"),
    "google/gemma-4-e4b",
    "google/gemma-4-e4b",
)

WEAK_MODEL_NAME = _pick_model(
    os.getenv("OLLAMA_WEAK_MODEL"),
    "google/gemma-4-e4b",
    PRIMARY_MODEL_NAME,
)

PRIMARY_FALLBACK_MODEL = None
if PRIMARY_MODEL_NAME != "google/gemma-4-e4b":
    fallback_candidate = _pick_model("google/gemma-4-e4b", PRIMARY_MODEL_NAME)
    if fallback_candidate != PRIMARY_MODEL_NAME:
        PRIMARY_FALLBACK_MODEL = fallback_candidate


def build_local_llm(
    model_name: str,
    temperature: float = 0.0,
    fallback_model: str | None = PRIMARY_FALLBACK_MODEL,
) -> LocalOllamaLLM:
    return LocalOllamaLLM(
        model=model_name,
        fallback_model=fallback_model,
        temperature=temperature,
        num_predict=512,
    )


def build_agent(
    *,
    role: str,
    goal: str,
    backstory: str,
    model_name: str = PRIMARY_MODEL_NAME,
    temperature: float = 0.0,
) -> Agent:
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=build_local_llm(
            model_name=model_name,
            temperature=temperature,
            fallback_model=None if model_name == WEAK_MODEL_NAME else PRIMARY_FALLBACK_MODEL,
        ),
        allow_delegation=False,
        max_iter=2,
        verbose=False,
    )


router_manager = build_agent(
    role="Router-Manager",
    goal=(
        "Route each loan application through the right review sequence, keep drift low, "
        "and coordinate the final decision path."
    ),
    backstory=(
        "You are the orchestration lead in an HDFC-style multi-agent credit pipeline. "
        "You track which specialist should review which risk first, keep agent drift under "
        "control, and route the case to the least-confused, most relevant specialist."
    ),
    temperature=0.0,
)

income_agent = build_agent(
    role="Income Agent",
    goal="Verify income documents, judge stability of earnings, and flag anomalies early.",
    backstory=(
        "You focus only on income evidence such as salary slips, bank statements, employer "
        "proof, and EMI load. You are careful about mismatches, missing proof, and signs of "
        "document manipulation."
    ),
)

fraud_agent = build_agent(
    role="Fraud Agent",
    goal="Detect suspicious patterns, forged evidence, identity mismatch, or tampered records.",
    backstory=(
        "You own Stage 2 fraud review. You look for abnormal cash flows, document tampering, "
        "identity mismatch, suspicious application behavior, and patterns that deserve manual "
        "investigation."
    ),
)

credit_agent = build_agent(
    role="Credit Agent",
    goal="Assess repayment ability, credit behavior, and overall lending risk.",
    backstory=(
        "You own Stage 3 credit review. You evaluate credit score, liabilities, repayment "
        "history, debt burden, and whether the applicant can realistically service the loan."
    ),
)

compliance_agent = build_agent(
    role="Compliance Agent",
    goal="Check internal policy fit, KYC completeness, and regulatory readiness before approval.",
    backstory=(
        "You own Stage 4 compliance review. You check policy rules, KYC completeness, RBI-style "
        "regulatory expectations, and whether the application is safe to move toward a decision."
    ),
)

summariser_agent = build_agent(
    role="Summariser Agent",
    goal="Compress multi-agent findings into a crisp decision memo without repeating noise.",
    backstory=(
        "You reduce agent memory into a concise decision-ready summary. You keep the strongest "
        "signals, discard repeated chatter, and present only the facts needed for a final call."
    ),
    temperature=0.0,
)

weak_model_agent = build_agent(
    role="Weak Model Agent",
    goal="Provide a lower-capability comparison view that helps expose sycophancy and blind spots.",
    backstory=(
        "You are an intentionally weaker analyst used for diagnostic comparison. Your job is not "
        "to be perfect; it is to give a simpler second opinion that helps reveal echo-chamber "
        "effects and overconfident agreement."
    ),
    model_name=WEAK_MODEL_NAME,
    temperature=0.0,
)
