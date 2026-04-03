"""LLM MDP Architect — calls LLM API to generate MDP specifications."""

import json
import time
from typing import Optional

from mdp.spec import MDPSpec, validate_spec
from llm.prompts import SYSTEM_PROMPT, MDP_GENERATION_PROMPT, REWARD_ONLY_PROMPT, STATE_ONLY_PROMPT


class LLMMDPArchitect:
    """LLM-based MDP specification generator.

    Calls an OpenAI-compatible API to generate MDP specs from network KPIs.
    Supports full MDP generation and component-only ablations.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.2,
        max_retries: int = 3,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = None
        self._api_key = api_key
        self._base_url = base_url
        self.call_count = 0
        self.total_latency_s = 0.0

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            kwargs = {"timeout": 120}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def generate_full_spec(self, kpi_summary: dict, context: Optional[dict] = None) -> Optional[MDPSpec]:
        """Generate a complete MDP specification from KPI summary.

        This is the main method — LLM redesigns the entire MDP.
        """
        prompt = MDP_GENERATION_PROMPT.format(kpi_summary=json.dumps(kpi_summary, indent=2))
        return self._call_and_parse(prompt)

    def generate_reward_only(
        self, kpi_summary: dict, current_features: list[str], current_action: str
    ) -> Optional[list[dict]]:
        """Generate only reward weights (ablation: reward-only LLM)."""
        prompt = REWARD_ONLY_PROMPT.format(
            kpi_summary=json.dumps(kpi_summary, indent=2),
            current_state_features=json.dumps(current_features),
            current_action_type=current_action,
        )
        response = self._call_llm(prompt)
        if response:
            try:
                return json.loads(self._extract_json(response))
            except (json.JSONDecodeError, ValueError):
                return None
        return None

    def generate_state_only(self, kpi_summary: dict, available_features: list[str]) -> Optional[list[str]]:
        """Generate only state feature selection (ablation: state-only LLM)."""
        prompt = STATE_ONLY_PROMPT.format(
            kpi_summary=json.dumps(kpi_summary, indent=2),
            available_features=json.dumps(available_features),
        )
        response = self._call_llm(prompt)
        if response:
            try:
                return json.loads(self._extract_json(response))
            except (json.JSONDecodeError, ValueError):
                return None
        return None

    def _call_and_parse(self, prompt: str) -> Optional[MDPSpec]:
        """Call LLM and parse response into MDPSpec with retry logic."""
        for attempt in range(self.max_retries):
            response = self._call_llm(prompt)
            if response is None:
                continue

            try:
                json_str = self._extract_json(response)
                spec_dict = json.loads(json_str)
                spec = MDPSpec.from_dict(spec_dict)

                # Validate
                valid, err = validate_spec(spec)
                if valid:
                    return spec
                else:
                    if attempt < self.max_retries - 1:
                        prompt += f"\n\nPrevious attempt failed validation: {err}. Fix and try again."
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                if attempt < self.max_retries - 1:
                    prompt += f"\n\nPrevious response was not valid JSON: {e}. Output ONLY valid JSON."

        return None

    def _call_llm(self, user_prompt: str) -> Optional[str]:
        """Make a single LLM API call."""
        client = self._get_client()
        t0 = time.time()

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=2000,
            )
            latency = time.time() - t0
            self.call_count += 1
            self.total_latency_s += latency
            return response.choices[0].message.content
        except Exception as e:
            print(f"  LLM API error: {e}")
            return None

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from LLM response (may be wrapped in markdown code blocks)."""
        # Try to find JSON in code blocks
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            return text[start:end].strip()
        if "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            return text[start:end].strip()
        # Try raw JSON
        for i, c in enumerate(text):
            if c in "{[":
                depth = 0
                for j in range(i, len(text)):
                    if text[j] in "{[":
                        depth += 1
                    elif text[j] in "}]":
                        depth -= 1
                    if depth == 0:
                        return text[i : j + 1]
        return text.strip()

    def get_stats(self) -> dict:
        return {
            "call_count": self.call_count,
            "total_latency_s": self.total_latency_s,
            "avg_latency_s": self.total_latency_s / max(self.call_count, 1),
        }
