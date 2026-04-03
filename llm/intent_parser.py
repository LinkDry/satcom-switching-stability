"""LLM Intent Parser: Natural language → ObjectiveProfile.

Uses fine-tuned Qwen3-4B to parse operator commands into formal objectives.
This is the LLM's irreplaceable role — NL understanding that MLP cannot do.
"""

import json
import re
import time
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llm.operator_intent import ObjectiveProfile

BASE_MODEL = "Qwen/Qwen3-4B"
ADAPTER_PATH = str(Path(__file__).parent.parent / "models" / "qwen3-4b-satcom-lora")


class LLMIntentParser:
    """Parses operator NL commands into ObjectiveProfile using fine-tuned LLM."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.call_count = 0
        self.total_latency = 0.0

    def _load(self):
        if self.model is not None:
            return
        print("Loading LLM intent parser...")
        torch.cuda.empty_cache()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb_config, device_map={"": 0}
        )
        self.model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

    def parse_intent(self, operator_command: str) -> ObjectiveProfile:
        """Alias for backwards compatibility."""
        return self.parse(operator_command)

    def parse(self, operator_command: str) -> ObjectiveProfile:
        """Parse NL operator command into ObjectiveProfile with RAG context."""
        self._load()
        t0 = time.time()

        # Retrieve domain knowledge via RAG
        from llm.rag_knowledge import retrieve_context
        rag_context = retrieve_context(operator_command, top_k=3)

        instruction = (
            "You are a LEO satellite network operations AI. Parse the operator's command "
            "into a formal objective profile.\n\n"
            "## Domain Knowledge\n"
            f"{rag_context}\n\n"
            "## Output Format (JSON, all values 0.0-1.0)\n"
            "- throughput_priority: how much to prioritize total throughput\n"
            "- fairness_priority: how much to ensure equal service across beams\n"
            "- outage_tolerance: tolerance for outages (0=zero tolerance, 1=acceptable)\n"
            "- switching_tolerance: tolerance for beam switching (0=minimize, 1=acceptable)\n"
            "- queue_priority: how much to prioritize reducing queue delays\n"
            "- description: short label for the scenario\n"
            "Output ONLY valid JSON."
        )

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"Operator command: {operator_command}"},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=150, temperature=0.3,
                do_sample=True, pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        self.total_latency += time.time() - t0
        self.call_count += 1

        return self._parse_profile(response, operator_command)

    def _parse_profile(self, response: str, original_cmd: str) -> ObjectiveProfile:
        """Extract ObjectiveProfile from LLM response, with fallback."""
        try:
            match = re.search(r'\{[^}]+\}', response)
            if match:
                data = json.loads(match.group())
                return ObjectiveProfile(
                    throughput_priority=float(data.get("throughput_priority", 0.5)),
                    fairness_priority=float(data.get("fairness_priority", 0.3)),
                    outage_tolerance=float(data.get("outage_tolerance", 0.3)),
                    switching_tolerance=float(data.get("switching_tolerance", 0.5)),
                    queue_priority=float(data.get("queue_priority", 0.5)),
                    description=str(data.get("description", "llm_parsed")),
                )
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        # Keyword-based fallback if LLM parsing fails
        cmd = original_cmd.lower()
        if any(w in cmd for w in ["emergency", "disaster", "crisis"]):
            return ObjectiveProfile(0.3, 0.2, 0.0, 0.8, 1.0, "emergency")
        elif any(w in cmd for w in ["fairness", "equal", "fair"]):
            return ObjectiveProfile(0.5, 0.9, 0.2, 0.3, 0.7, "fairness")
        elif any(w in cmd for w in ["throughput", "maximize", "capacity"]):
            return ObjectiveProfile(1.0, 0.1, 0.5, 0.5, 0.3, "max_throughput")
        else:
            return ObjectiveProfile(0.5, 0.3, 0.3, 0.5, 0.5, "default")


class RuleIntentParser:
    """Keyword-based intent parser (no LLM). Baseline for comparison."""

    def parse(self, command: str) -> ObjectiveProfile:
        cmd = command.lower()
        if any(w in cmd for w in ["emergency", "disaster", "crisis", "rescue"]):
            return ObjectiveProfile(0.3, 0.2, 0.0, 0.8, 1.0, "emergency")
        elif any(w in cmd for w in ["fairness", "equal", "fair", "balanced"]):
            return ObjectiveProfile(0.5, 0.9, 0.2, 0.3, 0.7, "fairness")
        elif any(w in cmd for w in ["throughput", "maximize", "capacity", "speed"]):
            return ObjectiveProfile(1.0, 0.1, 0.5, 0.5, 0.3, "max_throughput")
        elif any(w in cmd for w in ["power", "energy", "save", "green"]):
            return ObjectiveProfile(0.4, 0.3, 0.8, 0.3, 0.5, "power_saving")
        elif any(w in cmd for w in ["iot", "sensor", "collect", "m2m"]):
            return ObjectiveProfile(0.2, 0.8, 0.3, 0.2, 0.9, "iot_collection")
        else:
            return ObjectiveProfile(0.5, 0.3, 0.3, 0.5, 0.5, "default")
