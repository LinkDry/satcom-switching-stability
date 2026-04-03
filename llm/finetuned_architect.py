"""Fine-tuned Qwen3-4B as MDP Architect.

Loads LoRA adapter, generates reward weights from KPIs.
Inference: ~1-3s on RTX 4060 (vs 20-40s API calls).
Uses RAG context to reason about novel/unseen traffic regimes.
"""

import json
import re
import time
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llm.rag_knowledge import retrieve_context

BASE_MODEL = "Qwen/Qwen3-4B"
ADAPTER_PATH = str(Path(__file__).parent.parent / "models" / "qwen3-4b-satcom-lora")
WEIGHT_KEYS = ["sum_rate", "fairness", "outage", "switching", "queue"]


class FinetunedLLMArchitect:
    """Fine-tuned Qwen3-4B for KPI → reward weights."""

    def __init__(self, device="cuda"):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.call_count = 0
        self.total_latency = 0.0

    def _load(self):
        if self.model is not None:
            return
        print("Loading fine-tuned Qwen3-4B...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        torch.cuda.empty_cache()
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb_config, device_map={"": 0}
        )
        self.model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
        print("  Model loaded.")

    def _describe_kpis(self, kpi: dict) -> str:
        """Translate numerical KPIs into natural language for RAG retrieval."""
        parts = []
        avg = kpi.get("avg_demand", 0)
        gini = kpi.get("spatial_gini", 0)
        peak = kpi.get("peak_beam_demand", 0)
        active = kpi.get("active_beam_fraction", 0)
        var = kpi.get("demand_variance", 0)

        # Demand level
        if avg > 50:
            parts.append("high demand dense urban traffic")
        elif avg < 15:
            parts.append("low demand sparse traffic")

        # Spatial patterns
        if gini > 0.5:
            parts.append("extreme spatial imbalance with hot and cold regions")
        elif gini < 0.15:
            parts.append("uniform spatial distribution maritime-like")

        # Peak / burst detection
        if peak > 120:
            parts.append("emergency disaster demand spike")
        if var > 1500 and avg < 30:
            parts.append("bursty IoT sensor data collection pattern")

        # Active beam patterns
        if active > 0.85 and avg < 25:
            parts.append("many active beams low individual demand IoT device collection")
        if active < 0.4:
            parts.append("few active beams energy saving power efficient")

        # Transition / handover hints
        if 0.3 < gini < 0.5 and 20 < avg < 40:
            parts.append("transition handover mixed regime")

        return ". ".join(parts) if parts else "moderate balanced traffic"

    def predict_weights(self, kpi: dict) -> dict:
        """Generate reward weights from KPI dict, with RAG domain context."""
        self._load()
        t0 = time.time()

        # Retrieve domain knowledge based on KPI patterns
        kpi_description = self._describe_kpis(kpi)
        rag_context = retrieve_context(kpi_description, top_k=2)

        instruction = (
            "You are the MDP Architect for a 19-beam LEO satellite beam scheduling system. "
            "Given network KPIs, output optimal reward weights as JSON. "
            "Keys: sum_rate, fairness, outage, switching, queue. "
            "All weights must be POSITIVE."
        )
        if rag_context:
            instruction += f"\n\n## Domain Knowledge\n{rag_context}"

        kpi_lines = "\n".join(f"- {k}: {v:.2f}" for k, v in kpi.items() if isinstance(v, (int, float)))
        user_input = f"Current Network KPIs:\n{kpi_lines}"

        # Use the model's chat template to match training format
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_input},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=100, temperature=0.1,
                do_sample=True, pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        latency = time.time() - t0
        self.call_count += 1
        self.total_latency += latency

        # Parse JSON from response
        weights = self._parse_weights(response)
        return weights

    def _parse_weights(self, response: str) -> dict:
        """Extract weight dict from model response, with robust fallback."""
        # Try strict JSON parsing first
        try:
            match = re.search(r'\{[^}]+\}', response)
            if match:
                parsed = json.loads(match.group())
                return {k: abs(float(parsed.get(k, 0.5))) for k in WEIGHT_KEYS}
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: extract individual key-value pairs with regex
        weights = {}
        for key in WEIGHT_KEYS:
            m = re.search(rf'"{key}"\s*:\s*([\d.]+)', response)
            if m:
                try:
                    weights[key] = abs(float(m.group(1)))
                except ValueError:
                    pass
        if len(weights) >= 3:  # Accept if at least 3 of 5 keys parsed
            for k in WEIGHT_KEYS:
                if k not in weights:
                    weights[k] = 0.5
            return weights

        # Final fallback: default weights
        print(f"  [FT-LLM] Parse failed, raw response: [{response[:200]}]")
        return {"sum_rate": 1.0, "fairness": 0.0, "outage": 1.0, "switching": 0.01, "queue": 0.0}

    def get_stats(self):
        return {
            "call_count": self.call_count,
            "total_latency_s": self.total_latency,
            "avg_latency_s": self.total_latency / max(1, self.call_count),
        }


if __name__ == "__main__":
    import os
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    arch = FinetunedLLMArchitect()

    # Test with different regime KPIs
    test_kpis = {
        "urban": {"avg_demand": 55.0, "demand_variance": 800.0, "spatial_gini": 0.45, "peak_beam_demand": 90.0, "active_beam_fraction": 0.8},
        "maritime": {"avg_demand": 12.0, "demand_variance": 50.0, "spatial_gini": 0.12, "peak_beam_demand": 25.0, "active_beam_fraction": 0.3},
        "disaster": {"avg_demand": 45.0, "demand_variance": 3000.0, "spatial_gini": 0.55, "peak_beam_demand": 180.0, "active_beam_fraction": 0.9},
        "mixed": {"avg_demand": 30.0, "demand_variance": 400.0, "spatial_gini": 0.25, "peak_beam_demand": 60.0, "active_beam_fraction": 0.5},
    }

    for regime, kpi in test_kpis.items():
        weights = arch.predict_weights(kpi)
        print(f"{regime:10s}: {weights}")
        print(f"  latency: {arch.total_latency / arch.call_count:.2f}s")
