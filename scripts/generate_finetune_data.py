"""Generate fine-tuning dataset for Qwen3.5-4B LoRA training.

Format: LlamaFactory alpaca format (instruction, input, output).
Task: Given network KPIs, generate optimal reward weights as JSON.
"""

import json
import numpy as np
from pathlib import Path

# Regime KPI distributions (from simulator observations)
REGIME_PROFILES = {
    "urban": {
        "avg_demand": (50, 15), "demand_variance": (800, 200),
        "spatial_gini": (0.45, 0.1), "peak_beam_demand": (90, 20),
        "active_beam_fraction": (0.7, 0.15),
    },
    "maritime": {
        "avg_demand": (12, 5), "demand_variance": (200, 80),
        "spatial_gini": (0.15, 0.05), "peak_beam_demand": (30, 10),
        "active_beam_fraction": (0.3, 0.1),
    },
    "disaster": {
        "avg_demand": (35, 15), "demand_variance": (2500, 800),
        "spatial_gini": (0.25, 0.08), "peak_beam_demand": (180, 40),
        "active_beam_fraction": (0.5, 0.15),
    },
    "mixed": {
        "avg_demand": (30, 10), "demand_variance": (600, 200),
        "spatial_gini": (0.30, 0.08), "peak_beam_demand": (70, 20),
        "active_beam_fraction": (0.5, 0.15),
    },
}

# Expert weights (from rule-based, 342.8 Mbps)
EXPERT_WEIGHTS = {
    "urban":    {"sum_rate": 1.0, "fairness": 0.0, "outage": 1.0, "switching": 0.01, "queue": 0.0},
    "maritime": {"sum_rate": 1.0, "fairness": 0.3, "outage": 1.0, "switching": 0.01, "queue": 0.0},
    "disaster": {"sum_rate": 1.0, "fairness": 0.0, "outage": 2.0, "switching": 0.01, "queue": 0.1},
    "mixed":    {"sum_rate": 1.0, "fairness": 0.0, "outage": 1.0, "switching": 0.01, "queue": 0.0},
}

INSTRUCTION = """You are the MDP Architect for a 19-beam LEO satellite beam scheduling system.
Given the current network KPIs, output the optimal reward weights as JSON.
All weights must be POSITIVE. The environment formula is: reward = sum_rate*R - outage*O - switching*S - queue*Q + fairness*F.
Output ONLY valid JSON, no explanation."""


def generate_dataset(n_per_regime=500, n_transition=200, seed=42):
    rng = np.random.default_rng(seed)
    samples = []

    for regime_name, profile in REGIME_PROFILES.items():
        weights = EXPERT_WEIGHTS[regime_name]
        for _ in range(n_per_regime):
            kpi = {}
            for k, (mean, std) in profile.items():
                kpi[k] = round(float(max(0, rng.normal(mean, std))), 2)

            # Add small noise to weights for diversity
            noisy_w = {}
            for wk, wv in weights.items():
                noise = rng.normal(0, 0.02)
                noisy_w[wk] = round(max(0, wv + noise), 4)

            kpi_text = "\n".join(f"- {k}: {v}" for k, v in kpi.items())
            samples.append({
                "instruction": INSTRUCTION,
                "input": f"Current Network KPIs:\n{kpi_text}",
                "output": json.dumps(noisy_w),
            })

    # Transition samples (interpolated KPIs between regimes)
    regime_names = list(REGIME_PROFILES.keys())
    for _ in range(n_transition):
        r1, r2 = rng.choice(regime_names, 2, replace=False)
        alpha = rng.uniform(0.2, 0.8)
        kpi = {}
        for k in REGIME_PROFILES[r1]:
            m1, s1 = REGIME_PROFILES[r1][k]
            m2, s2 = REGIME_PROFILES[r2][k]
            mean = (1 - alpha) * m1 + alpha * m2
            std = (1 - alpha) * s1 + alpha * s2
            kpi[k] = round(float(max(0, rng.normal(mean, std))), 2)

        # Interpolated weights
        w1, w2 = EXPERT_WEIGHTS[r1], EXPERT_WEIGHTS[r2]
        weights = {k: round((1 - alpha) * w1[k] + alpha * w2[k], 4) for k in w1}

        kpi_text = "\n".join(f"- {k}: {v}" for k, v in kpi.items())
        samples.append({
            "instruction": INSTRUCTION,
            "input": f"Current Network KPIs:\n{kpi_text}",
            "output": json.dumps(weights),
        })

    rng.shuffle(samples)

    # Split train/val
    split = int(len(samples) * 0.9)
    train_data = samples[:split]
    val_data = samples[split:]

    return train_data, val_data


def main():
    out_dir = Path(__file__).parent.parent / "data" / "finetune"
    out_dir.mkdir(parents=True, exist_ok=True)

    train, val = generate_dataset()
    print(f"Generated {len(train)} train, {len(val)} val samples")

    with open(out_dir / "train.json", "w") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)
    with open(out_dir / "val.json", "w") as f:
        json.dump(val, f, indent=2, ensure_ascii=False)

    print(f"Saved to {out_dir}")
    print(f"Sample:\n{json.dumps(train[0], indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
