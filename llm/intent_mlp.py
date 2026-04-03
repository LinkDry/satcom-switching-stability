"""Intent-Aware MLP: KPIs + ObjectiveProfile → reward weights.

Key innovation: Same KPIs produce DIFFERENT weights depending on operator intent.
Input: 5-dim KPI + 5-dim objective profile = 10-dim
Output: 5-dim reward weights

This is what makes the hybrid architecture work:
- LLM understands operator intent (NL → objective profile)
- MLP translates intent + network state into optimal weights (~1ms)
- DRL executes beam scheduling with those weights
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

KPI_KEYS = ["avg_demand", "demand_variance", "spatial_gini", "peak_beam_demand", "active_beam_fraction"]
OBJ_KEYS = ["throughput_priority", "fairness_priority", "outage_tolerance", "switching_tolerance", "queue_tolerance"]
WEIGHT_KEYS = ["sum_rate", "fairness", "outage", "switching", "queue"]

MODEL_DIR = Path(__file__).parent.parent / "models"


class IntentAwareMLP(nn.Module):
    """10-dim input (5 KPI + 5 objective) → 5 reward weights."""

    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, len(WEIGHT_KEYS)),
            nn.Softplus(),  # All weights positive
        )

    def forward(self, x):
        return self.net(x)

    def predict_weights(self, kpi: dict, objective_profile, kpi_mean=None, kpi_std=None) -> dict:
        """Predict weights from KPI dict + ObjectiveProfile."""
        kpi_vec = [float(kpi.get(k, 0)) for k in KPI_KEYS]
        if kpi_mean is not None and kpi_std is not None:
            kpi_vec = [(v - m) / max(s, 1e-6) for v, m, s in zip(kpi_vec, kpi_mean, kpi_std)]
        obj_vec = [
            objective_profile.throughput_priority,
            objective_profile.fairness_priority,
            objective_profile.outage_tolerance,
            objective_profile.switching_tolerance,
            objective_profile.queue_priority,
        ]
        x = torch.tensor([kpi_vec + obj_vec], dtype=torch.float32)
        with torch.no_grad():
            w = self(x)[0].numpy()
        return {k: float(v) for k, v in zip(WEIGHT_KEYS, w)}


def generate_intent_training_data(n_per_combo=1000, seed=42):
    """Generate training data: (KPI, objective) → optimal weights.

    Key insight: The SAME KPI pattern should produce DIFFERENT weights
    depending on the operator's objective profile.
    """
    rng = np.random.default_rng(seed)

    # KPI distributions per regime (same as before)
    kpi_profiles = {
        "urban": {"avg_demand": (40, 70), "demand_variance": (500, 1500), "spatial_gini": (0.3, 0.6), "peak_beam_demand": (60, 120), "active_beam_fraction": (0.5, 0.8)},
        "maritime": {"avg_demand": (5, 20), "demand_variance": (20, 100), "spatial_gini": (0.05, 0.2), "peak_beam_demand": (10, 40), "active_beam_fraction": (0.1, 0.4)},
        "disaster": {"avg_demand": (30, 60), "demand_variance": (2000, 5000), "spatial_gini": (0.4, 0.7), "peak_beam_demand": (120, 250), "active_beam_fraction": (0.7, 1.0)},
        "mixed": {"avg_demand": (20, 45), "demand_variance": (200, 800), "spatial_gini": (0.15, 0.4), "peak_beam_demand": (40, 90), "active_beam_fraction": (0.3, 0.6)},
        # Novel regimes too
        "iot_burst": {"avg_demand": (3, 15), "demand_variance": (100, 3000), "spatial_gini": (0.2, 0.5), "peak_beam_demand": (50, 200), "active_beam_fraction": (0.2, 0.5)},
        "polar": {"avg_demand": (10, 40), "demand_variance": (300, 1200), "spatial_gini": (0.3, 0.6), "peak_beam_demand": (30, 80), "active_beam_fraction": (0.3, 0.7)},
    }

    # Objective profiles with their optimal weight mappings
    # The key: same regime + different objective = different weights
    objective_scenarios = [
        # (objective_vec, weight_function)
        # Max throughput: high sum_rate, low everything else
        {"obj": [0.9, 0.1, 0.3, 0.5, 0.5], "base_w": [1.2, 0.05, 0.5, 0.01, 0.02]},
        # Emergency/disaster: high outage penalty, moderate throughput
        {"obj": [0.5, 0.2, 0.9, 0.3, 0.7], "base_w": [0.8, 0.1, 2.5, 0.01, 0.3]},
        # Fairness-first: high fairness, moderate throughput
        {"obj": [0.4, 0.9, 0.3, 0.5, 0.3], "base_w": [0.6, 0.5, 0.8, 0.01, 0.05]},
        # Energy saving: minimize switching, moderate throughput
        {"obj": [0.5, 0.3, 0.3, 0.9, 0.3], "base_w": [0.7, 0.1, 0.5, 0.3, 0.05]},
        # Queue management: high queue penalty
        {"obj": [0.5, 0.3, 0.3, 0.3, 0.9], "base_w": [0.7, 0.1, 0.5, 0.01, 0.5]},
        # Balanced
        {"obj": [0.5, 0.5, 0.5, 0.5, 0.5], "base_w": [1.0, 0.15, 1.0, 0.01, 0.1]},
    ]

    data_x, data_y = [], []

    for regime_name, kpi_ranges in kpi_profiles.items():
        for scenario in objective_scenarios:
            for _ in range(n_per_combo):
                # Sample KPI
                kpi = [rng.uniform(*kpi_ranges[k]) for k in KPI_KEYS]
                # Objective vector
                obj = [v + rng.normal(0, 0.05) for v in scenario["obj"]]
                obj = [max(0, min(1, v)) for v in obj]
                # Compute target weights (base + regime-specific adjustment + noise)
                base_w = np.array(scenario["base_w"])
                # Regime-specific modulation
                if regime_name == "disaster":
                    base_w[2] *= 1.5  # More outage penalty
                elif regime_name == "maritime":
                    base_w[1] *= 2.0  # More fairness
                elif regime_name == "iot_burst":
                    base_w[4] *= 2.0  # More queue penalty
                # Add noise
                weights = base_w + rng.normal(0, 0.03, len(base_w))
                weights = np.maximum(weights, 0.001)

                data_x.append(kpi + obj)
                data_y.append(weights.tolist())

    X = np.array(data_x, dtype=np.float32)
    Y = np.array(data_y, dtype=np.float32)
    print(f"  Generated {len(X)} samples ({len(kpi_profiles)} regimes × {len(objective_scenarios)} objectives × {n_per_combo})")
    return X, Y


def train_intent_mlp(save_path=None, epochs=300, lr=1e-3):
    """Train intent-aware MLP."""
    print("Generating intent-aware training data...")
    X, Y = generate_intent_training_data()

    # Normalize KPI features (first 5 dims)
    kpi_mean = X[:, :5].mean(axis=0)
    kpi_std = X[:, :5].std(axis=0) + 1e-8
    X[:, :5] = (X[:, :5] - kpi_mean) / kpi_std

    # Train/val split
    n = len(X)
    idx = np.random.permutation(n)
    split = int(0.9 * n)
    train_x, val_x = torch.tensor(X[idx[:split]]), torch.tensor(X[idx[split:]])
    train_y, val_y = torch.tensor(Y[idx[:split]]), torch.tensor(Y[idx[split:]])

    model = IntentAwareMLP(hidden=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(train_x))
        total_loss = 0
        for i in range(0, len(train_x), 256):
            batch_x = train_x[perm[i:i+256]]
            batch_y = train_y[perm[i:i+256]]
            pred = model(batch_x)
            loss = nn.functional.mse_loss(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_x)
                val_loss = nn.functional.mse_loss(val_pred, val_y).item()
            if val_loss < best_val:
                best_val = val_loss
            print(f"  Epoch {epoch}: train={total_loss/(len(train_x)//256+1):.6f} val={val_loss:.6f}")

    # Save
    save_path = save_path or str(MODEL_DIR / "intent_mlp.pt")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "kpi_mean": kpi_mean, "kpi_std": kpi_std,
    }, save_path)
    print(f"  Saved to {save_path} (best val={best_val:.6f})")
    return model, kpi_mean, kpi_std


def load_intent_mlp(path=None):
    """Load trained intent-aware MLP."""
    path = path or str(MODEL_DIR / "intent_mlp.pt")
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    model = IntentAwareMLP(hidden=128)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["kpi_mean"], ckpt["kpi_std"]


if __name__ == "__main__":
    train_intent_mlp()
