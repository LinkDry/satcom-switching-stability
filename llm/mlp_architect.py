"""MLP-based MDP Architect: learns KPI → reward weights mapping.

Phase 1: Train on synthetic KPI data with expert weight labels.
Inference: ~1ms on CPU, no API dependency.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# --- Feature and weight keys ---
KPI_KEYS = ["avg_demand", "demand_variance", "spatial_gini", "peak_beam_demand", "active_beam_fraction"]
WEIGHT_KEYS = ["sum_rate", "fairness", "outage", "switching", "queue"]
SCALES = np.array([100.0, 5000.0, 1.0, 200.0, 1.0], dtype=np.float32)

# Expert weights (from rule-based approach, 342.8 Mbps / 0 outage)
EXPERT_WEIGHTS = {
    "urban":    [1.0, 0.0, 1.0, 0.01, 0.0],
    "maritime": [1.0, 0.3, 1.0, 0.01, 0.0],
    "disaster": [1.0, 0.0, 2.0, 0.01, 0.1],
    "mixed":    [1.0, 0.0, 1.0, 0.01, 0.0],
}

# KPI ranges per regime (mean, std) for synthetic data generation
KPI_PROFILES = {
    "urban":    {"avg_demand": (55, 12), "demand_variance": (2000, 500), "spatial_gini": (0.45, 0.08),
                 "peak_beam_demand": (110, 25), "active_beam_fraction": (0.85, 0.1)},
    "maritime": {"avg_demand": (12, 5),  "demand_variance": (200, 100),  "spatial_gini": (0.15, 0.05),
                 "peak_beam_demand": (25, 10),  "active_beam_fraction": (0.4, 0.15)},
    "disaster": {"avg_demand": (35, 15), "demand_variance": (4000, 1500), "spatial_gini": (0.3, 0.1),
                 "peak_beam_demand": (180, 50), "active_beam_fraction": (0.7, 0.15)},
    "mixed":    {"avg_demand": (30, 10), "demand_variance": (1500, 600), "spatial_gini": (0.3, 0.1),
                 "peak_beam_demand": (70, 30),  "active_beam_fraction": (0.6, 0.15)},
}


class MLPArchitect(nn.Module):
    """Small MLP: 5 KPI inputs → 5 reward weights."""

    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(KPI_KEYS), hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, len(WEIGHT_KEYS)),
            nn.Softplus(),  # Ensures all weights are positive
        )

    def forward(self, x):
        return self.net(x)

    def predict_weights(self, kpi_dict: dict) -> dict:
        """Inference: KPI dict → weight dict. ~1ms on CPU."""
        x = torch.tensor([kpi_dict.get(k, 0.0) for k in KPI_KEYS], dtype=torch.float32)
        x = x / torch.tensor(SCALES)
        with torch.no_grad():
            w = self.net(x.unsqueeze(0)).squeeze(0).numpy()
        return {k: float(v) for k, v in zip(WEIGHT_KEYS, w)}


def generate_training_data(n_samples=20000, seed=42):
    """Generate synthetic (KPI, expert_weights) pairs."""
    rng = np.random.default_rng(seed)
    data_x, data_y = [], []

    regimes = list(EXPERT_WEIGHTS.keys())
    n_per_regime = n_samples // len(regimes)

    for regime in regimes:
        profile = KPI_PROFILES[regime]
        expert_w = EXPERT_WEIGHTS[regime]

        for _ in range(n_per_regime):
            kpi = []
            for key in KPI_KEYS:
                mean, std = profile[key]
                val = rng.normal(mean, std)
                val = max(0.0, val)
                kpi.append(val)
            data_x.append(kpi)

            # Expert weights with small noise for smoother learning
            noisy_w = [max(0.001, w + rng.normal(0, 0.02)) for w in expert_w]
            data_y.append(noisy_w)

    # Add transition samples (interpolated between regimes)
    for _ in range(n_samples // 10):
        r1, r2 = rng.choice(regimes, 2, replace=False)
        alpha = rng.uniform(0.2, 0.8)
        p1, p2 = KPI_PROFILES[r1], KPI_PROFILES[r2]
        w1, w2 = EXPERT_WEIGHTS[r1], EXPERT_WEIGHTS[r2]

        kpi = []
        for key in KPI_KEYS:
            m1, s1 = p1[key]
            m2, s2 = p2[key]
            val = rng.normal(alpha * m1 + (1-alpha) * m2, (s1 + s2) / 3)
            kpi.append(max(0.0, val))
        data_x.append(kpi)
        data_y.append([max(0.001, alpha * a + (1-alpha) * b) for a, b in zip(w1, w2)])

    print(f"  Generated {len(data_x)} samples ({n_per_regime}/regime + {n_samples//10} transitions)")
    return np.array(data_x, dtype=np.float32), np.array(data_y, dtype=np.float32)


def train_mlp(save_dir="models/mlp_architect", epochs=300, lr=1e-3):
    """Train MLP on generated data."""
    print("Generating training data...")
    X, Y = generate_training_data(n_samples=20000)
    print(f"  Data: {X.shape[0]} samples, {X.shape[1]} features → {Y.shape[1]} weights")

    X_norm = X / SCALES
    n = len(X_norm)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    X_train = torch.tensor(X_norm[idx[:split]])
    Y_train = torch.tensor(Y[idx[:split]])
    X_val = torch.tensor(X_norm[idx[split:]])
    Y_val = torch.tensor(Y[idx[split:]])

    model = MLPArchitect(hidden=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        pred = model(X_train)
        loss = loss_fn(pred, Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(X_val), Y_val).item()
            print(f"  Epoch {epoch+1}: train={loss.item():.6f} val={val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), Path(save_dir) / "mlp_architect.pt")

    print(f"\n  Best val loss: {best_val_loss:.6f}")

    # Verify predictions
    model.eval()
    print("  Verification:")
    for regime in EXPERT_WEIGHTS:
        profile = KPI_PROFILES[regime]
        kpi = {k: profile[k][0] for k in KPI_KEYS}  # Use mean KPI values
        pred_w = model.predict_weights(kpi)
        expert = dict(zip(WEIGHT_KEYS, EXPERT_WEIGHTS[regime]))
        print(f"    {regime:10s}: pred={pred_w}")
        print(f"    {'':10s}  expert={expert}")

    return model


def load_mlp(model_dir="models/mlp_architect"):
    """Load a trained MLP model."""
    model = MLPArchitect(hidden=64)
    model.load_state_dict(torch.load(Path(model_dir) / "mlp_architect.pt", weights_only=True))
    model.eval()
    return model


def continual_train_mlp(
    oracle_samples: list[dict],
    base_model_dir: str = "models/mlp_architect",
    save_dir: str = "models/oracle_mlp",
    epochs: int = 200,
    lr: float = 5e-4,
    replay_ratio: float = 0.3,
    replay_samples: int = 5000,
    verbose: int = 1,
):
    """Continual training: load existing MLP, fine-tune on oracle data + replay buffer.

    Args:
        oracle_samples: list of {kpi: dict, weights: dict, quality_score: float} from oracle
        base_model_dir: directory of pre-trained MLP to load
        save_dir: directory to save the continually-trained model
        epochs: training epochs
        lr: learning rate (lower than initial training to avoid catastrophic forgetting)
        replay_ratio: fraction of each batch that comes from original training data
        replay_samples: number of original-distribution samples to generate for replay
        verbose: print progress

    Returns:
        Trained MLPArchitect model
    """
    # Load base model
    model = MLPArchitect(hidden=64)
    base_path = Path(base_model_dir) / "mlp_architect.pt"
    if base_path.exists():
        model.load_state_dict(torch.load(base_path, weights_only=True))
        if verbose:
            print(f"  Loaded base model from {base_path}")
    else:
        if verbose:
            print("  No base model found, training from scratch")

    # Prepare oracle data
    oracle_x, oracle_y, oracle_w = [], [], []
    for s in oracle_samples:
        kpi = s["kpi"]
        weights = s["weights"]
        x = [kpi.get(k, 0.0) for k in KPI_KEYS]
        y = [weights.get(k, 0.0) for k in WEIGHT_KEYS]
        oracle_x.append(x)
        oracle_y.append(y)
        oracle_w.append(s.get("quality_score", 1.0))

    oracle_x = np.array(oracle_x, dtype=np.float32)
    oracle_y = np.array(oracle_y, dtype=np.float32)
    oracle_w = np.array(oracle_w, dtype=np.float32)

    if verbose:
        print(f"  Oracle data: {len(oracle_x)} samples")

    # Generate replay buffer from original distribution (prevents catastrophic forgetting)
    replay_x, replay_y = generate_training_data(n_samples=replay_samples, seed=42)
    if verbose:
        print(f"  Replay buffer: {len(replay_x)} samples")

    # Combine: oracle + replay
    all_x = np.concatenate([oracle_x, replay_x])
    all_y = np.concatenate([oracle_y, replay_y])
    # Weight: oracle samples weighted by quality_score, replay samples weighted uniformly
    all_w = np.concatenate([oracle_w, np.ones(len(replay_x), dtype=np.float32)])

    # Normalize inputs
    all_x_norm = all_x / SCALES

    # Train/val split
    n = len(all_x_norm)
    idx = np.random.permutation(n)
    split = int(0.85 * n)
    X_train = torch.tensor(all_x_norm[idx[:split]])
    Y_train = torch.tensor(all_y[idx[:split]])
    W_train = torch.tensor(all_w[idx[:split]])
    X_val = torch.tensor(all_x_norm[idx[split:]])
    Y_val = torch.tensor(all_y[idx[split:]])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction='none')

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        pred = model(X_train)
        per_sample_loss = loss_fn(pred, Y_train).mean(dim=1)  # (N,)
        weighted_loss = (per_sample_loss * W_train).mean()
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = nn.MSELoss()(val_pred, Y_val).item()
            if verbose:
                print(f"  Epoch {epoch+1}: train={weighted_loss.item():.6f} val={val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), Path(save_dir) / "mlp_architect.pt")

    if verbose:
        print(f"  Best val loss: {best_val_loss:.6f}")

    # Load best checkpoint
    best_path = Path(save_dir) / "mlp_architect.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, weights_only=True))
    model.eval()

    # Verify on novel regimes
    if verbose:
        from llm.oracle_data_generator import NOVEL_KPI_PROFILES, NOVEL_EXPERT_HINTS
        print("  Verification (novel regimes):")
        for regime, profile in NOVEL_KPI_PROFILES.items():
            kpi = {k: profile[k][0] for k in KPI_KEYS}
            pred_w = model.predict_weights(kpi)
            hint = NOVEL_EXPERT_HINTS.get(regime, {})
            print(f"    {regime:18s}: pred={pred_w}")
            print(f"    {'':18s}  hint={hint}")

    return model


def incremental_train_mlp(
    evolved_samples: list[dict],
    previous_best_samples: list[dict] = None,
    base_model_dir: str = "models/oracle_mlp",
    save_dir: str = "models/evolved_mlp",
    epochs: int = 150,
    lr: float = 3e-4,
    evolved_weight: float = 2.0,
    previous_best_weight: float = 1.5,
    replay_weight: float = 0.5,
    replay_samples: int = 3000,
    verbose: int = 1,
) -> 'MLPArchitect':
    """Incremental MLP training for evolution rounds.

    Three-tier weighting:
    - evolved_samples: highest weight (new LLM/synthetic-evolved data)
    - previous_best_samples: medium weight (proven good from prior rounds)
    - replay buffer: lowest weight (original distribution, prevents forgetting)
    """
    model = MLPArchitect(hidden=64)
    base_path = Path(base_model_dir) / "mlp_architect.pt"
    if base_path.exists():
        model.load_state_dict(torch.load(base_path, weights_only=True))
        if verbose:
            print(f"  Loaded base model from {base_path}")

    # Prepare evolved data
    ev_x, ev_y, ev_w = [], [], []
    for s in evolved_samples:
        x = [s["kpi"].get(k, 0.0) for k in KPI_KEYS]
        y = [s["weights"].get(k, 0.0) for k in WEIGHT_KEYS]
        ev_x.append(x)
        ev_y.append(y)
        ev_w.append(s.get("quality_score", 1.0) * evolved_weight)

    # Prepare previous best data
    pb_x, pb_y, pb_w = [], [], []
    if previous_best_samples:
        for s in previous_best_samples:
            x = [s["kpi"].get(k, 0.0) for k in KPI_KEYS]
            y = [s["weights"].get(k, 0.0) for k in WEIGHT_KEYS]
            pb_x.append(x)
            pb_y.append(y)
            pb_w.append(s.get("quality_score", 1.0) * previous_best_weight)

    # Generate replay buffer
    replay_x, replay_y = generate_training_data(n_samples=replay_samples, seed=42)

    # Combine all three tiers
    all_x = np.concatenate([
        np.array(ev_x, dtype=np.float32),
        np.array(pb_x, dtype=np.float32) if pb_x else np.empty((0, len(KPI_KEYS)), dtype=np.float32),
        replay_x,
    ])
    all_y = np.concatenate([
        np.array(ev_y, dtype=np.float32),
        np.array(pb_y, dtype=np.float32) if pb_y else np.empty((0, len(WEIGHT_KEYS)), dtype=np.float32),
        replay_y,
    ])
    all_w = np.concatenate([
        np.array(ev_w, dtype=np.float32),
        np.array(pb_w, dtype=np.float32) if pb_w else np.empty(0, dtype=np.float32),
        np.full(len(replay_x), replay_weight, dtype=np.float32),
    ])

    if verbose:
        print(f"  Training data: {len(ev_x)} evolved + {len(pb_x)} prev_best + {len(replay_x)} replay = {len(all_x)} total")

    # Normalize and train
    all_x_norm = all_x / SCALES
    n = len(all_x_norm)
    idx = np.random.permutation(n)
    split = int(0.85 * n)
    X_train = torch.tensor(all_x_norm[idx[:split]])
    Y_train = torch.tensor(all_y[idx[:split]])
    W_train = torch.tensor(all_w[idx[:split]])
    X_val = torch.tensor(all_x_norm[idx[split:]])
    Y_val = torch.tensor(all_y[idx[split:]])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction='none')

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        pred = model(X_train)
        per_sample_loss = loss_fn(pred, Y_train).mean(dim=1)
        weighted_loss = (per_sample_loss * W_train).mean()
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = nn.MSELoss()(model(X_val), Y_val).item()
            if verbose:
                print(f"  Epoch {epoch+1}: train={weighted_loss.item():.6f} val={val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), Path(save_dir) / "mlp_architect.pt")

    if verbose:
        print(f"  Best val loss: {best_val_loss:.6f}")

    best_path = Path(save_dir) / "mlp_architect.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, weights_only=True))
    model.eval()
    return model


if __name__ == "__main__":
    train_mlp()
