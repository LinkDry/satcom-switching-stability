#!/usr/bin/env python3
"""Generate paper figures from experiment data.

Produces:
  - fig_weight_oscillation.pdf  (Fig 2: MLP vs LLM weight stability)
  - fig_method_comparison.pdf   (Fig 3: Known vs Novel regime bar chart)
  - fig_architecture.pdf        (Fig 1: Three-timescale architecture)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
PAPER_DIR = Path(__file__).parent.parent.parent / "paper" / "figures"


def load_switch_log(subdir: str) -> list[dict]:
    """Load switch_log.json from a results subdirectory."""
    path = RESULTS_DIR / subdir / "switch_log.json"
    if not path.exists():
        raise FileNotFoundError(f"No switch_log.json at {path}")
    with open(path) as f:
        return json.load(f)


def fig_weight_oscillation(save_path: str | None = None):
    """Fig 2: MLP vs FT-LLM outage weight over training steps.

    Demonstrates the core finding: LLM weight oscillation destabilizes DRL.
    """
    mlp_log = load_switch_log("GEN_mlp_seed42")
    llm_log = load_switch_log("GEN_finetuned_seed42")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True)

    # Extract outage weights and steps
    mlp_steps = [e["step"] / 1000 for e in mlp_log]
    mlp_outage = [e["weights"]["outage"] for e in mlp_log]

    llm_steps = [e["step"] / 1000 for e in llm_log]
    llm_outage = [e["weights"]["outage"] for e in llm_log]

    # MLP subplot
    ax1.plot(mlp_steps, mlp_outage, "o-", color="#2196F3", linewidth=1.5,
             markersize=6, label="MLP Architect (M3)")
    ax1.axvline(x=250, color="gray", linestyle="--", alpha=0.6, linewidth=1)
    ax1.text(255, max(mlp_outage) * 0.95, "Novel\nregimes", fontsize=8,
             color="gray", va="top")
    ax1.set_ylabel("Outage Weight $w_o$", fontsize=10)
    ax1.set_ylim(-0.1, 2.0)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_title("(a) MLP: Stable weights (range 0.94–1.57)", fontsize=10)

    # LLM subplot
    ax2.plot(llm_steps, llm_outage, "s-", color="#F44336", linewidth=1.5,
             markersize=6, label="Fine-tuned LLM (M4)")
    ax2.axvline(x=250, color="gray", linestyle="--", alpha=0.6, linewidth=1)
    ax2.text(255, 0.9, "Novel\nregimes", fontsize=8, color="gray", va="top")
    ax2.set_ylabel("Outage Weight $w_o$", fontsize=10)
    ax2.set_xlabel("Training Step (×1000)", fontsize=10)
    ax2.set_ylim(-0.1, 1.2)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_title("(b) FT-LLM: Oscillating weights (range 0.004–0.999)", fontsize=10)

    # Annotate double-predict steps in LLM
    double_steps = set()
    for i, e in enumerate(llm_log):
        if i > 0 and llm_log[i]["step"] == llm_log[i - 1]["step"]:
            double_steps.add(e["step"] / 1000)
    for ds in double_steps:
        ax2.axvline(x=ds, color="#FF9800", linestyle=":", alpha=0.5, linewidth=1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def fig_method_comparison(save_path: str | None = None):
    """Fig 3: Bar chart comparing methods on known vs novel regimes."""
    # Data from generalization aggregate results (Round 10, 3-seed means)
    methods = ["Fixed\n(M1)", "Rule\n(M2)", "MLP\n(M3)", "FT-LLM\n(M4)"]
    known_rates = [330.7, 342.8, 357.9, 192.4]  # From Table I
    novel_rates = [0, 305.8, 325.2, 45.3]  # From Table II (Fixed not tested on novel)

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))

    bars1 = ax.bar(x - width / 2, known_rates, width, label="Known Regimes",
                   color="#2196F3", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width / 2, novel_rates, width, label="Novel Regimes",
                   color="#FF9800", alpha=0.85, edgecolor="white")

    # Value labels
    for bar in bars1:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Mean Sum Rate (Mbps)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 420)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def fig_architecture(save_path: str | None = None):
    """Fig 1: Three-timescale architecture diagram."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Layer boxes
    layers = [
        {"y": 5.2, "h": 1.2, "color": "#E3F2FD", "edge": "#1565C0",
         "title": "Strategic Layer: LLM Intent Parser",
         "detail": "NL commands → objective profiles\nLatency: ~14s | Frequency: hours",
         "timescale": "Hours"},
        {"y": 3.0, "h": 1.2, "color": "#FFF3E0", "edge": "#E65100",
         "title": "Tactical Layer: MLP Architect",
         "detail": "KPIs + objectives → reward weights w\nLatency: <1ms | Frequency: minutes",
         "timescale": "Minutes"},
        {"y": 0.8, "h": 1.2, "color": "#E8F5E9", "edge": "#2E7D32",
         "title": "Operational Layer: PPO Agent",
         "detail": "State sₜ → beam allocation aₜ\nLatency: <1ms | Frequency: per-step",
         "timescale": "ms"},
    ]

    for layer in layers:
        rect = mpatches.FancyBboxPatch(
            (0.8, layer["y"]), 7.0, layer["h"],
            boxstyle="round,pad=0.1",
            facecolor=layer["color"],
            edgecolor=layer["edge"],
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(4.3, layer["y"] + layer["h"] - 0.3, layer["title"],
                fontsize=10, fontweight="bold", ha="center", va="top",
                color=layer["edge"])
        ax.text(4.3, layer["y"] + 0.25, layer["detail"],
                fontsize=8, ha="center", va="bottom", color="#424242",
                linespacing=1.4)
        # Timescale label on right
        ax.text(8.5, layer["y"] + layer["h"] / 2, layer["timescale"],
                fontsize=9, ha="left", va="center", color=layer["edge"],
                fontstyle="italic")

    # Arrows between layers
    arrow_style = dict(arrowstyle="-|>", color="#616161", lw=1.5)
    ax.annotate("", xy=(4.3, 4.3), xytext=(4.3, 5.2),
                arrowprops=arrow_style)
    ax.text(5.5, 4.7, "objective\nprofile", fontsize=8, color="#616161",
            ha="center", va="center")

    ax.annotate("", xy=(4.3, 2.1), xytext=(4.3, 3.0),
                arrowprops=arrow_style)
    ax.text(5.5, 2.5, "reward\nweights w", fontsize=8, color="#616161",
            ha="center", va="center")

    # Feedback arrows (KPIs going up)
    feedback_style = dict(arrowstyle="-|>", color="#9E9E9E", lw=1,
                          linestyle="dashed")
    ax.annotate("", xy=(1.5, 3.0), xytext=(1.5, 2.0),
                arrowprops=feedback_style)
    ax.text(0.6, 2.5, "KPIs", fontsize=8, color="#9E9E9E", ha="center",
            va="center", rotation=90)

    # CUSUM detector
    ax.text(1.5, 3.55, "CUSUM\nDetector", fontsize=7, ha="center",
            va="center", color="#E65100",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF3E0",
                      edgecolor="#E65100", linewidth=1))

    # Environment at bottom
    rect_env = mpatches.FancyBboxPatch(
        (2.5, 0.1), 3.6, 0.5,
        boxstyle="round,pad=0.1",
        facecolor="#F5F5F5",
        edgecolor="#757575",
        linewidth=1.5
    )
    ax.add_patch(rect_env)
    ax.text(4.3, 0.35, "LEO Satellite Environment (19 beams)",
            fontsize=8, ha="center", va="center", color="#424242")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def main():
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Generating Paper Figures ===\n")

    print("Fig 1: Architecture diagram...")
    fig_architecture(str(PAPER_DIR / "fig_architecture.pdf"))

    print("Fig 2: Weight oscillation comparison...")
    fig_weight_oscillation(str(PAPER_DIR / "fig_weight_oscillation.pdf"))

    print("Fig 3: Method comparison bar chart...")
    fig_method_comparison(str(PAPER_DIR / "fig_method_comparison.pdf"))

    print("\n=== All figures generated ===")


if __name__ == "__main__":
    main()
