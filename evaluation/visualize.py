"""Plotting utilities for experiment results."""

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_regime_comparison(
    results: dict[str, dict],
    metric: str = "mean_mbps",
    title: str = "Per-Regime Sum Rate Comparison",
    save_path: Optional[str] = None,
):
    """Bar chart comparing methods across regimes.

    Args:
        results: {method_name: {regime: {metric: value}}}
    """
    methods = list(results.keys())
    regimes = sorted(
        set(r for m in results.values() for r in m.keys())
    )

    x = np.arange(len(regimes))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, method in enumerate(methods):
        values = [results[method].get(r, {}).get(metric, 0) for r in regimes]
        ax.bar(x + i * width, values, width, label=method)

    ax.set_xlabel("Traffic Regime")
    ax.set_ylabel(f"Sum Rate ({metric})")
    ax.set_title(title)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(regimes)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_time_series(
    series: dict[str, list[float]],
    regime_boundaries: Optional[list[int]] = None,
    window: int = 50,
    title: str = "Sum Rate Over Time",
    ylabel: str = "Sum Rate (Mbps)",
    save_path: Optional[str] = None,
):
    """Time series plot with optional regime boundary markers and smoothing."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for name, values in series.items():
        # Smooth with rolling average
        arr = np.array(values)
        if len(arr) > window:
            smoothed = np.convolve(arr, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(arr)), smoothed, label=name, alpha=0.8)
        else:
            ax.plot(arr, label=name, alpha=0.8)

    # Mark regime boundaries
    if regime_boundaries:
        for b in regime_boundaries:
            ax.axvline(x=b, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ablation_table(
    ablation_results: dict[str, float],
    title: str = "Ablation Study",
    save_path: Optional[str] = None,
):
    """Horizontal bar chart for ablation study results."""
    methods = list(ablation_results.keys())
    values = list(ablation_results.values())

    fig, ax = plt.subplots(figsize=(8, max(4, len(methods) * 0.6)))
    colors = ["#2196F3" if "Full" in m or "Ours" in m else "#90CAF9" for m in methods]
    bars = ax.barh(methods, values, color=colors)
    ax.set_xlabel("Mean Sum Rate (Mbps)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cost_performance_pareto(
    methods: dict[str, tuple[float, float]],
    title: str = "Cost-Performance Pareto",
    save_path: Optional[str] = None,
):
    """Scatter plot: x = LLM API calls, y = performance.

    Args:
        methods: {name: (api_calls, mean_rate)}
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, (cost, perf) in methods.items():
        marker = "s" if "Ours" in name or "Full" in name else "o"
        ax.scatter(cost, perf, s=100, marker=marker, zorder=5)
        ax.annotate(name, (cost, perf), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)

    ax.set_xlabel("LLM API Calls per Episode")
    ax.set_ylabel("Mean Sum Rate (Mbps)")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
