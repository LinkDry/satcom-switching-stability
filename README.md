# When Adaptive Rewards Hurt: Causal Probing and the Switching-Stability Dilemma in LLM-Guided LEO Satellite Scheduling

This repository contains the code, data, and experiment results for the paper:

> **When Adaptive Rewards Hurt: Causal Probing and the Switching-Stability Dilemma in LLM-Guided LEO Satellite Scheduling**
> Yuanhang Li

## Key Findings

- **Switching-stability dilemma**: Near-constant reward weights (342.1 Mbps) outperform optimally-tuned dynamic weights (103.3 ± 96.8 Mbps) because PPO requires quasi-stationary reward signals for value function convergence.
- **Single-variable causal probing**: ±20% perturbation per weight reveals switching penalty has +157 Mbps leverage for polar_handover, +130 Mbps for hot_cold.
- **Systematic architect comparison**: MLP (357.9/325.2 Mbps) > Rule (283.3 Mbps) > LLM (8.2 Mbps) — LLM fails from oscillation, not knowledge.
- **Three-timescale hybrid**: Comparable aggregate satisfaction (0.52 vs 0.57/0.53) while uniquely providing semantic intent differentiation.

## Repository Structure

```
├── simulator/          # LEO satellite beam allocation environment (Gymnasium)
├── agents/             # PPO agent, baselines (MaxWeight heuristic, random)
├── mdp/                # MDP specification system with JSON schema validation
├── llm/                # LLM/MLP architect, RAG, causal probing, intent parsing
├── training/           # Training loops (fixed MDP, two-timescale)
├── evaluation/         # Metrics computation and visualization
├── scripts/            # All experiment entry points
├── configs/            # default.yaml (satellite/channel/PPO params), mdp_schema.json
├── data/finetune/      # Fine-tuning dataset (alpaca format)
├── finetune/           # LoRA fine-tuning config (LlamaFactory)
├── models/             # MLP checkpoints (base, oracle, evolved rounds 1-7)
├── results/            # Experiment metrics (metrics.json + switch_log.json per run)
└── paper/              # LaTeX source and compiled PDF
```

## Quick Start

### Installation

```bash
git clone https://github.com/LinkDry/satcom-switching-stability.git
cd satcom-switching-stability
pip install -r requirements.txt
```

For LLM-based experiments, set your API key:
```bash
export DASHSCOPE_API_KEY="your-key-here"
```

### Reproduce Key Results

**Table I — Architecture comparison (MLP vs Rule vs LLM):**
```bash
python scripts/run_generalization.py --seeds 42 123 456
```

**Table II — Causal sensitivity probing:**
```bash
python scripts/run_evolution_v2.py  # Runs perturbation probes per round
```

**Table III — Intent satisfaction (hybrid architecture):**
```bash
python scripts/run_hybrid_v2.py --seed 42
python scripts/run_hybrid_v2.py --seed 123
python scripts/run_hybrid_v2.py --seed 456
```

**RAG-augmented evolution:**
```bash
python scripts/run_idea2_rag.py --seed 42 --rounds 4 5 6 7 --save-prefix RAG_EVOLVED_v2
python scripts/analyze_rag_results.py --rounds
```

### Pre-computed Results

All experiment results are included in `results/`. Key files:
- `results/generalization_aggregate.json` — 8-seed MLP vs LLM comparison
- `results/probe_round{1..7}.json` — Causal sensitivity data per evolution round
- `results/intent_phase_stats.json` — Per-phase intent satisfaction breakdown
- `results/hybrid_v2_full.log` — Full hybrid experiment log
- `results/anchor_db.json` — RAG anchor database (47 entries)

## Environment

19-beam LEO satellite at 600 km altitude, Ka-band (20 GHz), 3GPP TR 38.821 channel model. Seven traffic regimes: urban, maritime, disaster, mixed, iot_burst, polar_handover, hot_cold. See `configs/default.yaml` for all parameters.

## Models

- `models/mlp/base_mlp.pt` — Base MLP architect (5→5, trained on 20k synthetic samples)
- `models/mlp/oracle_mlp.pt` — Oracle MLP (continually trained on LLM-generated data)
- `models/mlp/intent_mlp.pt` — Intent-aware MLP (10→5, KPI + objective profile)
- `models/evolved/evolved_v2_round{1..7}.pt` — Evolution checkpoints with causal feedback

The fine-tuned Qwen3-4B LoRA adapter (~127 MB) is not included in this repo. To reproduce fine-tuning:
```bash
cd finetune
python train_direct.py  # Or use LlamaFactory with train_lora.yaml
```

## Citation

```bibtex
@article{li2026switching,
  title={When Adaptive Rewards Hurt: Causal Probing and the Switching-Stability Dilemma in LLM-Guided LEO Satellite Scheduling},
  author={Li, Yuanhang},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT
