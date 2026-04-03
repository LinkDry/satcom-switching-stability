"""Policy transfer / warm-start across MDP changes."""

import torch
import numpy as np
from stable_baselines3 import PPO


def transfer_policy(
    source_model: PPO,
    target_model: PPO,
    fine_tune_steps: int = 1000,
) -> dict:
    """Transfer compatible parameters from source to target model.

    When MDP changes, observation/action dimensions may differ.
    - Transfer layers with matching dimensions
    - Reinitialize layers with mismatched dimensions
    - Run brief fine-tuning to adapt

    Returns transfer statistics.
    """
    source_params = dict(source_model.policy.named_parameters())
    target_params = dict(target_model.policy.named_parameters())

    transferred = 0
    reinitialized = 0

    for name, target_param in target_params.items():
        if name in source_params:
            source_param = source_params[name]
            if source_param.shape == target_param.shape:
                target_param.data.copy_(source_param.data)
                transferred += 1
            else:
                # Reinitialize with Xavier for weights, zero for biases
                if "weight" in name and target_param.dim() >= 2:
                    torch.nn.init.xavier_uniform_(target_param.data)
                elif "bias" in name:
                    target_param.data.zero_()
                reinitialized += 1
        else:
            reinitialized += 1

    # Also transfer optimizer state for matching parameters
    _transfer_optimizer_state(source_model, target_model)

    stats = {
        "transferred": transferred,
        "reinitialized": reinitialized,
        "total_params": len(target_params),
    }

    # Brief fine-tuning
    if fine_tune_steps > 0:
        target_model.learn(total_timesteps=fine_tune_steps)
        stats["fine_tune_steps"] = fine_tune_steps

    return stats


def _transfer_optimizer_state(source: PPO, target: PPO):
    """Transfer optimizer momentum/variance for matching parameters."""
    try:
        source_opt = source.policy.optimizer.state_dict()
        target_opt = target.policy.optimizer.state_dict()

        # Only transfer if param groups match in count
        if len(source_opt.get("state", {})) == len(target_opt.get("state", {})):
            for key in source_opt["state"]:
                if key in target_opt["state"]:
                    src_state = source_opt["state"][key]
                    tgt_state = target_opt["state"][key]
                    for k, v in src_state.items():
                        if k in tgt_state and isinstance(v, torch.Tensor):
                            if v.shape == tgt_state[k].shape:
                                tgt_state[k].copy_(v)
            target.policy.optimizer.load_state_dict(target_opt)
    except Exception:
        pass  # Optimizer transfer is best-effort


def create_warm_started_agent(
    env,
    source_agent,
    new_obs_dim: int | None = None,
    fine_tune_steps: int = 1000,
    **ppo_kwargs,
) -> tuple:
    """Create a new PPO agent warm-started from a source agent.

    Used when MDP spec changes the observation/action space.
    """
    from agents.ppo_agent import PPOAgent

    # Create new agent with the new env
    new_agent = PPOAgent(env, **ppo_kwargs)

    # Transfer what we can
    stats = transfer_policy(
        source_agent.model,
        new_agent.model,
        fine_tune_steps=fine_tune_steps,
    )

    return new_agent, stats
