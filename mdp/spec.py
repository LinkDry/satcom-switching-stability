"""MDP specification dataclass with JSON schema validation."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import jsonschema

SCHEMA_PATH = Path(__file__).parent.parent / "config" / "mdp_schema.json"


@dataclass
class RewardComponent:
    name: str
    weight: float


@dataclass
class Constraint:
    type: str
    value: float


@dataclass
class MDPSpec:
    """A validated MDP specification that configures the Gym environment."""

    spec_id: str
    state_features: list[str]
    action_type: str  # "per_beam" | "per_cluster" | "global_topk"
    reward_components: list[RewardComponent]
    constraints: list[Constraint] = field(default_factory=list)
    action_params: dict = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "spec_id": self.spec_id,
            "state_features": self.state_features,
            "action_type": self.action_type,
            "action_params": self.action_params,
            "reward_components": [
                {"name": rc.name, "weight": rc.weight}
                for rc in self.reward_components
            ],
            "constraints": [
                {"type": c.type, "value": c.value} for c in self.constraints
            ],
            "description": self.description,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "MDPSpec":
        return cls(
            spec_id=d.get("spec_id", "unknown"),
            state_features=d["state_features"],
            action_type=d["action_type"],
            action_params=d.get("action_params", {}),
            reward_components=[
                RewardComponent(rc["name"], rc["weight"])
                for rc in d["reward_components"]
            ],
            constraints=[
                Constraint(c["type"], c["value"])
                for c in d.get("constraints", [])
            ],
            description=d.get("description", ""),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "MDPSpec":
        return cls.from_dict(json.loads(json_str))


def load_schema() -> dict:
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def validate_spec(spec: MDPSpec) -> tuple[bool, Optional[str]]:
    """Validate an MDP spec against the JSON schema.

    Returns (is_valid, error_message).
    """
    schema = load_schema()
    try:
        jsonschema.validate(instance=spec.to_dict(), schema=schema)
        return True, None
    except jsonschema.ValidationError as e:
        return False, str(e.message)
