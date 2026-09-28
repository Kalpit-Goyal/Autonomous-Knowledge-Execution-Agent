"""Action registry.

An action is anything that changes state. Registering one gives it a schema the
LLM can call, a risk level, and an irreversibility flag. The flag is the only
piece of behaviour expressed as a literal in code, and it is a safety property
rather than a business decision - whether approval is actually required is
resolved against ``policies.json`` by :func:`needs_approval`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from app.knowledge.policy_store import requires_approval as policy_requires_approval
from app.schemas import RiskLevel

logger = logging.getLogger(__name__)

ActionFn = Callable[[dict[str, Any], "ActionContext"], dict[str, Any]]


@dataclass
class ActionContext:
    """Everything a handler may need beyond its own arguments."""

    session_id: str
    actor: str = "agent"
    why: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    approval_id: str | None = None
    user_id: str = "anonymous"
    approver_role: str = ""


@dataclass(frozen=True)
class ActionSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: ActionFn
    risk: RiskLevel = RiskLevel.LOW
    irreversible: bool = False
    effect: str = ""

    def describe(self) -> str:
        flags = []
        if self.irreversible:
            flags.append("IRREVERSIBLE")
        if needs_approval(self.name):
            flags.append("REQUIRES HUMAN APPROVAL")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        lines = [f"- {self.name}: {self.description}  ({self.effect}){suffix}"]
        schema = self.args_model.model_json_schema()
        required = set(schema.get("required", []))
        props = schema.get("properties", {})
        if props:
            for field_name, spec in props.items():
                mark = "required" if field_name in required else "optional"
                lines.append(
                    f"    {field_name} ({spec.get('type', 'any')}, {mark}): "
                    f"{spec.get('description', '')}"
                )
        return "\n".join(lines)


ACTIONS: dict[str, ActionSpec] = {}


def register(spec: ActionSpec) -> ActionSpec:
    if spec.name in ACTIONS:
        raise ValueError(f"action already registered: {spec.name}")
    ACTIONS[spec.name] = spec
    return spec


def get_action(name: str) -> ActionSpec | None:
    return ACTIONS.get(name)


def is_irreversible(name: str) -> bool:
    spec = ACTIONS.get(name)
    return bool(spec and spec.irreversible)


def needs_approval(name: str) -> bool:
    """Approval requirement: the flag on the action, or the policy file.

    A policy can demand approval for an action whose code does not set the
    irreversible flag, which is how new approval rules get added without code
    changes.
    """
    if is_irreversible(name):
        return True
    try:
        return policy_requires_approval(name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("policy lookup failed for %s: %s", name, exc)
        return False


def registry_as_prompt_block() -> str:
    """The action catalogue, rendered for the decision prompt.

    Built from the registry, so the model can never be offered a stale list.
    """
    if not ACTIONS:
        return "(no actions registered)"
    ordered = sorted(ACTIONS.values(), key=lambda s: (s.irreversible, s.name))
    return "\n".join(spec.describe() for spec in ordered)


def registry_as_tool_schemas() -> list[dict[str, Any]]:
    """OpenAI-style function schemas, derived from each action's pydantic model."""
    schemas: list[dict[str, Any]] = []
    for spec in sorted(ACTIONS.values(), key=lambda s: s.name):
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.args_model.model_json_schema(),
                },
            }
        )
    return schemas


def validate_args(
    spec: ActionSpec, args: dict[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        model = spec.args_model.model_validate(args or {})
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        return None, f"invalid arguments for {spec.name}: {problems}"
    return model.model_dump(), None
