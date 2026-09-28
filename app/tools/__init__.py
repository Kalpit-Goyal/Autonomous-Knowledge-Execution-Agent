"""Read tools and the action registry.

Two kinds of capability live here.

*Read tools* answer questions and return :class:`Evidence`. They never change
state and are never gated.

*Actions* change state. Every action is registered with a
:class:`ActionSpec` carrying its own risk level, whether it is irreversible, and
a description. The description is what the LLM sees, so adding an action to the
registry is the only step needed to make it available to the agent - there is no
hand-maintained prompt list to fall out of sync.
"""

# Importing the definitions is what populates ACTIONS. Everything downstream
# reads the registry, so this import is load-bearing.
from app.tools import action_defs  # noqa: E402,F401  (import for side effect)
from app.tools.action_registry import (
    ACTIONS,
    ActionContext,
    ActionSpec,
    get_action,
    is_irreversible,
    needs_approval,
    register,
    registry_as_prompt_block,
    registry_as_tool_schemas,
    validate_args,
)
from app.tools.read_tools import READ_TOOLS, read_tool_names, run_read_tool

__all__ = [
    "ACTIONS",
    "ActionContext",
    "ActionSpec",
    "READ_TOOLS",
    "get_action",
    "is_irreversible",
    "needs_approval",
    "read_tool_names",
    "register",
    "registry_as_prompt_block",
    "registry_as_tool_schemas",
    "run_read_tool",
    "validate_args",
]
