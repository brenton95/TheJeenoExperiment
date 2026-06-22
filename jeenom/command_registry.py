"""Command Registry — L1 of the 5-level hierarchy.

Commands are named compositions of L0 primitives. Every command in this
registry declares its canonical primitive sequence, required claim inputs,
produced claim outputs, static assumptions, and known failure modes.

Motor commands   → executed by Spine via MotorCommandTemplate / SkillPlanTemplate.
Sensory commands → queried by Cortex to build EvidenceFrame requests to Sense.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MotorCommand:
    """L1 motor command: a named sequence of ACTION_PRIMITIVES."""

    name: str
    primitive_sequence: tuple[str, ...] = ()
    required_claims: tuple[str, ...] = ()
    produced_claims: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    failure_modes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SensoryCommand:
    """L1 sensory command: a named observation step with declared evidence needs."""

    name: str
    required_claims: tuple[str, ...] = ()
    produced_claims: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    failure_modes: tuple[str, ...] = ()


MOTOR_COMMANDS: dict[str, MotorCommand] = {
    "navigate_to_object": MotorCommand(
        name="navigate_to_object",
        primitive_sequence=("plan_grid_path", "execute_next_path_action"),
        required_claims=("agent_pose", "target_location", "occupancy_grid"),
        produced_claims=("executed_action",),
        assumptions=("target_location is known and reachable",),
        failure_modes=("no_path_found", "target_unreachable"),
    ),
    "done": MotorCommand(
        name="done",
        primitive_sequence=("done",),
        required_claims=("adjacency_to_target",),
        produced_claims=(),
        assumptions=("agent is adjacent to target",),
        failure_modes=(),
    ),
    "turn_left": MotorCommand(
        name="turn_left",
        primitive_sequence=("turn_left",),
        required_claims=(),
        produced_claims=(),
        assumptions=(),
        failure_modes=(),
    ),
    "turn_right": MotorCommand(
        name="turn_right",
        primitive_sequence=("turn_right",),
        required_claims=(),
        produced_claims=(),
        assumptions=(),
        failure_modes=(),
    ),
    "move_forward": MotorCommand(
        name="move_forward",
        primitive_sequence=("move_forward",),
        required_claims=(),
        produced_claims=(),
        assumptions=(),
        failure_modes=(),
    ),
    "pickup": MotorCommand(
        name="pickup",
        primitive_sequence=("pickup",),
        required_claims=(),
        produced_claims=(),
        assumptions=("agent is adjacent to item",),
        failure_modes=(),
    ),
    "drop": MotorCommand(
        name="drop",
        primitive_sequence=("drop",),
        required_claims=(),
        produced_claims=(),
        assumptions=("agent is carrying an item",),
        failure_modes=(),
    ),
    "toggle": MotorCommand(
        name="toggle",
        primitive_sequence=("toggle",),
        required_claims=(),
        produced_claims=(),
        assumptions=("agent is adjacent to a toggleable object",),
        failure_modes=(),
    ),
    "abort": MotorCommand(
        name="abort",
        primitive_sequence=(),
        required_claims=(),
        produced_claims=(),
        assumptions=(),
        failure_modes=("task_aborted",),
    ),
}

# Skills whose canonical primitive sequence is just [skill_name] — derivable from registry.
DIRECT_ACTION_SKILLS: frozenset[str] = frozenset(
    cmd.name
    for cmd in MOTOR_COMMANDS.values()
    if cmd.primitive_sequence == (cmd.name,)
)

SENSORY_COMMANDS: dict[str, SensoryCommand] = {
    "act_until_evidence": SensoryCommand(
        name="act_until_evidence",
        required_claims=("object_location", "agent_pose"),
        produced_claims=("target_visible", "target_location", "target_object"),
        assumptions=("target predicate and authorized action are present in task params",),
        failure_modes=("object_not_visible",),
    ),
    "locate_object": SensoryCommand(
        name="locate_object",
        required_claims=("object_location", "agent_pose", "occupancy_grid"),
        produced_claims=("target_location",),
        assumptions=("object exists in scene",),
        failure_modes=("object_not_visible",),
    ),
    "navigate_to_object": SensoryCommand(
        name="navigate_to_object",
        required_claims=("object_location", "agent_pose", "occupancy_grid", "adjacency_to_target"),
        produced_claims=("target_location", "adjacency_to_target"),
        assumptions=("object_location known from prior locate_object",),
        failure_modes=("target_lost",),
    ),
    "verify_adjacent": SensoryCommand(
        name="verify_adjacent",
        required_claims=("agent_pose", "object_location", "adjacency_to_target"),
        produced_claims=("adjacency_to_target",),
        assumptions=("agent has navigated to target",),
        failure_modes=("not_adjacent",),
    ),
}

# Fallback evidence needs for any step not explicitly in SENSORY_COMMANDS.
_DEFAULT_EVIDENCE_NEEDS: tuple[str, ...] = ("adjacency_to_target",)


def get_motor_command(skill: str) -> MotorCommand | None:
    """Return the MotorCommand for `skill`, or None if unknown."""
    return MOTOR_COMMANDS.get(skill)


def get_sensory_command(step: str) -> SensoryCommand | None:
    """Return the SensoryCommand for `step`, or None if unknown."""
    return SENSORY_COMMANDS.get(step)


def evidence_needs_for_step(step: str) -> list[str]:
    """Return the required claim names for a procedure step."""
    cmd = SENSORY_COMMANDS.get(step)
    if cmd is not None:
        return list(cmd.required_claims)
    return list(_DEFAULT_EVIDENCE_NEEDS)


def canonical_primitives_for_skill(skill: str) -> list[str] | None:
    """Return the canonical primitive sequence for a motor skill, or None if unknown."""
    cmd = MOTOR_COMMANDS.get(skill)
    if cmd is None:
        return None
    return list(cmd.primitive_sequence)
