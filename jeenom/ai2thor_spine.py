from __future__ import annotations

from typing import Any

from .schemas import ExecutionContext, ExecutionReport


AI2THOR_ACTIONS: dict[str, str] = {
    "move_forward": "MoveAhead",
    "turn_right": "RotateRight",
    "turn_left": "RotateLeft",
}


class Ai2thorSpine:
    """Motor dispatch for AI2-THOR: maps JEENO primitives to controller.step() calls.

    No path planning (blocked on sense + wrong approach for AI2-THOR).
    No sense dependency. Controller is injected, enabling mock-based testing.
    """

    def __init__(
        self,
        memory: Any,
        controller: Any,
        compiler: Any,
        plan_cache: Any = None,
    ) -> None:
        self.memory = memory
        self.controller = controller
        self.compiler = compiler
        self.plan_cache = plan_cache
        self.active_skill: str | None = None

    def tick(
        self,
        execution_contract: Any,
        percepts: Any = None,
        loop_index: int = 0,
        allow_llm_compile: bool = True,
    ) -> tuple[ExecutionReport, ExecutionContext, list[str], dict[str, Any]]:
        self.active_skill = execution_contract.skill
        skill = execution_contract.skill

        if skill == "done":
            report = ExecutionReport(
                status="running",
                progress={"contract": skill},
                source="spine",
            )
        elif skill in AI2THOR_ACTIONS:
            report = self._execute_env_action(skill)
        else:
            report = ExecutionReport(
                status="failed",
                reason=f"unknown_action_primitive:{skill}",
                progress={"contract": skill},
                source="spine",
            )

        context = ExecutionContext(
            active_skill=skill,
            params=dict(execution_contract.params),
        )
        plan = [skill]
        cache_meta: dict[str, Any] = {
            "cache": "disabled",
            "source": "direct_dispatch",
            "cache_key": skill,
            "compiler_backend": "ai2thor_spine",
            "runtime_compiler_call": False,
        }
        return report, context, plan, cache_meta

    def _execute_env_action(self, name: str) -> ExecutionReport:
        action_str = AI2THOR_ACTIONS.get(name)
        if action_str is None:
            return ExecutionReport(
                status="failed",
                reason=f"unknown_action_primitive:{name}",
                progress={"contract": name},
                source="spine",
            )

        event = self.controller.step(action=action_str, renderImage=False)
        return ExecutionReport(
            status="running",
            progress={
                "executed_action": name,
                "ai2thor_action": action_str,
                "metadata_keys": list(event.metadata.keys()) if hasattr(event, "metadata") else [],
            },
            source="spine",
        )
