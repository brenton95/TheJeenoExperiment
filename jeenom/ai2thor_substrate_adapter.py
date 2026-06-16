from __future__ import annotations

from typing import Any

from .ai2thor_domain_helper import Ai2thorDomainHelper
from .ai2thor_operational_context import Ai2thorOperationalContext
from .capability_registry import (
    CapabilityRegistry,
    PrimitiveManifest,
    _top_level_task_capability,
)
from .llm_compiler import CompilerBackend
from .memory import OperationalMemory
from .orpi import OrpiManifest
from .plan_cache import PlanCache


def _ai2thor_manifest_dict() -> dict[str, Any]:
    return {
        "name": "ai2thor_primitive_registry_v1",
        "primitives": [
            _top_level_task_capability(
                name="task.go_to_object.apple",
                description=(
                    "Run the go_to_object recipe for a grounded apple target "
                    "(boundary-test skeleton; sense/spine not yet implemented)."
                ),
                inputs=["target.object_type", "target_location"],
                outputs=["task_complete", "execution_report"],
                side_effects=["moves_agent"],
                implementation_status="unsupported",
                runtime_binding=None,
                safety_class="actuation",
                authority_level="operator",
            ),
        ],
    }


class Ai2thorSubstrateAdapter:
    """AI2-THOR implementation skeleton of JEENOM's concrete HOW boundary.

    Boundary-test spike (plan 002): every SubstrateAdapter Protocol method
    is present so the adapter satisfies the contract and wires into a
    RuntimePackage. Controller is injected (never constructed internally) so
    this is testable on WSL2 without a live Unity process. Live-episode
    methods (run_task_episode / run_motor_actions) are not implemented yet —
    that is a later plan (sense/spine work).
    """

    def __init__(
        self,
        *,
        controller: Any = None,
        scene_id: str = "FloorPlan1",
        operational_context: Ai2thorOperationalContext | None = None,
    ) -> None:
        self.controller = controller
        self.scene_id = scene_id
        self.operational_context = operational_context or Ai2thorOperationalContext.default(
            scene_id=scene_id
        )
        self._capability_registry: CapabilityRegistry = CapabilityRegistry(
            PrimitiveManifest.from_dict(_ai2thor_manifest_dict())
        )
        self._orpi_manifest: OrpiManifest = OrpiManifest.from_context_and_registry(
            self.operational_context,
            self._capability_registry,
        )
        self._preview_open = False
        self._task_window_open = False

    def capability_registry(self) -> CapabilityRegistry:
        return self._capability_registry

    def orpi_manifest(self) -> OrpiManifest:
        return self._orpi_manifest

    def create_sense(
        self,
        memory: OperationalMemory,
        compiler: CompilerBackend,
        plan_cache: PlanCache,
    ) -> Any:
        raise NotImplementedError("Ai2thorSubstrateAdapter.create_sense: sense work not started")

    def create_spine(
        self,
        memory: OperationalMemory,
        compiler: CompilerBackend,
        plan_cache: PlanCache,
    ) -> Any:
        raise NotImplementedError("Ai2thorSubstrateAdapter.create_spine: spine work not started")

    def known_action_names(self) -> list[str]:
        return []

    def is_action_known(self, action_name: str) -> bool:
        return action_name in self.known_action_names()

    def prewarm_templates(self, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Ai2thorSubstrateAdapter.prewarm_templates not implemented")

    def open_preview(self, *, seed: int) -> None:
        self._preview_open = True

    def pump_render_window(self) -> None:
        return None

    def close_preview(self) -> None:
        self._preview_open = False

    def close_task_window(self) -> None:
        self._task_window_open = False

    def has_preview_window(self) -> bool:
        return self._preview_open

    def has_task_window(self) -> bool:
        return self._task_window_open

    def sense_idle_scene(self, sense: Any, *, seed: int) -> None:
        raise NotImplementedError("Ai2thorSubstrateAdapter.sense_idle_scene not implemented")

    def run_task_episode(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "success": False,
            "task_complete": False,
            "error": "Ai2thorSubstrateAdapter.run_task_episode: not implemented (spike skeleton)",
        }

    def run_motor_actions(self, *, seed: int, actions: list[str]) -> dict[str, Any]:
        return {
            "success": False,
            "task_complete": False,
            "error": (
                "Ai2thorSubstrateAdapter.run_motor_actions: not implemented (spike skeleton). "
                f"Known: {self.known_action_names()}"
            ),
            "actions_executed": [],
            "steps_taken": 0,
        }

    def close(self) -> None:
        self._preview_open = False
        self._task_window_open = False


def build_ai2thor_runtime_package(
    *,
    controller: Any = None,
    scene_id: str = "FloorPlan1",
) -> Any:
    """Convenience builder mirroring how MiniGrid wires a RuntimePackage."""
    from .runtime_package import RuntimePackage

    adapter = Ai2thorSubstrateAdapter(controller=controller, scene_id=scene_id)
    domain_helper = Ai2thorDomainHelper(operational_context=adapter.operational_context)
    return RuntimePackage(
        substrate=adapter,
        operational_context=adapter.operational_context,
        domain_helper=domain_helper,
    )
