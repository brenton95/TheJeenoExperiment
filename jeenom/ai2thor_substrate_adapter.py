from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .ai2thor_domain_helper import Ai2thorDomainHelper
from .ai2thor_operational_context import Ai2thorOperationalContext
from .capability_registry import (
    CapabilityRegistry,
    PrimitiveManifest,
    _top_level_task_capability,
)
from .ai2thor_sense import Ai2thorSense
from .ai2thor_spine import AI2THOR_ACTIONS, Ai2thorSpine
from .cortex import Cortex
from .llm_compiler import CompilerBackend
from .memory import OperationalMemory
from .orpi import OrpiManifest
from .plan_cache import PlanCache, procedure_key
from .primitive_library import TASK_PRIMITIVES
from .schemas import ExecutionContext


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
    ) -> Ai2thorSense:
        return Ai2thorSense(memory, compiler, plan_cache=plan_cache)

    def create_spine(
        self,
        memory: OperationalMemory,
        compiler: CompilerBackend,
        plan_cache: PlanCache,
    ) -> Ai2thorSpine:
        return Ai2thorSpine(memory, self.controller, compiler, plan_cache=plan_cache)

    def known_action_names(self) -> list[str]:
        return sorted(AI2THOR_ACTIONS)

    def is_action_known(self, action_name: str) -> bool:
        return action_name in self.known_action_names()

    def prewarm_templates(self, **kwargs: Any) -> dict[str, Any]:
        return {"compiled_templates": [], "cache_entries": 0}

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
        observation = self.controller.step(action="Pass", renderImage=False)
        sense.sense_idle_scene(observation, env_id=self.scene_id, seed=seed)

    def run_task_episode(
        self,
        *,
        instruction: str,
        compiler_name: str,
        compiler: CompilerBackend,
        seed: int,
        max_loops: int,
        memory: OperationalMemory,
        plan_cache: PlanCache,
        progress_callback: Any = None,
        task_override: Any = None,
        procedure_override: Any = None,
        step_budget: int | None = None,
    ) -> dict[str, Any]:
        cortex = Cortex(memory, compiler, plan_cache=plan_cache)
        spine = Ai2thorSpine(memory, self.controller, compiler, plan_cache=plan_cache)
        # Single source of truth for the reach threshold: the spine derives it
        # from the substrate's actual grid spacing (grid_size * 1.5). Sense must
        # use that same live value for its adjacency claim, not a hardcoded copy
        # — otherwise the two "have we arrived?" judges silently disagree on any
        # scene whose grid spacing isn't the 0.25 default. (See orpi_spec F9.)
        sense = Ai2thorSense(
            memory,
            compiler,
            plan_cache=plan_cache,
            adjacency_threshold=spine.reach_threshold,
        )

        loop_records: list[dict[str, Any]] = []
        runtime_llm_calls_during_render = 0
        cache_miss_during_render = 0

        if task_override is not None:
            task = task_override
        else:
            if progress_callback is not None:
                progress_callback("task_compile_started", {"instruction": instruction})
            task = compiler.compile_task(
                instruction,
                available_task_primitives=TASK_PRIMITIVES,
                memory=memory,
            )
        if progress_callback is not None:
            progress_callback("task_compiled", {"task": asdict(task)})

        procedure_cache_key = procedure_key(task)
        procedure_entry = plan_cache.lookup(procedure_cache_key)
        if procedure_entry is not None:
            procedure = procedure_entry.template
            procedure_cache_status = "hit"
            procedure_source = "cache"
        elif procedure_override is not None:
            procedure = procedure_override
            procedure_cache_status = "override"
            procedure_source = procedure.source
        else:
            if progress_callback is not None:
                progress_callback(
                    "procedure_compile_started",
                    {"task_type": task.task_type, "params": dict(task.params)},
                )
            procedure = compiler.compile_procedure(
                task,
                available_task_primitives=TASK_PRIMITIVES,
                memory=memory,
            )
            procedure_cache_status = "disabled"
            if plan_cache.enabled:
                procedure_cache_status = "miss"
            procedure_source = procedure.source

        readiness = cortex.onboard_task(task, procedure)
        if procedure_entry is None and plan_cache.enabled and readiness.status == "executable":
            plan_cache.store(
                key=procedure_cache_key,
                template_type="procedure",
                template=procedure,
                created_at_loop=-1,
            )

        execution_context = ExecutionContext(
            active_skill="idle",
            params=dict(cortex.resolved_task_params),
        )

        for loop_idx in range(max_loops):
            evidence_frame = cortex.make_evidence_frame()
            if not evidence_frame.needs:
                break

            observation = self.controller.step(action="Pass", renderImage=False)
            evidence, percepts, world_sample, sense_plan, sense_meta = sense.tick(
                observation=observation,
                evidence_frame=evidence_frame,
                execution_context=execution_context,
                loop_index=loop_idx,
                allow_llm_compile=False,
            )
            cortex.update_from_evidence(evidence, world_sample=world_sample)
            # Render-loop invariant metrics: derive from the tick meta, never
            # hardcode. The AI2-THOR sense/spine are deterministic and report
            # runtime_compiler_call=False, so these stay 0 — but they stay 0
            # *because the meta says so*, so the golden-path assertion has teeth
            # if a future change introduces a runtime compile. Mirrors
            # run_demo.run_episode's counting.
            if sense_meta.get("runtime_compiler_call"):
                if sense_meta.get("cache") == "miss":
                    cache_miss_during_render += 1
                if sense_meta.get("compiler_backend") == "llm_compiler":
                    runtime_llm_calls_during_render += 1
            contract = cortex.choose_execution_contract()

            loop_record: dict[str, Any] = {
                "loop": loop_idx,
                "evidence_needs": list(evidence_frame.needs),
                "sense_plan": [step if isinstance(step, str) else step.name for step in sense_plan],
                "sense_plan_cache": sense_meta["cache"],
                "sense_plan_source": sense_meta["source"],
                "world_sample": world_sample.summary(),
                "operational_evidence": evidence.claims,
                "percepts": percepts.cues,
            }

            if contract is None:
                loop_record["contract"] = None
                loop_record["skill_plan"] = None
                loop_record["skill_plan_cache"] = None
                loop_record["skill_plan_source"] = None
                loop_record["report"] = None
                loop_record["action"] = None
                loop_record["compiler_call_count_so_far"] = len(compiler.call_history)
                loop_records.append(loop_record)
                break

            report, execution_context, skill_plan, skill_meta = spine.tick(
                contract,
                percepts,
                loop_index=loop_idx,
                allow_llm_compile=False,
            )
            cortex.update_from_report(report)
            if skill_meta.get("runtime_compiler_call"):
                if skill_meta.get("cache") == "miss":
                    cache_miss_during_render += 1
                if skill_meta.get("compiler_backend") == "llm_compiler":
                    runtime_llm_calls_during_render += 1

            loop_record["contract"] = asdict(contract)
            loop_record["skill_plan"] = [
                step if isinstance(step, str) else step.name for step in skill_plan
            ]
            loop_record["skill_plan_cache"] = skill_meta["cache"]
            loop_record["skill_plan_source"] = skill_meta["source"]
            loop_record["report"] = asdict(report)
            loop_record["action"] = report.progress.get("executed_action")
            loop_record["compiler_call_count_so_far"] = len(compiler.call_history)
            loop_records.append(loop_record)

            if cortex.execution_state["task_complete"] or report.status == "failed":
                break

        memory_updates = cortex.finalize()
        budget_exhausted = any(
            isinstance(lr.get("report"), dict)
            and lr["report"].get("reason") == "budget_exhausted"
            for lr in loop_records
        )
        final_state = dict(cortex.execution_state)
        final_state["budget_exhausted"] = budget_exhausted

        compiler_usage = compiler.usage_summary()
        return {
            "compiler_backend": compiler.active_backend,
            "compiler_logs": list(compiler.logs),
            "compiler_usage": compiler_usage,
            "task": asdict(task),
            "procedure": asdict(procedure),
            "procedure_cache": {
                "status": procedure_cache_status,
                "source": procedure_source,
            },
            "readiness": asdict(readiness),
            "loop_records": loop_records,
            "final_claims": dict(cortex.claims),
            "final_state": final_state,
            "persisted_knowledge": dict(memory.knowledge),
            "episodic_memory": dict(memory.episodic_memory),
            "last_world_sample": (
                cortex.last_world_sample.summary()
                if cortex.last_world_sample
                else None
            ),
            "trace_events": [asdict(event) for event in cortex.trace],
            "memory_updates": [asdict(update) for update in memory_updates],
            "plan_cache": plan_cache.summary(include_entries=True),
            "runtime_llm_calls_during_render": runtime_llm_calls_during_render,
            "cache_miss_during_render": cache_miss_during_render,
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
