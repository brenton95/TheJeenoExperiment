from __future__ import annotations

from typing import Any

from . import run_demo
from .capability_registry import CapabilityRegistry
from .llm_compiler import CompilerBackend
from .memory import OperationalMemory
from .minigrid_adapter import MiniGridAdapter
from .minigrid_envs import ensure_custom_minigrid_envs_registered
from .minigrid_operational_context import MiniGridOperationalContext
from .orpi import OrpiManifest
from .plan_cache import PlanCache
from .primitive_library import ACTION_PRIMITIVES
from .sense import MiniGridSense
from .spine import MiniGridSpine


class MiniGridSubstrateAdapter:
    """MiniGrid implementation of JEENOM's concrete HOW boundary."""

    def __init__(
        self,
        *,
        env_id: str,
        render_mode: str,
        observability: str = "partial",
        operational_context: MiniGridOperationalContext | None = None,
    ) -> None:
        ensure_custom_minigrid_envs_registered()
        if observability not in {"partial", "full"}:
            raise ValueError("observability must be 'partial' or 'full'")
        self.env_id = env_id
        self.render_mode = render_mode
        self.observability = observability
        self.operational_context = operational_context or MiniGridOperationalContext.default(
            env_id=env_id
        )
        self._capability_registry: CapabilityRegistry = CapabilityRegistry.minigrid_default()
        self._orpi_manifest: OrpiManifest = OrpiManifest.from_context_and_registry(
            self.operational_context,
            self._capability_registry,
        )
        self.preview_adapter: MiniGridAdapter | None = None
        self.task_adapter: MiniGridAdapter | None = None

    def _build_env(self, render_mode: str | None = None):
        mode = self.render_mode if render_mode is None else render_mode
        if self.observability == "partial":
            return run_demo.build_env(self.env_id, mode)
        return run_demo.build_full_env(self.env_id, mode)

    def capability_registry(self) -> CapabilityRegistry:
        return self._capability_registry

    def orpi_manifest(self) -> OrpiManifest:
        return self._orpi_manifest

    def create_sense(
        self,
        memory: OperationalMemory,
        compiler: CompilerBackend,
        plan_cache: PlanCache,
    ) -> MiniGridSense:
        return MiniGridSense(memory, compiler, plan_cache=plan_cache)

    def create_spine(
        self,
        memory: OperationalMemory,
        compiler: CompilerBackend,
        plan_cache: PlanCache,
    ) -> MiniGridSpine:
        return MiniGridSpine(memory, None, compiler, plan_cache=plan_cache)

    def known_action_names(self) -> list[str]:
        return sorted(ACTION_PRIMITIVES)

    def is_action_known(self, action_name: str) -> bool:
        return action_name in ACTION_PRIMITIVES

    def prewarm_templates(self, **kwargs: Any) -> dict[str, Any]:
        return run_demo.prewarm_jit_cache(**kwargs)

    def open_preview(self, *, seed: int) -> None:
        if self.render_mode != "human":
            return
        self.close_preview()
        env = self._build_env(self.render_mode)
        self.preview_adapter = MiniGridAdapter(env)
        self.preview_adapter.reset(seed=seed)
        try:
            env.render()
        except Exception:  # noqa: BLE001
            pass

    def pump_render_window(self) -> None:
        adapter = self.preview_adapter or self.task_adapter
        if adapter is None:
            return
        try:
            adapter.env.render()
        except Exception:  # noqa: BLE001
            pass

    def close_preview(self) -> None:
        if self.preview_adapter is None:
            return
        self.preview_adapter.close()
        self.preview_adapter = None

    def close_task_window(self) -> None:
        if self.task_adapter is None:
            return
        self.task_adapter.close()
        self.task_adapter = None

    def has_preview_window(self) -> bool:
        return self.preview_adapter is not None

    def has_task_window(self) -> bool:
        return self.task_adapter is not None

    def sense_idle_scene(self, sense: Any, *, seed: int) -> None:
        adapter = self.task_adapter or self.preview_adapter
        close_after = False
        if adapter is None:
            env = self._build_env("none")
            adapter = MiniGridAdapter(env)
            adapter.reset(seed=seed)
            close_after = True
        try:
            observation = adapter.observe()
            sense.sense_idle_scene(observation, env_id=self.env_id, seed=seed)
        finally:
            if close_after:
                adapter.close()

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
        progress_callback: Any,
        task_override: Any = None,
        procedure_override: Any = None,
        step_budget: int | None = None,
    ) -> dict[str, Any]:
        render_adapter = self.task_adapter or self.preview_adapter
        skip_reset = render_adapter is not None
        if render_adapter is None and self.observability == "full":
            render_adapter = MiniGridAdapter(self._build_env(self.render_mode))

        if self.preview_adapter is render_adapter:
            self.preview_adapter = None
        elif self.preview_adapter is not None:
            self.close_preview()
        if self.task_adapter is not None and self.task_adapter is not render_adapter:
            self.close_task_window()

        result = run_demo.run_episode(
            instruction=instruction,
            compiler_name=compiler_name,
            compiler=compiler,
            env_id=self.env_id,
            seed=seed,
            max_loops=max_loops,
            render_mode=self.render_mode,
            memory=memory,
            plan_cache=plan_cache,
            use_cache=plan_cache.enabled,
            prewarm=True,
            keep_render_open=True,
            render_adapter=render_adapter,
            skip_reset=skip_reset,
            progress_callback=progress_callback,
            task_override=task_override,
            procedure_override=procedure_override,
            step_budget=step_budget,
            observability=self.observability,
        )
        self.task_adapter = result.pop("_render_adapter", None)
        return result

    def run_motor_actions(self, *, seed: int, actions: list[str]) -> dict[str, Any]:
        unknown = [action for action in actions if action not in ACTION_PRIMITIVES]
        if unknown:
            return {
                "success": False,
                "task_complete": False,
                "error": (
                    f"Unknown motor action(s): {unknown}. "
                    f"Known: {self.known_action_names()}"
                ),
                "actions_executed": [],
                "steps_taken": 0,
            }
        adapter = self.task_adapter or self.preview_adapter
        if adapter is None:
            env = self._build_env(self.render_mode)
            adapter = MiniGridAdapter(env)
            adapter.reset(seed=seed)
        if adapter is self.preview_adapter:
            self.preview_adapter = None
        self.task_adapter = adapter

        executed: list[str] = []
        terminated = False
        truncated = False
        for action_name in actions:
            spec = ACTION_PRIMITIVES[action_name]
            _, _, terminated, truncated, _ = adapter.act(int(spec.runtime_value))
            executed.append(action_name)
            if self.render_mode == "human":
                try:
                    adapter.env.render()
                except Exception:  # noqa: BLE001
                    pass
            if terminated or truncated:
                break

        return {
            "success": True,
            "task_complete": True,
            "actions_executed": executed,
            "steps_taken": len(executed),
            "terminated": terminated,
            "truncated": truncated,
            "final_state": {"task_complete": True},
            "task": {
                "instruction": " ".join(actions),
                "task_type": "motor_command",
            },
        }

    def close(self) -> None:
        self.close_preview()
        self.close_task_window()
