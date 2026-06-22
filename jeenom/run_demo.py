from __future__ import annotations

import argparse
from dataclasses import asdict
from pprint import pprint

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from .cortex import Cortex
from .llm_compiler import CompilerBackend, build_compiler
from .memory import OperationalMemory
from .minigrid_envs import ensure_custom_minigrid_envs_registered
from .minigrid_adapter import MiniGridAdapter
from .plan_cache import PlanCache, procedure_key
from .primitive_library import ACTION_PRIMITIVES, TASK_PRIMITIVES
from .schemas import (
    EvidenceFrame,
    ExecutionContext,
    ExecutionContract,
    ExecutionReport,
    Percepts,
    ProcedureRecipe,
    TaskRequest,
)
from .sense import MiniGridSense
from .spine import MiniGridSpine


def build_env(env_id: str, render_mode: str | None):
    ensure_custom_minigrid_envs_registered()
    kwargs = {}
    if render_mode != "none":
        kwargs["render_mode"] = render_mode
    return gym.make(env_id, **kwargs)


def build_full_env(env_id: str, render_mode: str | None):
    ensure_custom_minigrid_envs_registered()
    kwargs = {}
    if render_mode != "none":
        kwargs["render_mode"] = render_mode
    return FullyObsWrapper(gym.make(env_id, **kwargs))


def run_motor_sequence(
    env_id: str,
    seed: int,
    render_mode: str,
    actions: list[str],
) -> dict:
    """Execute a sequence of motor-primitive actions for an authorized station caller.

    Each entry in `actions` must be a key in ACTION_PRIMITIVES with runtime_kind='env_action'.
    Returns a result dict compatible with last_result (task_complete always True).
    """
    unknown = [a for a in actions if a not in ACTION_PRIMITIVES]
    if unknown:
        return {
            "success": False,
            "task_complete": False,
            "error": f"Unknown motor action(s): {unknown}. Known: {sorted(ACTION_PRIMITIVES)}",
            "actions_executed": [],
            "steps_taken": 0,
        }
    env = build_env(env_id, render_mode)
    adapter = MiniGridAdapter(env)
    adapter.reset(seed=seed)
    executed: list[str] = []
    try:
        for action_name in actions:
            spec = ACTION_PRIMITIVES[action_name]
            adapter.act(int(spec.runtime_value))
            executed.append(action_name)
    finally:
        adapter.close()
    return {
        "success": True,
        "task_complete": True,
        "actions_executed": executed,
        "steps_taken": len(executed),
        "final_state": {"task_complete": True},
        "task": {"instruction": " ".join(actions), "task_type": "motor_command"},
    }


def _is_llm_compiler(compiler: CompilerBackend) -> bool:
    return getattr(compiler, "name", None) == "llm_compiler"


def _prewarm_target_location(task_request) -> tuple[int, int] | None:
    target_location = task_request.params.get("target_location")
    if isinstance(target_location, tuple):
        return target_location
    if isinstance(target_location, list) and len(target_location) == 2:
        return (int(target_location[0]), int(target_location[1]))
    return (0, 0)


def _assemble_result(
    *,
    compiler,
    task,
    procedure,
    procedure_cache_status: str,
    procedure_source: str,
    readiness,
    loop_records,
    cortex,
    memory,
    memory_updates,
    plan_cache,
    print_cache: bool,
    jit_prewarm: bool,
    prewarm_summary,
    runtime_llm_calls_during_render: int,
    cache_miss_during_render: int,
    render_adapter=None,
):
    compiler_usage = compiler.usage_summary()
    budget_exhausted = any(
        isinstance(lr.get("report"), dict)
        and lr["report"].get("reason") == "budget_exhausted"
        for lr in loop_records
    )
    final_state = dict(cortex.execution_state)
    final_state["budget_exhausted"] = budget_exhausted
    result = {
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
        "last_world_sample": cortex.last_world_sample.summary() if cortex.last_world_sample else None,
        "trace_events": [asdict(event) for event in cortex.trace],
        "memory_updates": [asdict(update) for update in memory_updates],
        "plan_cache": plan_cache.summary(include_entries=True),
        "print_cache": print_cache,
        "jit_prewarm": jit_prewarm,
        "jit_prewarm_summary": prewarm_summary,
        "runtime_llm_calls_during_render": runtime_llm_calls_during_render,
        "cache_miss_during_render": cache_miss_during_render,
    }
    if render_adapter is not None:
        result["_render_adapter"] = render_adapter
    return result


def _probe_requested_target(env_id: str, seed: int, task_request) -> dict[str, object] | None:
    requested_type = task_request.params.get("object_type")
    requested_color = task_request.params.get("color")
    if requested_type is None and requested_color is None:
        return None

    probe_env = build_env(env_id, "none")
    probe_adapter = MiniGridAdapter(probe_env)
    try:
        observation = probe_adapter.reset(seed=seed)
        matching_targets = probe_adapter.find_matching_objects(
            object_type=requested_type,
            color=requested_color,
        )
        available_targets = probe_adapter.find_matching_objects(object_type=requested_type)
        return {
            "env_mission": observation.raw.get("mission"),
            "matched_target": matching_targets[0] if matching_targets else None,
            "available_targets": available_targets,
        }
    finally:
        probe_adapter.close()


def prewarm_jit_cache(
    task_request,
    procedure_recipe,
    cortex,
    sense,
    spine,
    plan_cache,
    progress_callback=None,
):
    compiled_templates: list[dict[str, str]] = []

    if plan_cache.enabled:
        cache_key = procedure_key(task_request)
        if cache_key not in plan_cache.entries:
            plan_cache.store(
                key=cache_key,
                template_type="procedure",
                template=procedure_recipe,
                created_at_loop=-1,
            )
        compiled_templates.append(
            {
                "template_type": "procedure",
                "label": procedure_recipe.task_type,
                "status": "ready",
            }
        )

    base_params = dict(cortex.resolved_task_params)
    navigate_params = dict(base_params)
    navigate_params["target_location"] = _prewarm_target_location(task_request)
    dummy_percepts = Percepts(
        cues={
            "agent_pose": {"x": 0, "y": 0, "dir": 0},
            "target_location": navigate_params["target_location"],
            "occupancy_grid": [[True]],
            "direction": 0,
            "adjacency_to_target": False,
            "passable_positions": {(0, 0)},
            "grid_size": (1, 1),
            "target_object": {
                "type": base_params.get("object_type"),
                "color": base_params.get("color"),
            },
        },
        source="prewarm",
    )

    sense_warmups = [
        (
            "locate_object:idle",
            EvidenceFrame(
                needs=["object_location", "agent_pose", "occupancy_grid"],
                context=dict(base_params),
                active_step="locate_object",
            ),
            ExecutionContext(active_skill="idle", params=dict(base_params)),
        ),
        (
            "locate_object:turn_right",
            EvidenceFrame(
                needs=["object_location", "agent_pose", "occupancy_grid"],
                context=dict(base_params),
                active_step="locate_object",
            ),
            ExecutionContext(active_skill="turn_right", params=dict(base_params)),
        ),
        (
            "navigate_to_object:turn_right",
            EvidenceFrame(
                needs=["object_location", "agent_pose", "occupancy_grid", "adjacency_to_target"],
                context=dict(base_params),
                active_step="navigate_to_object",
            ),
            ExecutionContext(active_skill="turn_right", params=dict(base_params)),
        ),
        (
            "navigate_to_object:navigate_to_object",
            EvidenceFrame(
                needs=["object_location", "agent_pose", "occupancy_grid", "adjacency_to_target"],
                context=dict(base_params),
                active_step="navigate_to_object",
            ),
            ExecutionContext(active_skill="navigate_to_object", params=dict(base_params)),
        ),
        (
            "verify_adjacent:navigate_to_object",
            EvidenceFrame(
                needs=["agent_pose", "object_location", "adjacency_to_target"],
                context=dict(base_params),
                active_step="verify_adjacent",
            ),
            ExecutionContext(active_skill="navigate_to_object", params=dict(navigate_params)),
        ),
        (
            "done:navigate_to_object",
            EvidenceFrame(
                needs=["adjacency_to_target"],
                context=dict(base_params),
                active_step="done",
            ),
            ExecutionContext(active_skill="navigate_to_object", params=dict(navigate_params)),
        ),
    ]
    if "act_until_evidence" in procedure_recipe.steps:
        sense_warmups = [
            (
                "act_until_evidence:idle",
                EvidenceFrame(
                    needs=["object_location", "agent_pose"],
                    context=dict(base_params),
                    active_step="act_until_evidence",
                ),
                ExecutionContext(active_skill="idle", params=dict(base_params)),
            ),
            (
                "act_until_evidence:action",
                EvidenceFrame(
                    needs=["object_location", "agent_pose"],
                    context=dict(base_params),
                    active_step="act_until_evidence",
                ),
                ExecutionContext(
                    active_skill=str(base_params.get("action_name") or "move_forward"),
                    params=dict(base_params),
                ),
            ),
        ]

    seen_sense_labels: set[str] = set()
    for label, evidence_frame, execution_context in sense_warmups:
        if label in seen_sense_labels:
            continue
        seen_sense_labels.add(label)
        if progress_callback is not None:
            progress_callback(
                "prewarm_template",
                {"template_type": "sense", "label": label},
            )
        _, meta = sense._resolve_template(
            evidence_frame=evidence_frame,
            execution_context=execution_context,
            loop_index=-1,
            allow_llm_compile=True,
        )
        compiled_templates.append(
            {
                "template_type": "sense",
                "label": label,
                "status": meta["cache"],
            }
        )

    skill_warmups = [
        (
            "turn_right",
            ExecutionContract(
                skill="turn_right",
                params=dict(base_params),
                stop_conditions=["adjacent_to_target", "task_complete"],
                source="cortex",
            ),
        ),
        (
            "navigate_to_object",
            ExecutionContract(
                skill="navigate_to_object",
                params=dict(navigate_params),
                stop_conditions=["adjacent_to_target", "task_complete"],
                source="cortex",
            ),
        ),
        (
            "done",
            ExecutionContract(
                skill="done",
                params=dict(navigate_params),
                stop_conditions=["adjacent_to_target", "task_complete"],
                source="cortex",
            ),
        ),
    ]
    if "act_until_evidence" in procedure_recipe.steps:
        action_name = str(base_params.get("action_name") or "move_forward")
        skill_warmups = [
            (
                action_name,
                ExecutionContract(
                    skill=action_name,
                    params=dict(base_params),
                    stop_conditions=[
                        str(base_params.get("stop_claim") or "target_visible")
                    ],
                    source="cortex",
                ),
            )
        ]

    seen_skill_labels: set[str] = set()
    for label, contract in skill_warmups:
        if label in seen_skill_labels:
            continue
        seen_skill_labels.add(label)
        if progress_callback is not None:
            progress_callback(
                "prewarm_template",
                {"template_type": "skill", "label": label},
            )
        _, meta = spine._resolve_template(
            execution_contract=contract,
            percepts=dummy_percepts,
            loop_index=-1,
            allow_llm_compile=True,
        )
        compiled_templates.append(
            {
                "template_type": "skill",
                "label": label,
                "status": meta["cache"],
            }
        )

    return {
        "compiled_templates": compiled_templates,
        "cache_entries": len(plan_cache.entries),
    }


def run_episode(
    instruction: str | None = None,
    compiler_name: str = "smoke_test",
    env_id: str = "MiniGrid-GoToDoor-8x8-v0",
    seed: int = 42,
    max_loops: int = 128,
    render_mode: str = "none",
    memory_root=None,
    use_cache: bool = True,
    print_cache: bool = False,
    prewarm: bool = True,
    compiler: CompilerBackend | None = None,
    memory: OperationalMemory | None = None,
    plan_cache: PlanCache | None = None,
    keep_render_open: bool = False,
    render_adapter: MiniGridAdapter | None = None,
    skip_reset: bool = False,
    progress_callback=None,
    task_override: TaskRequest | None = None,
    procedure_override: ProcedureRecipe | None = None,
    step_budget: int | None = None,
    observability: str = "partial",
):
    if observability not in {"partial", "full"}:
        raise ValueError("observability must be 'partial' or 'full'")
    episode_build_env = build_full_env if observability == "full" else build_env
    compiler = compiler or build_compiler(compiler_name)
    memory = memory or OperationalMemory(root=memory_root)
    plan_cache = plan_cache or PlanCache(enabled=use_cache)
    cache_enabled = plan_cache.enabled
    cortex = Cortex(memory, compiler, plan_cache=plan_cache)
    sense = MiniGridSense(memory, compiler, plan_cache=plan_cache)
    spine = MiniGridSpine(memory, None, compiler, plan_cache=plan_cache, step_budget=step_budget)

    loop_records = []
    jit_prewarm = False
    prewarm_summary = None
    runtime_llm_calls_during_render = 0
    cache_miss_during_render = 0
    target_probe = None
    aligned_target = None
    env = render_adapter.env if render_adapter is not None else None
    adapter = render_adapter
    retain_adapter = keep_render_open or render_adapter is not None
    adapter_handed_off = False

    try:
        observation = None
        if instruction is None:
            if adapter is None:
                env = episode_build_env(env_id, render_mode)
                adapter = MiniGridAdapter(env)
            observation = adapter.reset(seed=seed)
            operator_instruction = observation.raw.get("mission") or "Find the goal."
        else:
            operator_instruction = instruction
        if task_override is not None:
            task = task_override
        else:
            if progress_callback is not None:
                progress_callback(
                    "task_compile_started",
                    {"instruction": operator_instruction},
                )
            task = compiler.compile_task(
                operator_instruction,
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
            if cache_enabled:
                procedure_cache_status = "miss"
            procedure_source = procedure.source
        if progress_callback is not None:
            progress_callback(
                "procedure_ready",
                {
                    "procedure": asdict(procedure),
                    "cache_status": procedure_cache_status,
                    "cache_key": procedure_cache_key,
                },
            )

        readiness = cortex.onboard_task(task, procedure)
        if progress_callback is not None:
            progress_callback("readiness_checked", {"readiness": asdict(readiness)})
        if procedure_entry is None and cache_enabled and readiness.status == "executable":
            plan_cache.store(
                key=procedure_cache_key,
                template_type="procedure",
                template=procedure,
                created_at_loop=-1,
            )

        if instruction is not None and task.task_type == "go_to_object":
            target_probe = _probe_requested_target(env_id=env_id, seed=seed, task_request=task)
            if target_probe is not None:
                aligned_target = target_probe.get("matched_target")
                if aligned_target is None:
                    requested_target = {
                        "color": task.params.get("color"),
                        "object_type": task.params.get("object_type"),
                    }
                    cortex.execution_state["current_skill"] = "abort"
                    cortex.execution_state["last_report"] = asdict(
                        ExecutionReport(
                            status="failed",
                            reason="target_absent",
                            progress={
                                "requested_target": requested_target,
                                "available_targets": target_probe["available_targets"],
                                "env_mission": target_probe["env_mission"],
                            },
                            source="cortex",
                        )
                    )
                    cortex.record_trace(
                        "target_preflight_failed",
                        {
                            "requested_target": requested_target,
                            "available_targets": target_probe["available_targets"],
                            "env_mission": target_probe["env_mission"],
                        },
                        step_name=cortex._current_step_name(),
                    )
                    memory_updates = cortex.finalize()
                    result = _assemble_result(
                        compiler=compiler,
                        task=task,
                        procedure=procedure,
                        procedure_cache_status=procedure_cache_status,
                        procedure_source=procedure_source,
                        readiness=readiness,
                        loop_records=loop_records,
                        cortex=cortex,
                        memory=memory,
                        memory_updates=memory_updates,
                        plan_cache=plan_cache,
                        print_cache=print_cache,
                        jit_prewarm=jit_prewarm,
                        prewarm_summary=prewarm_summary,
                        runtime_llm_calls_during_render=runtime_llm_calls_during_render,
                        cache_miss_during_render=cache_miss_during_render,
                        render_adapter=adapter if retain_adapter else None,
                    )
                    adapter_handed_off = retain_adapter and adapter is not None
                    return result

        should_prewarm = (
            prewarm
            and render_mode == "human"
            and cache_enabled
            and _is_llm_compiler(compiler)
            and readiness.status == "executable"
        )
        if should_prewarm:
            if progress_callback is not None:
                progress_callback("prewarm_started", {"procedure": list(procedure.steps)})
            prewarm_summary = prewarm_jit_cache(
                task_request=task,
                procedure_recipe=procedure,
                cortex=cortex,
                sense=sense,
                spine=spine,
                plan_cache=plan_cache,
                progress_callback=progress_callback,
            )
            jit_prewarm = True
            print("JIT PREWARM")
            print("compiled templates:")
            pprint(prewarm_summary["compiled_templates"])
            print("cache entries:")
            pprint(prewarm_summary["cache_entries"])
            print()
            if progress_callback is not None:
                progress_callback("prewarm_finished", dict(prewarm_summary))

        if adapter is None:
            env = episode_build_env(env_id, render_mode)
            adapter = MiniGridAdapter(env)
        if skip_reset:
            # Continue from current adapter state — window stays open, position preserved.
            if aligned_target is not None:
                adapter.retarget_to_object(aligned_target)
            observation = adapter.observe()
        else:
            observation = adapter.reset(seed=seed)
            if aligned_target is not None:
                adapter.retarget_to_object(aligned_target)
                observation = adapter.observe()
        spine.adapter = adapter
        if progress_callback is not None:
            progress_callback("runtime_started", {"render_mode": render_mode})

        execution_context = ExecutionContext(
            active_skill="idle",
            params=dict(cortex.resolved_task_params),
        )
        allow_runtime_llm = not (render_mode == "human" and cache_enabled and _is_llm_compiler(compiler))

        for loop_idx in range(max_loops):
            evidence_frame = cortex.make_evidence_frame()
            if not evidence_frame.needs:
                break

            evidence, percepts, world_sample, sense_plan, sense_meta = sense.tick(
                observation=adapter.observe(),
                evidence_frame=evidence_frame,
                execution_context=execution_context,
                loop_index=loop_idx,
                allow_llm_compile=allow_runtime_llm,
            )
            cortex.update_from_evidence(evidence, world_sample=world_sample)
            contract = cortex.choose_execution_contract()

            if render_mode == "human" and sense_meta.get("runtime_compiler_call"):
                if sense_meta["cache"] == "miss":
                    cache_miss_during_render += 1
                if sense_meta.get("compiler_backend") == "llm_compiler":
                    runtime_llm_calls_during_render += 1
                print("WARNING: runtime compiler call during human render")

            loop_record = {
                "loop": loop_idx,
                "evidence_needs": list(evidence_frame.needs),
                "sense_plan": [step.name for step in sense_plan],
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
                allow_llm_compile=allow_runtime_llm,
            )
            cortex.update_from_report(report)

            if render_mode == "human" and skill_meta.get("runtime_compiler_call"):
                if skill_meta["cache"] == "miss":
                    cache_miss_during_render += 1
                if skill_meta.get("compiler_backend") == "llm_compiler":
                    runtime_llm_calls_during_render += 1
                print("WARNING: runtime compiler call during human render")

            loop_record["contract"] = asdict(contract)
            loop_record["skill_plan"] = [step.name for step in skill_plan]
            loop_record["skill_plan_cache"] = skill_meta["cache"]
            loop_record["skill_plan_source"] = skill_meta["source"]
            loop_record["report"] = asdict(report)
            loop_record["action"] = report.progress.get("executed_action")
            loop_record["compiler_call_count_so_far"] = len(compiler.call_history)
            loop_records.append(loop_record)

            if cortex.execution_state["task_complete"] or report.status == "failed":
                break

        memory_updates = cortex.finalize()
        result = _assemble_result(
            compiler=compiler,
            task=task,
            procedure=procedure,
            procedure_cache_status=procedure_cache_status,
            procedure_source=procedure_source,
            readiness=readiness,
            loop_records=loop_records,
            cortex=cortex,
            memory=memory,
            memory_updates=memory_updates,
            plan_cache=plan_cache,
            print_cache=print_cache,
            jit_prewarm=jit_prewarm,
            prewarm_summary=prewarm_summary,
            runtime_llm_calls_during_render=runtime_llm_calls_during_render,
            cache_miss_during_render=cache_miss_during_render,
            render_adapter=adapter if retain_adapter else None,
        )
        adapter_handed_off = retain_adapter and adapter is not None
        return result
    finally:
        if adapter is not None and (not retain_adapter or not adapter_handed_off):
            adapter.close()


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run the JEENOM MiniGrid demo.")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--compiler", choices=["smoke_test", "llm"], default="smoke_test")
    parser.add_argument("--env-id", default="MiniGrid-GoToDoor-8x8-v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-loops", type=int, default=128)
    parser.add_argument("--render-mode", choices=["none", "human", "rgb_array"], default="none")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-prewarm", action="store_true")
    parser.add_argument("--print-cache", action="store_true")
    args = parser.parse_args(argv)

    result = run_episode(
        instruction=args.instruction,
        compiler_name=args.compiler,
        env_id=args.env_id,
        seed=args.seed,
        max_loops=args.max_loops,
        render_mode=args.render_mode,
        use_cache=not args.no_cache,
        print_cache=args.print_cache,
        prewarm=not args.no_prewarm,
    )

    print("COMPILER BACKEND")
    print(result["compiler_backend"])
    print()
    print("LLM USED")
    print(result["compiler_usage"]["llm_used"])
    print()
    print("COMPILER USAGE")
    pprint(result["compiler_usage"])
    print()
    if result["compiler_logs"]:
        print("COMPILER LOGS")
        pprint(result["compiler_logs"])
        print()
    print("TASK")
    pprint(result["task"])
    print()
    print("PROCEDURE CACHE")
    pprint(result["procedure_cache"])
    print()
    print("READINESS")
    pprint(result["readiness"])
    print()
    print("JIT PREWARM")
    pprint(result["jit_prewarm"])
    if result["jit_prewarm_summary"] is not None:
        pprint(result["jit_prewarm_summary"])
    print()
    print("RUNTIME LLM CALLS DURING RENDER")
    pprint(result["runtime_llm_calls_during_render"])
    print()
    print("CACHE MISS DURING RENDER")
    pprint(result["cache_miss_during_render"])
    print()

    for loop_record in result["loop_records"]:
        print(f"LOOP {loop_record['loop']}")
        print("  evidence_needs:")
        pprint(loop_record["evidence_needs"])
        print("  sense_plan:")
        pprint(loop_record["sense_plan"])
        print("  sense_plan_cache:")
        pprint(loop_record["sense_plan_cache"])
        print("  sense_plan_source:")
        pprint(loop_record["sense_plan_source"])
        print("  world_sample:")
        pprint(loop_record["world_sample"])
        print("  operational_evidence:")
        pprint(loop_record["operational_evidence"])
        print("  percepts:")
        pprint(loop_record["percepts"])
        print("  contract:")
        pprint(loop_record["contract"])
        print("  skill_plan:")
        pprint(loop_record["skill_plan"])
        print("  skill_plan_cache:")
        pprint(loop_record["skill_plan_cache"])
        print("  skill_plan_source:")
        pprint(loop_record["skill_plan_source"])
        print("  report:")
        pprint(loop_record["report"])
        print("  action:")
        pprint(loop_record["action"])
        print("  compiler_call_count_so_far:")
        pprint(loop_record["compiler_call_count_so_far"])
        print()

    print("FINAL CLAIMS")
    pprint(result["final_claims"])
    print()
    print("FINAL STATE")
    pprint(result["final_state"])
    print()
    print("PERSISTED KNOWLEDGE")
    pprint(result["persisted_knowledge"])
    print()
    print("EPISODIC MEMORY")
    pprint(result["episodic_memory"])
    print()
    print("LAST WORLD SAMPLE")
    pprint(result["last_world_sample"])
    print()
    print("TRACE EVENTS")
    pprint(result["trace_events"])
    print()
    print("PLAN CACHE")
    pprint(result["plan_cache"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
