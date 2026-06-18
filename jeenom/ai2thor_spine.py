from __future__ import annotations

from collections import deque
from typing import Any

from . import geometry
from .schemas import ExecutionContext, ExecutionReport


AI2THOR_ACTIONS: dict[str, str] = {
    "move_forward": "MoveAhead",
    "turn_right": "RotateRight",
    "turn_left": "RotateLeft",
}

GRID_SIZE: float = 0.25
ROTATE_STEP: int = 90
REACH_THRESHOLD: float = GRID_SIZE * 1.5

# AI2-THOR yaw → JEENO floor-plane (jx, jy) movement vectors.
# yaw 0 → +z (AI2-THOR) → +jy; yaw 90 → +x → +jx; etc.
_YAW_TO_FLOOR_VEC: dict[int, tuple[int, int]] = {
    0: (0, 1),
    90: (1, 0),
    180: (0, -1),
    270: (-1, 0),
}


def _quantize(value: float) -> int:
    return round(value / GRID_SIZE)


def _quantize_point(jx: float, jy: float) -> tuple[int, int]:
    return (_quantize(jx), _quantize(jy))


class Ai2thorSpine:
    """Motor dispatch + navigation for AI2-THOR.

    Motor dispatch (plan 003) maps single JEENO primitives to controller.step()
    calls. Navigation (plan 005) plans a floor-plane path from agent to target
    via BFS over GetReachablePositions, converts to action sequences, and detects
    task success via a postcondition proximity check (resolves F3).

    Controller is injected, enabling mock-based testing without Unity.
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
        self._reachable_set: set[tuple[int, int]] | None = None

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
        elif skill == "navigate_to_object":
            report = self._navigate(execution_contract, percepts)
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

    # ── Navigation (plan 005) ────────────────────────────────────────────

    def _navigate(
        self,
        execution_contract: Any,
        percepts: Any,
    ) -> ExecutionReport:
        if percepts is None:
            return ExecutionReport(
                status="failed",
                reason="no_percepts",
                progress={"contract": execution_contract.skill},
                source="spine",
            )

        agent_pose = percepts.cues.get("agent_pose")
        target_location = percepts.cues.get("target_location")
        if agent_pose is None or target_location is None:
            return ExecutionReport(
                status="failed",
                reason="missing_pose_or_target",
                progress={"contract": execution_contract.skill},
                source="spine",
            )

        reachable = self._get_reachable_positions()
        agent_floor = (float(agent_pose["x"]), float(agent_pose["y"]))
        target_floor = (float(target_location[0]), float(target_location[1]))

        # Already adjacent?
        if self._is_adjacent(agent_floor, target_floor):
            return ExecutionReport(
                status="succeeded",
                progress={
                    "contract": execution_contract.skill,
                    "already_adjacent": True,
                    "agent_floor": agent_floor,
                    "target_floor": target_floor,
                },
                source="spine",
            )

        agent_q = _quantize_point(*agent_floor)
        target_q = _quantize_point(*target_floor)

        goals = self._navigation_goals(target_q, reachable)
        if not goals:
            return ExecutionReport(
                status="failed",
                reason="no_reachable_goal_adjacent_to_target",
                progress={
                    "contract": execution_contract.skill,
                    "target_quantized": target_q,
                },
                source="spine",
            )

        path = self._bfs_path(agent_q, goals, reachable)
        if not path:
            return ExecutionReport(
                status="failed",
                reason="no_path_found",
                progress={
                    "contract": execution_contract.skill,
                    "agent_quantized": agent_q,
                    "goals": list(goals),
                },
                source="spine",
            )

        if len(path) == 1:
            return ExecutionReport(
                status="succeeded",
                progress={
                    "contract": execution_contract.skill,
                    "already_at_goal": True,
                    "path": path,
                },
                source="spine",
            )

        agent_yaw = int(agent_pose.get("dir", 0)) % 360
        actions = self._path_to_actions(path, agent_yaw)

        for action_name in actions:
            self._execute_env_action(action_name)

        # Postcondition: check proximity after executing the full path.
        final_event = self.controller.step(action="Done", renderImage=False)
        final_meta = final_event.metadata if hasattr(final_event, "metadata") else {}
        final_agent = final_meta.get("agent", {}).get("position", {})
        final_jx = geometry.as_coord(final_agent.get("x", 0.0))
        final_jy = geometry.as_coord(final_agent.get("z", 0.0))
        final_floor = (float(final_jx), float(final_jy))

        if self._is_adjacent(final_floor, target_floor):
            return ExecutionReport(
                status="succeeded",
                progress={
                    "contract": execution_contract.skill,
                    "path": path,
                    "actions": actions,
                    "final_floor": final_floor,
                    "target_floor": target_floor,
                },
                source="spine",
            )

        return ExecutionReport(
            status="failed",
            reason="postcondition_not_met",
            progress={
                "contract": execution_contract.skill,
                "path": path,
                "actions": actions,
                "final_floor": final_floor,
                "target_floor": target_floor,
                "distance": geometry.euclidean(final_floor, target_floor),
            },
            source="spine",
        )

    def _get_reachable_positions(self) -> set[tuple[int, int]]:
        if self._reachable_set is not None:
            return self._reachable_set

        event = self.controller.step(action="GetReachablePositions", renderImage=False)
        raw_points = event.metadata.get("actionReturn", [])
        # Project AI2-THOR {x, y, z} → JEENO floor (jx, jy) = (AThor.x, AThor.z), then quantize.
        self._reachable_set = {
            _quantize_point(float(p["x"]), float(p["z"]))
            for p in raw_points
        }
        return self._reachable_set

    def _navigation_goals(
        self,
        target_q: tuple[int, int],
        reachable: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        tx, ty = target_q
        goals = []
        for dx, dy in _YAW_TO_FLOOR_VEC.values():
            neighbor = (tx + dx, ty + dy)
            if neighbor in reachable:
                goals.append(neighbor)
        return goals

    def _bfs_path(
        self,
        start: tuple[int, int],
        goals: list[tuple[int, int]],
        reachable: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        goal_set = set(goals)
        if not goal_set:
            return []
        if start in goal_set:
            return [start]

        frontier: deque[tuple[int, int]] = deque([start])
        parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

        while frontier:
            current = frontier.popleft()
            if current in goal_set:
                path: list[tuple[int, int]] = []
                node: tuple[int, int] | None = current
                while node is not None:
                    path.append(node)
                    node = parents[node]
                path.reverse()
                return path

            cx, cy = current
            for dx, dy in _YAW_TO_FLOOR_VEC.values():
                nxt = (cx + dx, cy + dy)
                if nxt in parents or nxt not in reachable:
                    continue
                parents[nxt] = current
                frontier.append(nxt)

        return []

    def _path_to_actions(self, path: list[tuple[int, int]], yaw: int) -> list[str]:
        actions: list[str] = []
        current_yaw = yaw

        for i in range(len(path) - 1):
            cx, cy = path[i]
            nx, ny = path[i + 1]
            move = (nx - cx, ny - cy)

            desired_yaw: int | None = None
            for y, vec in _YAW_TO_FLOOR_VEC.items():
                if vec == move:
                    desired_yaw = y
                    break
            if desired_yaw is None:
                continue

            actions.extend(self._turn_actions(current_yaw, desired_yaw))
            current_yaw = desired_yaw
            actions.append("move_forward")

        return actions

    def _turn_actions(self, current_yaw: int, desired_yaw: int) -> list[str]:
        diff = (desired_yaw - current_yaw) % 360
        if diff == 0:
            return []
        if diff == 90:
            return ["turn_right"]
        if diff == 180:
            return ["turn_right", "turn_right"]
        if diff == 270:
            return ["turn_left"]
        # Non-90° step — shouldn't happen with default rotateStepDegrees=90
        return []

    # ── Success detection (resolves F3) ──────────────────────────────────

    @staticmethod
    def _is_adjacent(a: tuple[float, float], b: tuple[float, float]) -> bool:
        return geometry.euclidean(a, b) <= REACH_THRESHOLD

    # ── Motor dispatch (plan 003 — unchanged) ────────────────────────────

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
