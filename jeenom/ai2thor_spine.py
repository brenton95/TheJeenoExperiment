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

# AI2-THOR yaw → JEENO floor-plane (jx, jy) movement vectors.
# yaw 0 → +z (AI2-THOR) → +jy; yaw 90 → +x → +jx; etc.
_YAW_TO_FLOOR_VEC: dict[int, tuple[int, int]] = {
    0: (0, 1),
    90: (1, 0),
    180: (0, -1),
    270: (-1, 0),
}


def _derive_grid_size(raw_points: list[dict[str, Any]]) -> float:
    """Derive the grid step from GetReachablePositions raw float coords.

    Returns the minimum nonzero pairwise distance — this *is* the substrate's
    actual grid spacing. O(n²) but cached and only called once per episode.
    """
    if len(raw_points) < 2:
        return 0.25  # fallback for degenerate/single-point sets

    coords = [(float(p["x"]), float(p["z"])) for p in raw_points]
    min_dist = float("inf")
    eps = 1e-6
    # Check neighbors — BFS-connected grids have neighbors in the first N points
    for i, (ax, az) in enumerate(coords):
        for bx, bz in coords[i + 1 :]:
            d = ((ax - bx) ** 2 + (az - bz) ** 2) ** 0.5
            if d > eps and d < min_dist:
                min_dist = d
    if min_dist == float("inf"):
        return 0.25
    return min_dist


class Ai2thorSpine:
    """Motor dispatch + closed-loop navigation for AI2-THOR.

    Motor dispatch (plan 003) maps single JEENO primitives to controller.step()
    calls. Navigation (plan 007) plans a floor-plane path via BFS over
    GetReachablePositions, executes actions one-at-a-time checking
    lastActionSuccess, and re-plans once on a blocked move. Grid spacing is
    derived from the substrate; rotation step is injected config.

    Controller is injected, enabling mock-based testing without Unity.
    """

    def __init__(
        self,
        memory: Any,
        controller: Any,
        compiler: Any,
        plan_cache: Any = None,
        rotate_step: int = 90,
    ) -> None:
        self.memory = memory
        self.controller = controller
        self.compiler = compiler
        self.plan_cache = plan_cache
        self.rotate_step = rotate_step
        self.active_skill: str | None = None
        self._reachable_set: set[tuple[int, int]] | None = None
        self._grid_size: float | None = None

    @property
    def grid_size(self) -> float:
        if self._grid_size is None:
            self._get_reachable_positions()
        assert self._grid_size is not None
        return self._grid_size

    @property
    def reach_threshold(self) -> float:
        return self.grid_size * 1.5

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
            report = self._motor_dispatch(skill)
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

    # ── Navigation (plan 007 — closed-loop) ─────────────────────────────

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

        agent_q = self._quantize_point(*agent_floor)
        target_q = self._quantize_point(*target_floor)

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

        max_actions = max(4 * len(path), 20)
        total_executed: list[str] = []

        for action_name in actions:
            if len(total_executed) >= max_actions:
                return ExecutionReport(
                    status="failed",
                    reason="navigation_step_budget_exceeded",
                    progress={
                        "contract": execution_contract.skill,
                        "total_executed": total_executed,
                        "budget": max_actions,
                    },
                    source="spine",
                )

            success, event_meta = self._execute_env_action_raw(action_name)
            total_executed.append(action_name)

            if not success and action_name == "move_forward":
                # Re-read actual pose from metadata and re-plan once
                actual_pos = event_meta.get("agent", {}).get("position", {})
                actual_rot = event_meta.get("agent", {}).get("rotation", {})
                actual_jx = geometry.as_coord(actual_pos.get("x", 0.0))
                actual_jy = geometry.as_coord(actual_pos.get("z", 0.0))
                actual_yaw = int(actual_rot.get("y", 0)) % 360

                new_q = self._quantize_point(float(actual_jx), float(actual_jy))
                # Exclude the cell that just blocked us
                blocked_jx, blocked_jy = self._blocked_cell_ahead(
                    float(actual_jx), float(actual_jy), actual_yaw,
                )
                blocked_q = self._quantize_point(blocked_jx, blocked_jy)
                replan_reachable = reachable - {blocked_q}
                new_path = self._bfs_path(new_q, goals, replan_reachable)
                if not new_path or len(new_path) < 2:
                    return ExecutionReport(
                        status="failed",
                        reason="navigation_blocked",
                        progress={
                            "contract": execution_contract.skill,
                            "total_executed": total_executed,
                            "replan_failed": True,
                        },
                        source="spine",
                    )
                new_actions = self._path_to_actions(new_path, actual_yaw)
                return self._execute_remaining(
                    new_actions, execution_contract,
                    target_floor, total_executed, max_actions,
                )

        return self._postcondition_check(
            execution_contract, target_floor, total_executed, actions,
        )

    def _execute_remaining(
        self,
        actions: list[str],
        execution_contract: Any,
        target_floor: tuple[float, float],
        total_executed: list[str],
        max_actions: int,
    ) -> ExecutionReport:
        """Execute a re-planned action list. No further re-planning allowed."""
        for action_name in actions:
            if len(total_executed) >= max_actions:
                return ExecutionReport(
                    status="failed",
                    reason="navigation_step_budget_exceeded",
                    progress={
                        "contract": execution_contract.skill,
                        "total_executed": total_executed,
                        "budget": max_actions,
                    },
                    source="spine",
                )

            success, _ = self._execute_env_action_raw(action_name)
            total_executed.append(action_name)

            if not success and action_name == "move_forward":
                return ExecutionReport(
                    status="failed",
                    reason="navigation_blocked",
                    progress={
                        "contract": execution_contract.skill,
                        "total_executed": total_executed,
                        "blocked_after_replan": True,
                    },
                    source="spine",
                )

        return self._postcondition_check(
            execution_contract, target_floor, total_executed, actions,
        )

    def _postcondition_check(
        self,
        execution_contract: Any,
        target_floor: tuple[float, float],
        total_executed: list[str],
        actions: list[str],
    ) -> ExecutionReport:
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
                    "actions": total_executed,
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
                "actions": total_executed,
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

        self._grid_size = _derive_grid_size(raw_points)

        self._reachable_set = {
            self._quantize_point(float(p["x"]), float(p["z"]))
            for p in raw_points
        }
        return self._reachable_set

    def _quantize(self, value: float) -> int:
        return round(value / self.grid_size)

    def _quantize_point(self, jx: float, jy: float) -> tuple[int, int]:
        return (self._quantize(jx), self._quantize(jy))

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
        if diff == self.rotate_step:
            return ["turn_right"]
        if diff == 2 * self.rotate_step:
            return ["turn_right", "turn_right"]
        if diff == 3 * self.rotate_step:
            return ["turn_left"]
        return []

    def _blocked_cell_ahead(
        self, jx: float, jy: float, yaw: int,
    ) -> tuple[float, float]:
        """Return the JEENO floor coord of the cell directly ahead of (jx, jy)."""
        vec = _YAW_TO_FLOOR_VEC.get(yaw % 360, (0, 0))
        return (jx + vec[0] * self.grid_size, jy + vec[1] * self.grid_size)

    def _is_adjacent(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        return geometry.euclidean(a, b) <= self.reach_threshold

    # ── Motor dispatch (plan 003) ────────────────────────────────────────

    def _execute_env_action_raw(self, name: str) -> tuple[bool, dict[str, Any]]:
        """Execute an action and return (lastActionSuccess, event_metadata)."""
        action_str = AI2THOR_ACTIONS.get(name)
        if action_str is None:
            return False, {}

        event = self.controller.step(action=action_str, renderImage=False)
        meta = event.metadata if hasattr(event, "metadata") else {}
        success = meta.get("lastActionSuccess", True)
        return success, meta

    def _motor_dispatch(self, name: str) -> ExecutionReport:
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
