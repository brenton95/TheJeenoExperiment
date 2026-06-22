from __future__ import annotations

from typing import Any

import pytest

from jeenom.ai2thor_domain_helper import Ai2thorDomainHelper
from jeenom.ai2thor_operational_context import Ai2thorOperationalContext
from jeenom.ai2thor_sense import Ai2thorSense
from jeenom.ai2thor_spine import AI2THOR_ACTIONS, Ai2thorSpine
from jeenom.ai2thor_substrate_adapter import (
    Ai2thorSubstrateAdapter,
    build_ai2thor_runtime_package,
)
from jeenom.capability_registry import CapabilityRegistry
from jeenom.orpi import OrpiManifest
from jeenom.runtime_package import RuntimePackage
from jeenom.schemas import (
    EvidenceFrame,
    ExecutionContext,
    ExecutionContract,
    ExecutionReport,
    OperationalEvidence,
    Percepts,
    WorldModelSample,
)
from jeenom.substrate_adapter import SubstrateAdapter


class _FakeEvent:
    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata


class _MockController:
    """Stub exposing .step(**kwargs) -> object with .metadata, no Unity needed."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def step(self, **kwargs: Any) -> _FakeEvent:
        self.calls.append(kwargs)
        return _FakeEvent(metadata={"objects": [], "agent": {"position": {"x": 0, "z": 0}}})


@pytest.fixture
def adapter() -> Ai2thorSubstrateAdapter:
    return Ai2thorSubstrateAdapter(controller=_MockController(), scene_id="FloorPlan1")


def test_adapter_satisfies_substrate_adapter_protocol(adapter: Ai2thorSubstrateAdapter) -> None:
    protocol_methods = [
        name
        for name in vars(SubstrateAdapter)
        if not name.startswith("_")
    ]
    assert protocol_methods, "expected SubstrateAdapter Protocol to define methods"
    for method_name in protocol_methods:
        assert hasattr(adapter, method_name), f"adapter missing protocol method: {method_name}"
        assert callable(getattr(adapter, method_name))


def test_controller_is_injected_not_constructed() -> None:
    bare = Ai2thorSubstrateAdapter()
    assert bare.controller is None

    mock = _MockController()
    wired = Ai2thorSubstrateAdapter(controller=mock)
    assert wired.controller is mock


def test_capability_registry_and_orpi_manifest(adapter: Ai2thorSubstrateAdapter) -> None:
    registry = adapter.capability_registry()
    assert isinstance(registry, CapabilityRegistry)

    manifest = adapter.orpi_manifest()
    assert isinstance(manifest, OrpiManifest)


def test_known_action_names_and_is_action_known(adapter: Ai2thorSubstrateAdapter) -> None:
    known = adapter.known_action_names()
    assert isinstance(known, list)
    for name in known:
        assert adapter.is_action_known(name)
    assert not adapter.is_action_known("definitely_not_a_real_action")


def test_adapter_wires_into_valid_runtime_package() -> None:
    operational_context = Ai2thorOperationalContext.default(scene_id="FloorPlan1")
    adapter = Ai2thorSubstrateAdapter(
        controller=_MockController(),
        scene_id="FloorPlan1",
        operational_context=operational_context,
    )
    domain_helper = Ai2thorDomainHelper(operational_context=operational_context)

    package = RuntimePackage(
        substrate=adapter,
        operational_context=operational_context,
        domain_helper=domain_helper,
    )

    assert package.domain_helper.operational_context is package.operational_context

    registry = package.resolve_capability_registry()
    assert isinstance(registry, CapabilityRegistry)

    manifest = package.resolve_orpi_manifest()
    assert isinstance(manifest, OrpiManifest)


def test_build_ai2thor_runtime_package_convenience_helper() -> None:
    mock = _MockController()
    package = build_ai2thor_runtime_package(controller=mock, scene_id="FloorPlan1")
    assert isinstance(package, RuntimePackage)
    assert package.substrate.controller is mock
    assert package.domain_helper.operational_context is package.operational_context


def test_capability_registry_has_apple_go_to_object_handle(adapter: Ai2thorSubstrateAdapter) -> None:
    registry = adapter.capability_registry()
    spec = registry._by_name.get("task.go_to_object.apple")
    assert spec is not None
    assert spec.layer == "task"


def test_run_motor_actions_stub_does_not_run_live_episode(
    adapter: Ai2thorSubstrateAdapter,
) -> None:
    motor_result = adapter.run_motor_actions(seed=0, actions=["move_forward"])
    assert motor_result["task_complete"] is False
    assert motor_result["success"] is False


def test_create_sense_returns_ai2thor_sense(
    adapter: Ai2thorSubstrateAdapter,
) -> None:
    sense = adapter.create_sense(memory=None, compiler=None, plan_cache=None)  # type: ignore[arg-type]
    assert isinstance(sense, Ai2thorSense)


def test_create_spine_returns_ai2thor_spine(
    adapter: Ai2thorSubstrateAdapter,
) -> None:
    spine = adapter.create_spine(memory=None, compiler=None, plan_cache=None)  # type: ignore[arg-type]
    assert isinstance(spine, Ai2thorSpine)
    assert spine.controller is adapter.controller


def test_preview_and_task_window_lifecycle(adapter: Ai2thorSubstrateAdapter) -> None:
    assert adapter.has_preview_window() is False
    adapter.open_preview(seed=0)
    assert adapter.has_preview_window() is True
    adapter.close_preview()
    assert adapter.has_preview_window() is False

    assert adapter.has_task_window() is False
    adapter.close_task_window()
    assert adapter.has_task_window() is False

    adapter.close()
    assert adapter.has_preview_window() is False
    assert adapter.has_task_window() is False


def test_mock_controller_step_returns_metadata() -> None:
    mock = _MockController()
    event = mock.step(action="Done")
    assert hasattr(event, "metadata")
    assert "objects" in event.metadata
    assert mock.calls == [{"action": "Done"}]


def test_known_action_names_is_non_empty_and_all_pass_is_action_known(
    adapter: Ai2thorSubstrateAdapter,
) -> None:
    known = adapter.known_action_names()
    assert len(known) > 0, "known_action_names must be non-empty now that spine is wired"
    for name in known:
        assert adapter.is_action_known(name), f"{name} should be known"


def test_spine_motor_dispatch_move_forward() -> None:
    mock = _MockController()
    spine = Ai2thorSpine(memory=None, controller=mock, compiler=None)
    contract = ExecutionContract(skill="move_forward")

    report, context, plan, cache_meta = spine.tick(contract)

    assert isinstance(report, ExecutionReport)
    assert len(mock.calls) == 1
    assert mock.calls[0]["action"] == "MoveAhead"
    assert mock.calls[0]["renderImage"] is False


def test_spine_motor_dispatch_turn_right() -> None:
    mock = _MockController()
    spine = Ai2thorSpine(memory=None, controller=mock, compiler=None)
    contract = ExecutionContract(skill="turn_right")

    report, context, plan, cache_meta = spine.tick(contract)

    assert len(mock.calls) == 1
    assert mock.calls[0]["action"] == "RotateRight"
    assert mock.calls[0]["renderImage"] is False


def test_spine_motor_dispatch_turn_left() -> None:
    mock = _MockController()
    spine = Ai2thorSpine(memory=None, controller=mock, compiler=None)
    contract = ExecutionContract(skill="turn_left")

    report, context, plan, cache_meta = spine.tick(contract)

    assert len(mock.calls) == 1
    assert mock.calls[0]["action"] == "RotateLeft"
    assert mock.calls[0]["renderImage"] is False


def test_spine_unknown_skill_fails() -> None:
    mock = _MockController()
    spine = Ai2thorSpine(memory=None, controller=mock, compiler=None)
    contract = ExecutionContract(skill="teleport_to_moon")

    report, context, plan, cache_meta = spine.tick(contract)

    assert report.status == "failed"
    assert "unknown_action_primitive" in (report.reason or "")
    assert len(mock.calls) == 0


def test_spine_tick_returns_4_tuple() -> None:
    mock = _MockController()
    spine = Ai2thorSpine(memory=None, controller=mock, compiler=None)
    contract = ExecutionContract(skill="move_forward")

    result = spine.tick(contract)

    assert len(result) == 4
    report, context, plan, cache_meta = result
    assert isinstance(report, ExecutionReport)
    assert isinstance(context, ExecutionContext)
    assert isinstance(plan, list)
    assert isinstance(cache_meta, dict)


# ── Sense tests ──────────────────────────────────────────────────────────


class _FakeObservation:
    """Mimics AI2-THOR event with .metadata for sense tests."""

    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata


def _apple_metadata(
    apple_x: float = 1.5,
    apple_z: float = 3.7,
    apple_y: float = 0.9,
    agent_x: float = 0.0,
    agent_z: float = 0.0,
    agent_rot_y: float = 90.0,
) -> dict[str, Any]:
    return {
        "objects": [
            {
                "objectType": "Apple",
                "name": "Apple_1",
                "position": {"x": apple_x, "y": apple_y, "z": apple_z},
            },
        ],
        "agent": {
            "position": {"x": agent_x, "y": 0.0, "z": agent_z},
            "rotation": {"x": 0.0, "y": agent_rot_y, "z": 0.0},
        },
    }


def test_sense_tick_returns_5_tuple() -> None:
    sense = Ai2thorSense(memory=None, compiler=None)
    obs = _FakeObservation(_apple_metadata())
    ef = EvidenceFrame(needs=["target_location"])
    ec = ExecutionContext(active_skill="go_to_object", params={"object_type": "apple"})

    result = sense.tick(obs, ef, ec)

    assert len(result) == 5
    evidence, percepts, sample, plan, cache_meta = result
    assert isinstance(evidence, OperationalEvidence)
    assert isinstance(percepts, Percepts)
    assert isinstance(sample, WorldModelSample)
    assert isinstance(plan, list)
    assert isinstance(cache_meta, dict)
    assert cache_meta["runtime_compiler_call"] is False


def test_sense_parses_apple_from_metadata() -> None:
    sense = Ai2thorSense(memory=None, compiler=None)
    obs = _FakeObservation(_apple_metadata(apple_x=1.5, apple_z=3.7))
    ef = EvidenceFrame(needs=["target_location"])
    ec = ExecutionContext(active_skill="go_to_object", params={"object_type": "apple"})

    evidence, percepts, sample, plan, cache_meta = sense.tick(obs, ef, ec)

    assert sample.target_visible is True
    assert sample.target_object is not None
    assert sample.target_object["type"] == "apple"
    assert sample.target_location is not None


def test_sense_project_to_spine_cues() -> None:
    sense = Ai2thorSense(memory=None, compiler=None)
    obs = _FakeObservation(_apple_metadata())
    ef = EvidenceFrame(needs=["target_location"])
    ec = ExecutionContext(active_skill="go_to_object", params={"object_type": "apple"})

    _, percepts, _, _, _ = sense.tick(obs, ef, ec)

    assert "agent_pose" in percepts.cues
    assert "target_location" in percepts.cues
    assert percepts.cues["agent_pose"] is not None


def test_sense_agent_pose_from_metadata() -> None:
    sense = Ai2thorSense(memory=None, compiler=None)
    obs = _FakeObservation(_apple_metadata(agent_x=2.0, agent_z=5.0, agent_rot_y=180.0))
    ef = EvidenceFrame(needs=[])
    ec = ExecutionContext(active_skill="go_to_object", params={})

    _, percepts, sample, _, _ = sense.tick(obs, ef, ec)

    pose = sample.agent_pose
    assert pose is not None
    assert pose["x"] == 2.0
    assert pose["y"] == 5.0
    assert pose["z"] == 0.0
    assert pose["dir"] == 180


def test_sense_no_target_when_type_missing() -> None:
    sense = Ai2thorSense(memory=None, compiler=None)
    obs = _FakeObservation(_apple_metadata())
    ef = EvidenceFrame(needs=[])
    ec = ExecutionContext(active_skill="go_to_object", params={"object_type": "banana"})

    _, _, sample, _, _ = sense.tick(obs, ef, ec)

    assert sample.target_visible is False
    assert sample.target_location is None


def test_sense_coord_preserves_float_precision() -> None:
    """Three distinct AI2-THOR axes -> correct JEENO fields, float-preserved."""
    sense = Ai2thorSense(memory=None, compiler=None)
    obs = _FakeObservation(_apple_metadata(apple_x=1.5, apple_z=3.7, apple_y=0.9))
    ef = EvidenceFrame(needs=["target_location"])
    ec = ExecutionContext(active_skill="go_to_object", params={"object_type": "apple"})

    _, _, sample, _, _ = sense.tick(obs, ef, ec)

    obj = sample.target_object
    assert obj is not None
    assert isinstance(obj["x"], float), f"expected float x, got {type(obj['x'])}"
    assert abs(obj["x"] - 1.5) < 0.01
    assert abs(obj["y"] - 3.7) < 0.01
    assert abs(obj["z"] - 0.9) < 0.01

    assert sample.target_location is not None
    tx, ty = sample.target_location
    assert isinstance(tx, float), f"expected float tx, got {type(tx)}"
    assert abs(tx - 1.5) < 0.01
    assert abs(ty - 3.7) < 0.01
    assert len(sample.target_location) == 2


# ── Navigation tests (plan 005) ─────────────────────────────────────────


def _make_reachable_grid(
    x_range: range,
    z_range: range,
    grid_size: float = 0.25,
) -> list[dict[str, float]]:
    """Build a canned GetReachablePositions return in AI2-THOR coords."""
    return [
        {"x": x * grid_size, "y": 0.0, "z": z * grid_size}
        for x in x_range
        for z in z_range
    ]


class _NavMockController:
    """Mock controller that tracks agent position and returns canned reachable positions.

    Supports blocked cells: if a MoveAhead would land on a cell in
    ``blocked_cells``, the agent stays put and ``lastActionSuccess`` is False.
    Grid step is derived from reachable_points (same as the spine does).
    """

    def __init__(
        self,
        reachable_points: list[dict[str, float]],
        agent_x: float = 0.0,
        agent_z: float = 0.0,
        agent_yaw: float = 0.0,
        blocked_cells: set[tuple[float, float]] | None = None,
        grid_size: float = 0.25,
        objects: list[dict[str, Any]] | None = None,
    ) -> None:
        self.reachable_points = reachable_points
        self.agent_x = agent_x
        self.agent_z = agent_z
        self.agent_yaw = agent_yaw
        self.blocked_cells: set[tuple[float, float]] = blocked_cells or set()
        self.grid_size = grid_size
        self.objects: list[dict[str, Any]] = objects or []
        self.calls: list[dict[str, Any]] = []

    def step(self, **kwargs: Any) -> _FakeEvent:
        self.calls.append(kwargs)
        action = kwargs.get("action", "")

        if action == "GetReachablePositions":
            return _FakeEvent(metadata={"actionReturn": self.reachable_points})

        last_action_success = True

        if action == "MoveAhead":
            yaw = int(self.agent_yaw) % 360
            dx, dz = 0.0, 0.0
            if yaw == 0:
                dz = self.grid_size
            elif yaw == 90:
                dx = self.grid_size
            elif yaw == 180:
                dz = -self.grid_size
            elif yaw == 270:
                dx = -self.grid_size
            new_x = round(self.agent_x + dx, 6)
            new_z = round(self.agent_z + dz, 6)
            if (new_x, new_z) in self.blocked_cells:
                last_action_success = False
            else:
                self.agent_x = new_x
                self.agent_z = new_z
        elif action == "RotateRight":
            self.agent_yaw = (self.agent_yaw + 90) % 360
        elif action == "RotateLeft":
            self.agent_yaw = (self.agent_yaw - 90) % 360

        return _FakeEvent(metadata={
            "lastActionSuccess": last_action_success,
            "objects": self.objects,
            "agent": {
                "position": {"x": self.agent_x, "y": 0.0, "z": self.agent_z},
                "rotation": {"x": 0.0, "y": self.agent_yaw, "z": 0.0},
            },
        })


def test_nav_plans_path_and_succeeds() -> None:
    """Agent at (0,0) facing +jy (yaw=0), apple at (0.5, 0.5).
    Reachable grid 0..3 x 0..3 (quantized). Agent should navigate to a cell
    adjacent to the apple and report succeeded."""
    reachable = _make_reachable_grid(range(4), range(4))
    ctrl = _NavMockController(reachable, agent_x=0.0, agent_z=0.0, agent_yaw=0.0)
    spine = Ai2thorSpine(memory=None, controller=ctrl, compiler=None)

    # target at JEENO (0.5, 0.5) — quantizes to (2, 2) with gridSize=0.25
    percepts = Percepts(cues={
        "agent_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "dir": 0},
        "target_location": (0.5, 0.5),
    })
    contract = ExecutionContract(skill="navigate_to_object", params={"object_type": "apple"})

    report, context, plan, cache_meta = spine.tick(contract, percepts)

    assert report.status == "succeeded", f"expected succeeded, got {report.status}: {report.reason}"
    assert report.progress.get("actions") is not None
    actions = report.progress["actions"]
    assert all(a in ("move_forward", "turn_right", "turn_left") for a in actions)


def test_nav_success_detection_already_adjacent() -> None:
    """Agent already within reach of target → succeeded immediately, no movement."""
    reachable = _make_reachable_grid(range(4), range(4))
    ctrl = _NavMockController(reachable, agent_x=0.0, agent_z=0.0, agent_yaw=0.0)
    spine = Ai2thorSpine(memory=None, controller=ctrl, compiler=None)

    # agent at (0,0) in JEENO, target at (0.25, 0) — within REACH_THRESHOLD
    percepts = Percepts(cues={
        "agent_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "dir": 0},
        "target_location": (0.25, 0.0),
    })
    contract = ExecutionContract(skill="navigate_to_object")

    report, _, _, _ = spine.tick(contract, percepts)

    assert report.status == "succeeded"
    assert report.progress.get("already_adjacent") is True
    # No movement actions should have been dispatched
    move_calls = [c for c in ctrl.calls if c.get("action") in ("MoveAhead", "RotateRight", "RotateLeft")]
    assert len(move_calls) == 0


def test_nav_unreachable_target_fails_honestly() -> None:
    """Target surrounded by no reachable cells → no_reachable_goal_adjacent_to_target."""
    # Only agent's position is reachable — nothing near the target
    reachable = [{"x": 0.0, "y": 0.0, "z": 0.0}]
    ctrl = _NavMockController(reachable, agent_x=0.0, agent_z=0.0, agent_yaw=0.0)
    spine = Ai2thorSpine(memory=None, controller=ctrl, compiler=None)

    percepts = Percepts(cues={
        "agent_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "dir": 0},
        "target_location": (5.0, 5.0),
    })
    contract = ExecutionContract(skill="navigate_to_object")

    report, _, _, _ = spine.tick(contract, percepts)

    assert report.status == "failed"
    assert "no_reachable_goal" in (report.reason or "")


def test_nav_no_path_found_fails_honestly() -> None:
    """Goal cells exist but are disconnected from agent → no_path_found."""
    # 0.25 grid: agent at (0,0), target at (2.0, 0.0).
    # Only two reachable points with a huge gap between them — derived grid is 2.0.
    # Use a proper grid so derive_grid_size = 0.25, then create a disconnection.
    # Agent row: x=0..0.75, z=0.  Far island: x=2.0, z=0.  Gap at x=1.0..1.75.
    reachable = [
        {"x": i * 0.25, "y": 0.0, "z": 0.0} for i in range(4)  # 0.0..0.75
    ] + [
        {"x": 2.0, "y": 0.0, "z": 0.0},  # isolated island near target
    ]
    ctrl = _NavMockController(reachable, agent_x=0.0, agent_z=0.0, agent_yaw=0.0)
    spine = Ai2thorSpine(memory=None, controller=ctrl, compiler=None)

    # target at JEENO (2.25, 0.0) → quantized (9, 0); adjacent cell (8, 0) reachable but disconnected
    percepts = Percepts(cues={
        "agent_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "dir": 0},
        "target_location": (2.25, 0.0),
    })
    contract = ExecutionContract(skill="navigate_to_object")

    report, _, _, _ = spine.tick(contract, percepts)

    assert report.status == "failed"
    assert report.reason == "no_path_found"


def test_nav_action_sequence_straight_line() -> None:
    """Agent faces +jy, target is directly ahead — should be pure move_forward actions."""
    reachable = _make_reachable_grid(range(1), range(5))  # a column x=0, z=0..4
    ctrl = _NavMockController(reachable, agent_x=0.0, agent_z=0.0, agent_yaw=0.0)
    spine = Ai2thorSpine(memory=None, controller=ctrl, compiler=None)

    # target at JEENO (0.0, 0.75) → quantized (0, 3); agent at (0, 0) facing yaw 0 (+jy)
    # goal = adjacent to (0, 3) = (0, 2) (reachable). path = (0,0)->(0,1)->(0,2).
    percepts = Percepts(cues={
        "agent_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "dir": 0},
        "target_location": (0.0, 0.75),
    })
    contract = ExecutionContract(skill="navigate_to_object")

    report, _, _, _ = spine.tick(contract, percepts)

    assert report.status == "succeeded"
    actions = report.progress["actions"]
    # Already facing the right way, so no turns
    assert all(a == "move_forward" for a in actions)
    assert len(actions) == 2  # two steps: (0,0)->(0,1)->(0,2)


def test_nav_action_sequence_requires_turn() -> None:
    """Agent faces +jy (yaw 0) but target is to the right (+jx) — needs RotateRight first."""
    reachable = _make_reachable_grid(range(5), range(1))  # a row z=0, x=0..4
    ctrl = _NavMockController(reachable, agent_x=0.0, agent_z=0.0, agent_yaw=0.0)
    spine = Ai2thorSpine(memory=None, controller=ctrl, compiler=None)

    # target at JEENO (0.75, 0.0) → quantized (3, 0); agent at (0, 0).
    # goal = adjacent to (3, 0) = (2, 0). path = (0,0)->(1,0)->(2,0).
    # Agent faces yaw 0 (+jy), needs to face yaw 90 (+jx) → one turn_right.
    percepts = Percepts(cues={
        "agent_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "dir": 0},
        "target_location": (0.75, 0.0),
    })
    contract = ExecutionContract(skill="navigate_to_object")

    report, _, _, _ = spine.tick(contract, percepts)

    assert report.status == "succeeded"
    actions = report.progress["actions"]
    assert actions[0] == "turn_right"
    assert actions.count("move_forward") == 2


def test_nav_existing_motor_dispatch_unaffected() -> None:
    """Existing single-action dispatch (move_forward, turn_right, etc.) still works."""
    mock = _MockController()
    spine = Ai2thorSpine(memory=None, controller=mock, compiler=None)

    for skill, expected_action in AI2THOR_ACTIONS.items():
        mock.calls.clear()
        contract = ExecutionContract(skill=skill)
        report, _, _, _ = spine.tick(contract)
        assert report.status == "running"
        assert len(mock.calls) == 1
        assert mock.calls[0]["action"] == expected_action


def test_nav_missing_percepts_fails() -> None:
    """navigate_to_object without percepts → failed, not a crash."""
    ctrl = _NavMockController([], agent_x=0.0, agent_z=0.0)
    spine = Ai2thorSpine(memory=None, controller=ctrl, compiler=None)
    contract = ExecutionContract(skill="navigate_to_object")

    report, _, _, _ = spine.tick(contract, percepts=None)

    assert report.status == "failed"
    assert report.reason == "no_percepts"


# ── Closed-loop navigation tests (plan 007) ────────────────────────────


def test_nav_blocked_move_replans_and_succeeds() -> None:
    """Agent's first path hits a blocked cell after some moves; re-plan from new
    position finds an alternate route and succeeds."""
    # 5x5 grid.  Agent at (0,0) facing yaw 0 (+jy).
    # Target at (0.25, 0.75) — BFS finds path going +jy then +jx.
    # Block (0.0, 0.5) so after the agent moves to (0, 0.25), the next MoveAhead
    # into (0, 0.5) fails.  Re-plan from (0, 0.25) can go +jx first, then +jy
    # to reach a goal adjacent to target.
    reachable = _make_reachable_grid(range(5), range(5))
    blocked = {(0.0, 0.5)}
    ctrl = _NavMockController(
        reachable, agent_x=0.0, agent_z=0.0, agent_yaw=0.0,
        blocked_cells=blocked,
    )
    spine = Ai2thorSpine(memory=None, controller=ctrl, compiler=None)

    percepts = Percepts(cues={
        "agent_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "dir": 0},
        "target_location": (0.25, 0.75),
    })
    contract = ExecutionContract(skill="navigate_to_object", params={"object_type": "apple"})

    report, _, _, _ = spine.tick(contract, percepts)

    assert report.status == "succeeded", f"expected succeeded, got {report.status}: {report.reason}"


def test_nav_genuinely_stuck_fails_honestly() -> None:
    """Target boxed off after a block — spine reports navigation_blocked, no hang."""
    # Tiny grid: agent at (0,0), only path to target goes through (0.25, 0) which is blocked
    reachable = [
        {"x": 0.0, "y": 0.0, "z": 0.0},
        {"x": 0.25, "y": 0.0, "z": 0.0},
        {"x": 0.5, "y": 0.0, "z": 0.0},
    ]
    blocked = {(0.25, 0.0)}  # blocks the only corridor
    ctrl = _NavMockController(
        reachable, agent_x=0.0, agent_z=0.0, agent_yaw=90.0,
        blocked_cells=blocked,
    )
    spine = Ai2thorSpine(memory=None, controller=ctrl, compiler=None)

    # Target at (0.75, 0.0) — adjacent goal is (0.5, 0.0) but path is blocked
    percepts = Percepts(cues={
        "agent_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "dir": 90},
        "target_location": (0.75, 0.0),
    })
    contract = ExecutionContract(skill="navigate_to_object", params={"object_type": "apple"})

    report, _, _, _ = spine.tick(contract, percepts)

    assert report.status == "failed"
    assert report.reason == "navigation_blocked"


def test_nav_step_budget_exceeded() -> None:
    """Re-plan path longer than initial budget → navigation_step_budget_exceeded.

    Budget = max(4 * len(initial_path), 20). We build a grid where the direct
    path is short (budget stays small) but after a block the only detour exceeds it.
    """
    gs = 0.25
    # Two parallel columns connected at the bottom:
    #   col A: x=0, z=0..2   (agent at z=0, target near z=2)
    #   col B: x=0.25, z=0..2
    #   bottom row connecting them: already shared at z=0
    # This gives direct path (0,0)→(0,1)→(0,2) = 3 nodes → budget=max(12,20)=20.
    #
    # For the detour to exceed budget, we need a much longer path.
    # Build: col A x=0 z=0..2, long row at z=0 from x=0 to x=3.0 (12 cells),
    # col B at x=3.0 z=0..2.
    col_a = [{"x": 0.0, "y": 0.0, "z": z * gs} for z in range(3)]
    row = [{"x": x * gs, "y": 0.0, "z": 0.0} for x in range(1, 13)]
    col_b = [{"x": 12 * gs, "y": 0.0, "z": z * gs} for z in range(1, 3)]
    # Connect top: x=12*0.25=3.0 at z=0.5 back to target area
    # Add a bridge at z=0.5 from x=3.0 back to x=0.25 (near target)
    bridge = [{"x": x * gs, "y": 0.0, "z": 2 * gs} for x in range(1, 13)]
    reachable = col_a + row + col_b + bridge

    blocked = {(0.0, gs)}  # block (0, 0.25) — first move
    ctrl = _NavMockController(
        reachable, agent_x=0.0, agent_z=0.0, agent_yaw=0.0,
        blocked_cells=blocked, grid_size=gs,
    )
    spine = Ai2thorSpine(memory=None, controller=ctrl, compiler=None)

    # Target at (0.0, 0.75) → quantized (0, 3). Adjacent: (0,2) in col_a.
    # Direct BFS: (0,0)→(0,1)→(0,2) = 3 nodes, budget=max(12,20)=20.
    # Block at (0,1). Re-plan excluding (0,1): detour east along z=0,
    # up col B, west along z=2 bridge back to (1,2) adjacent to (0,2).
    # Path: (0,0)→(1,0)→...→(12,0)→(12,1)→(12,2)→(11,2)→...→(1,2) = 25 nodes.
    # Actions: turn_right + 11*move_fwd + turn_left + move_fwd + turn_left + 11*move_fwd
    #        = 3 turns + 23 moves = 26 actions. Budget=20 → exceeded.
    percepts = Percepts(cues={
        "agent_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "dir": 0},
        "target_location": (0.0, 0.75),
    })
    contract = ExecutionContract(skill="navigate_to_object", params={"object_type": "apple"})

    report, _, _, _ = spine.tick(contract, percepts)

    assert report.status == "failed"
    assert report.reason == "navigation_step_budget_exceeded"


def test_nav_derived_grid_spacing() -> None:
    """Grid at 0.1 spacing (not 0.25) — proves spacing is derived, not assumed."""
    grid_size = 0.1
    reachable = _make_reachable_grid(range(10), range(10), grid_size=grid_size)
    ctrl = _NavMockController(
        reachable, agent_x=0.0, agent_z=0.0, agent_yaw=0.0,
        grid_size=grid_size,
    )
    spine = Ai2thorSpine(memory=None, controller=ctrl, compiler=None)

    # Target at (0.0, 0.55) in JEENO coords — at 0.1 step, quantized to (0, 5.5)→(0, 6)
    # adjacent goals at (0, 5) etc.  Agent should navigate there.
    percepts = Percepts(cues={
        "agent_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "dir": 0},
        "target_location": (0.0, 0.55),
    })
    contract = ExecutionContract(skill="navigate_to_object", params={"object_type": "apple"})

    report, _, _, _ = spine.tick(contract, percepts)

    assert report.status == "succeeded", f"expected succeeded, got {report.status}: {report.reason}"
    assert spine.grid_size == pytest.approx(grid_size, abs=1e-6)


def test_nav_rotation_step_param() -> None:
    """Constructing Ai2thorSpine(rotate_step=90) — _turn_actions uses the param."""
    reachable = _make_reachable_grid(range(5), range(1))
    ctrl = _NavMockController(reachable, agent_x=0.0, agent_z=0.0, agent_yaw=0.0)
    spine = Ai2thorSpine(memory=None, controller=ctrl, compiler=None, rotate_step=90)

    assert spine.rotate_step == 90

    # Agent faces yaw 0, needs yaw 90 → one turn_right (90° step)
    turns = spine._turn_actions(0, 90)
    assert turns == ["turn_right"]

    # 180° → two turn_rights
    turns = spine._turn_actions(0, 180)
    assert turns == ["turn_right", "turn_right"]

    # 270° → one turn_left
    turns = spine._turn_actions(0, 270)
    assert turns == ["turn_left"]


# ── A1: parse_go_to_object_utterance tests ────────────────────────────

def test_parse_go_to_object_utterance_apple() -> None:
    ctx = Ai2thorOperationalContext.default()
    helper = Ai2thorDomainHelper(operational_context=ctx)
    result = helper.parse_go_to_object_utterance("go to the red apple")
    assert result is not None
    assert result["color"] == "red"
    assert result["object_type"] == "apple"
    assert result["verb"] == "go to"


def test_parse_go_to_object_utterance_no_color() -> None:
    ctx = Ai2thorOperationalContext.default()
    helper = Ai2thorDomainHelper(operational_context=ctx)
    result = helper.parse_go_to_object_utterance("go to the apple")
    assert result is not None
    assert result["color"] == ""
    assert result["object_type"] == "apple"


def test_parse_go_to_object_utterance_non_go_to() -> None:
    ctx = Ai2thorOperationalContext.default()
    helper = Ai2thorDomainHelper(operational_context=ctx)
    assert helper.parse_go_to_object_utterance("hello world") is None
    assert helper.parse_go_to_object_utterance("pick up the red apple") is None


def test_parse_go_to_object_utterance_verb_variants() -> None:
    ctx = Ai2thorOperationalContext.default()
    helper = Ai2thorDomainHelper(operational_context=ctx)
    for verb in ("navigate to", "head to", "reach", "find", "get to"):
        result = helper.parse_go_to_object_utterance(f"{verb} the red apple")
        assert result is not None, f"failed for verb: {verb}"
        assert result["object_type"] == "apple"


# ── A2: run_task_episode end-to-end test ──────────────────────────────

def test_run_task_episode_go_to_red_apple() -> None:
    """Drive run_task_episode through the mock controller with an apple a few
    cells from the agent. Asserts task_complete=True, zero runtime LLM calls.

    Mimics the operator_station call path: compose_known_task +
    compose_known_procedure → pass as task_override/procedure_override."""
    from jeenom.ai2thor_domain_helper import Ai2thorDomainHelper
    from jeenom.ai2thor_operational_context import Ai2thorOperationalContext
    from jeenom.llm_compiler import build_compiler, canonical_task_params
    from jeenom.memory import OperationalMemory
    from jeenom.plan_cache import PlanCache
    from jeenom.schemas import ProcedureRecipe, TaskRequest

    reachable = _make_reachable_grid(range(5), range(5))
    apple_obj = {
        "objectType": "Apple",
        "position": {"x": 0.5, "y": 0.0, "z": 0.5},
    }
    ctrl = _NavMockController(
        reachable,
        agent_x=0.0,
        agent_z=0.0,
        agent_yaw=0.0,
        objects=[apple_obj],
    )

    adapter = Ai2thorSubstrateAdapter(controller=ctrl)
    compiler = build_compiler("smoke_test")
    # Temp memory root: OperationalMemory() defaults to the repo's memory/ dir
    # and would mutate the tracked knowledge.yaml on every test run.
    import tempfile
    from pathlib import Path

    memory = OperationalMemory(root=Path(tempfile.mkdtemp()))
    plan_cache = PlanCache(enabled=True)

    ctx = Ai2thorOperationalContext.default()
    helper = Ai2thorDomainHelper(operational_context=ctx)
    parsed = helper.parse_go_to_object_utterance("go to the red apple")
    assert parsed is not None

    task_override = TaskRequest(
        instruction=f"go to the {parsed['color']} {parsed['object_type']}",
        task_type="go_to_object",
        params=canonical_task_params(
            color=parsed["color"],
            object_type=parsed["object_type"],
        ),
        source="operator_station_known_family",
    )
    procedure_override = ProcedureRecipe(
        task_type="go_to_object",
        steps=["locate_object", "navigate_to_object", "verify_adjacent", "done"],
        source="smoke_test_compiler",
    )

    result = adapter.run_task_episode(
        instruction="go to the red apple",
        compiler_name="smoke_test",
        compiler=compiler,
        seed=42,
        max_loops=128,
        memory=memory,
        plan_cache=plan_cache,
        task_override=task_override,
        procedure_override=procedure_override,
    )

    assert "final_state" in result, f"missing final_state key; keys: {list(result.keys())}"
    assert result["final_state"]["task_complete"] is True, (
        f"task_complete should be True; final_state={result['final_state']}"
    )
    assert result["runtime_llm_calls_during_render"] == 0
    assert result["cache_miss_during_render"] == 0
