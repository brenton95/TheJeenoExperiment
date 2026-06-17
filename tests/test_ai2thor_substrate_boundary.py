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


def test_run_task_episode_and_run_motor_actions_do_not_run_live_episode(
    adapter: Ai2thorSubstrateAdapter,
) -> None:
    episode_result = adapter.run_task_episode(instruction="go to the apple")
    assert episode_result["task_complete"] is False
    assert episode_result["success"] is False

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
    agent_x: float = 0.0,
    agent_z: float = 0.0,
    agent_rot_y: float = 90.0,
) -> dict[str, Any]:
    return {
        "objects": [
            {
                "objectType": "Apple",
                "name": "Apple_1",
                "position": {"x": apple_x, "y": 0.9, "z": apple_z},
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
    assert pose["x"] == 2
    assert pose["y"] == 5
    assert pose["dir"] == 180


def test_sense_no_target_when_type_missing() -> None:
    sense = Ai2thorSense(memory=None, compiler=None)
    obs = _FakeObservation(_apple_metadata())
    ef = EvidenceFrame(needs=[])
    ec = ExecutionContext(active_skill="go_to_object", params={"object_type": "banana"})

    _, _, sample, _, _ = sense.tick(obs, ef, ec)

    assert sample.target_visible is False
    assert sample.target_location is None


@pytest.mark.xfail(
    reason="TODO(F2): SceneObject.x/y are int; float precision requires Steve's coord fix",
    strict=True,
)
def test_sense_coord_preserves_float_precision() -> None:
    """Asserts float coord precision — will xfail until F2 lands (int->float)."""
    sense = Ai2thorSense(memory=None, compiler=None)
    obs = _FakeObservation(_apple_metadata(apple_x=1.5, apple_z=3.7))
    ef = EvidenceFrame(needs=["target_location"])
    ec = ExecutionContext(active_skill="go_to_object", params={"object_type": "apple"})

    _, _, sample, _, _ = sense.tick(obs, ef, ec)

    assert sample.target_location is not None
    x, y = sample.target_location
    assert isinstance(x, float), f"expected float x, got {type(x)}"
    assert abs(x - 1.5) < 0.01
    assert abs(y - 3.7) < 0.01
