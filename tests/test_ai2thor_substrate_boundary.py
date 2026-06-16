from __future__ import annotations

from typing import Any

import pytest

from jeenom.ai2thor_domain_helper import Ai2thorDomainHelper
from jeenom.ai2thor_operational_context import Ai2thorOperationalContext
from jeenom.ai2thor_substrate_adapter import (
    Ai2thorSubstrateAdapter,
    build_ai2thor_runtime_package,
)
from jeenom.capability_registry import CapabilityRegistry
from jeenom.orpi import OrpiManifest
from jeenom.runtime_package import RuntimePackage
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


def test_create_sense_and_create_spine_not_yet_implemented(
    adapter: Ai2thorSubstrateAdapter,
) -> None:
    with pytest.raises(NotImplementedError):
        adapter.create_sense(memory=None, compiler=None, plan_cache=None)  # type: ignore[arg-type]

    with pytest.raises(NotImplementedError):
        adapter.create_spine(memory=None, compiler=None, plan_cache=None)  # type: ignore[arg-type]


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
