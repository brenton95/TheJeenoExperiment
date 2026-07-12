"""Plan 014 (Task 2) — attribute-conditional selection on AI2-THOR.

Proves the kernel can select an object by a property AI2-THOR defines and the
kernel was never built for (state, e.g. open/closed) — carried opaquely inside
the existing `target_ref` bag the same way F13 carries `object_id`. This is a
ONE-TURN flow ("go to the open fridge"): the attribute is known at PARSE TIME
by the adapter's parser, not selected by the kernel from prior grounding — so
`_stamp_target_ref` never fires (no `active_claims.last_grounded_target`) and
the parse-time bag survives untouched. See plan 014 "What this is NOT".

Kernel diff for this capability is EXACTLY the already-committed
`compose_known_task` forward (352a806) — nothing here should require any
further kernel or MiniGrid edit.
"""
from __future__ import annotations

from typing import Any

from jeenom.ai2thor_substrate_adapter import build_ai2thor_runtime_package
from jeenom.operator_station import OperatorStationSession


class _Event:
    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata


def _open_and_closed_fridge_metadata() -> dict[str, Any]:
    """Two description-identical fridges, differing ONLY in isOpen.

    The OPEN (matching) fridge is listed FIRST and the CLOSED (non-matching)
    fridge LAST. The type-only fallback in Ai2thorSense._parse_metadata is
    LAST-WINS (it overwrites target_object on every type match, no break), so
    a scan-order/type-only resolver would wrongly land on the CLOSED fridge —
    this ordering is what makes the test actually bite (a mutation check that
    disables the attribute match must flip the resolved id to Fridge|closed).
    """
    return {
        "objects": [
            {
                "objectType": "Fridge",
                "objectId": "Fridge|open",
                "position": {"x": 5.0, "y": 0.9, "z": 0.0},
                "isOpen": True,
            },
            {
                "objectType": "Fridge",
                "objectId": "Fridge|closed",
                "position": {"x": 2.0, "y": 0.9, "z": 0.0},
                "isOpen": False,
            },
        ],
        "agent": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
        },
    }


class _MockController:
    """Static-scene stub: every .step() returns the same fixed metadata."""

    def __init__(self, metadata: dict[str, Any]) -> None:
        self._metadata = metadata
        self.calls: list[dict[str, Any]] = []

    def step(self, **kwargs: Any) -> _Event:
        self.calls.append(kwargs)
        return _Event(metadata=self._metadata)


def _make_session() -> OperatorStationSession:
    controller = _MockController(_open_and_closed_fridge_metadata())
    package = build_ai2thor_runtime_package(controller=controller, scene_id="FloorPlan1")
    return OperatorStationSession(runtime_package=package, seed=0)


# ── Parser (Step 3): the attribute rides its own group, not the color slot ──


def test_parser_emits_target_ref_attributes_for_open() -> None:
    session = _make_session()
    task = session.compose_known_task("go to the open fridge")
    assert task.params.get("color") is None
    assert task.params.get("object_type") == "fridge"
    assert task.params.get("target_ref") == {"attributes": {"isOpen": True}}


def test_parser_does_not_throw_on_attribute_word_and_does_not_route_to_color() -> None:
    """The old bug: a greedy color group would swallow "open" and hit the
    OPERATOR_COLORS hard enum, raising SchemaValidationError. Must not happen."""
    session = _make_session()
    task = session.compose_known_task("go to the closed fridge")
    assert task.params.get("color") is None
    assert task.params.get("target_ref") == {"attributes": {"isOpen": False}}


# ── Sense (Steps 1-2): filters grid_objects by attribute, one-turn ──────────


def test_sense_emits_non_none_state() -> None:
    """Boundary assertion mirroring the object_id boundary test: Ai2thorSense
    must populate a real state dict, not the old hardcoded None (Step 1)."""
    session = _make_session()
    session.sense.sense_idle_scene(
        session.substrate.controller.step(), env_id="FloorPlan1", seed=0
    )
    scene = session.memory.scene_model
    states = [o.state for o in scene.objects if o.object_type == "fridge"]
    assert len(states) == 2
    assert None not in states
    assert {"isOpen": False} in states
    assert {"isOpen": True} in states


def test_one_turn_attribute_select_resolves_by_state_not_scan_order() -> None:
    """The acceptance test: the OPEN fridge is second in scan order and the
    CLOSED fridge is first. A scan-order/type-only resolver would pick the
    closed one first; the attribute-aware resolver must pick the open one."""
    from jeenom.schemas import EvidenceFrame, ExecutionContext

    session = _make_session()
    sense = session.sense
    observation = _Event(_open_and_closed_fridge_metadata())
    evidence_frame = EvidenceFrame(needs=["target_location"])
    execution_context = ExecutionContext(
        active_skill="locate_object",
        params={
            "object_type": "fridge",
            "color": None,
            "target_ref": {"attributes": {"isOpen": True}},
        },
    )

    _, _, sample, _, _ = sense.tick(observation, evidence_frame, execution_context)

    assert sample.target_visible is True
    assert sample.target_object is not None
    assert sample.target_object["object_id"] == "Fridge|open", (
        "sense must resolve BY the attribute, not scan order "
        "(last-wins type-only fallback would pick the closed fridge, listed last)"
    )
    assert sample.target_location == (5.0, 0.0)


class _NavEvent:
    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata


def _nav_fridges() -> list[dict[str, Any]]:
    """Open (matching) fridge FIRST in scan order, closed (non-matching) LAST —
    both reachable. Ai2thorSense's type-only fallback is LAST-WINS, so this
    ordering makes a scan-order-blind resolver land on the WRONG (closed)
    fridge; only the attribute-aware pass correctly picks the open one."""
    return [
        {"objectType": "Fridge", "objectId": "Fridge|open",
         "position": {"x": 0.25, "y": 0.9, "z": 0.0}, "isOpen": True},
        {"objectType": "Fridge", "objectId": "Fridge|closed",
         "position": {"x": 0.75, "y": 0.9, "z": 0.0}, "isOpen": False},
    ]


class _NavMockController:
    """Navigation-capable mock (mirrors test_ai2thor_target_ref_population.py)
    so the real episode reaches the sense/locate step instead of failing
    navigation before sense ever runs."""

    def __init__(self) -> None:
        self.ax = 0.0
        self.az = 0.0
        self.yaw = 0.0
        self.calls: list[dict[str, Any]] = []

    def step(self, **kwargs: Any) -> _NavEvent:
        self.calls.append(kwargs)
        action = kwargs.get("action", "")
        if action == "GetReachablePositions":
            pts = [{"x": x * 0.25, "y": 0.0, "z": z * 0.25}
                   for x in range(5) for z in range(5)]
            return _NavEvent({"actionReturn": pts})
        if action == "MoveAhead":
            y = int(self.yaw) % 360
            if y == 0:
                self.az = round(self.az + 0.25, 6)
            elif y == 90:
                self.ax = round(self.ax + 0.25, 6)
            elif y == 180:
                self.az = round(self.az - 0.25, 6)
            elif y == 270:
                self.ax = round(self.ax - 0.25, 6)
        elif action == "RotateRight":
            self.yaw = (self.yaw + 90) % 360
        elif action == "RotateLeft":
            self.yaw = (self.yaw - 90) % 360
        return _NavEvent({
            "lastActionSuccess": True,
            "objects": _nav_fridges(),
            "agent": {
                "position": {"x": self.ax, "y": 0.0, "z": self.az},
                "rotation": {"x": 0.0, "y": self.yaw, "z": 0.0},
            },
        })


def test_one_turn_attribute_select_through_real_front_door(monkeypatch) -> None:
    """LOAD-BEARING: the parse-time attribute actually reaches Ai2thorSense
    through the REAL one-turn dispatch (ticket -> cortex -> locate step ->
    execution_context.params), and the LIVE sense resolves the open fridge —
    NOT the closed one, which the last-wins type-only fallback would pick
    (it is listed last in scan order). No prior grounding turn (one-turn
    scope; _stamp_target_ref never fires here)."""
    import jeenom.ai2thor_sense as ai2thor_sense_module

    controller = _NavMockController()
    package = build_ai2thor_runtime_package(controller=controller, scene_id="FloorPlan1")
    session = OperatorStationSession(runtime_package=package, seed=0)

    captures: list[dict[str, Any]] = []
    original_tick = ai2thor_sense_module.Ai2thorSense.tick

    def spy_tick(self, observation, evidence_frame, execution_context, *args, **kwargs):
        ctx_params = getattr(execution_context, "params", {}) or {}
        delivered_ref = ctx_params.get("target_ref")
        result = original_tick(
            self, observation, evidence_frame, execution_context, *args, **kwargs
        )
        sample = result[2]
        captures.append({
            "delivered_target_ref": delivered_ref,
            "resolved_object_id": (
                sample.target_object.get("object_id")
                if sample.target_object else None
            ),
        })
        return result

    monkeypatch.setattr(ai2thor_sense_module.Ai2thorSense, "tick", spy_tick)

    result = session.handle_utterance("go to the open fridge")

    refs_delivered = [
        c for c in captures if isinstance(c["delivered_target_ref"], dict)
    ]
    assert refs_delivered, "target_ref never reached the live Ai2thorSense"
    assert refs_delivered[0]["delivered_target_ref"] == {"attributes": {"isOpen": True}}

    resolved = [
        c["resolved_object_id"] for c in captures if c["resolved_object_id"] is not None
    ]
    assert "Fridge|open" in resolved
    assert "Fridge|closed" not in resolved, (
        "sense resolved the closed (scan-order-last-wins) fridge — attribute filter failed"
    )
    last_result = result.result["last_result"]
    assert last_result["final_state"]["task_complete"] is True
    assert last_result["runtime_llm_calls_during_render"] == 0
    assert last_result["cache_miss_during_render"] == 0


# ── Kernel-diff boundary (documents the scope line, does not touch kernel) ──


def test_stamp_target_ref_does_not_fire_without_prior_grounding() -> None:
    """One-turn scope: no active_claims.last_grounded_target exists yet, so
    _stamp_target_ref must early-return and NOT overwrite the parse-time
    attribute bag. Combined identity+attribute (stamp overwriting the bag) is
    documented out-of-scope in plan 014 SCOPE LINE, not built here."""
    session = _make_session()
    task = session.compose_known_task("go to the open fridge")
    before = dict(task.params.get("target_ref") or {})
    session._stamp_target_ref(task.params)
    after = task.params.get("target_ref")
    assert after == before, "stamp must no-op (no prior grounding) and preserve the parse-time bag"
