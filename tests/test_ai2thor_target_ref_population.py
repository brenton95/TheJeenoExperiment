"""A3 (Plan 013) — VERIFICATION GATE: does the F13 population chain fire on AI2-THOR?

This test drives the real two-turn front door of ``OperatorStationSession`` against
the AI2-THOR substrate with a MOCK controller (no live Unity):

  turn 1: a grounding query ("which apple is closest?") over TWO description-identical
          apples at different distances — records ``active_claims.last_grounded_target``.
  turn 2: a bare "go to the apple" whose ticket mint calls ``_stamp_target_ref``.

The plan (013) and SESSION_STATE predicted the chain would break at either the
*persist* link (active_claims None at mint) or the *fingerprint* gate (AI2-THOR
floats + hardcoded step_count=0 breaking exact tuple equality in
``StationActiveClaims.is_valid_for``).

WHAT WE ACTUALLY FOUND (documented by these tests, not papered over):

  * PERSIST link FIRES. ``active_claims`` is populated after turn 1 with a valid
    ``last_grounded_target``. SESSION_STATE's "active_claims is None (selection
    never recorded)" is STALE post-fix (b5851f2).

  * FINGERPRINT theory is DISPROVEN. Turn 2 reuses turn 1's cached scene_model
    (``_ensure_scene_model`` returns the cached model when the agent hasn't moved),
    so the fingerprint tuple is identical by construction; floats never get a
    chance to break equality. ``_claims_valid_for_current_environment()`` is True
    at mint time.

  * The REAL blocker is the COLOUR-MATCH GATE in ``_stamp_target_ref``
    (jeenom/operator_station.py:5297-5298):

        task_color = params.get("color")            # "" for colourless "the apple"
        if task_color is not None and entry.color != task_color:
            return                                  # None != "" -> True -> early return

    AI2-THOR objects are colourless: ``Ai2thorSense`` emits ``color=None``, so the
    grounded ``entry.color`` is None. But the domain helper / ``canonical_task_params``
    emit ``color=""`` (empty string) for a colourless request. ``None != ""`` is
    True, so the gate returns before stamping. On MiniGrid every object carries a
    real colour string, so ``entry.color == task_color`` holds and the gate passes —
    which is exactly why Aniketh's MiniGrid tests never exercised this failure.

  Two further blockers sit BEHIND the colour gate (would bite even if it were fixed):
    2. ``entry.object_id`` is None because A1 (emit object_id in Ai2thorSense) is
       NOT done — out of A3 scope. So target_ref would carry only a ``coord``, no id.
    3. That ``coord`` is ``(int(entry.x), int(entry.y))`` — int()-truncated floats,
       dead on real AI2-THOR float coords (the plan's "load-bearing constraint").

A3 VERDICT (as filed): BROKEN. First failing gate: operator_station.py:5297-5298.

FOLLOW-UP PASS (plan 013 A1+A2+parser fix, owner DECIDED the parser fix, option 3):
the three blockers are now cleared on our side and this file asserts the FIXED state:
  * parser (ai2thor_domain_helper.py): colourless request emits color=None, not "" —
    unblocks the colour gate WITHOUT touching Aniketh's kernel gate at 5297.
  * A1 (ai2thor_sense.py): native objectId emitted → SceneObject.object_id →
    GroundedObjectEntry.object_id → stamped into target_ref.
  * A2 (ai2thor_sense.py): target_ref-first resolution picks the object by object_id
    over scan order (see the by-object_id acceptance test below — the one that proves
    F13 moved).
The int-truncated coord branch remains dead on AI2-THOR floats (shape-parity only).
"""
from __future__ import annotations

from typing import Any

import pytest

from jeenom.ai2thor_substrate_adapter import build_ai2thor_runtime_package
from jeenom.operator_station import OperatorStationSession


class _Event:
    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata


def _two_apples_metadata() -> dict[str, Any]:
    """Agent at origin; two colourless apples at distance ~3.0 and ~1.0.

    Both carry a native AI2-THOR ``objectId`` (as live Unity would). They are
    description-identical (same objectType, no colour) — the exact case F13 exists
    to disambiguate and the case MiniGrid cannot produce.
    """
    return {
        "objects": [
            {
                "objectType": "Apple",
                "name": "Apple_far",
                "objectId": "Apple|+3.00|+0.90|+0.00",
                "position": {"x": 3.0, "y": 0.9, "z": 0.0},
            },
            {
                "objectType": "Apple",
                "name": "Apple_near",
                "objectId": "Apple|+1.00|+0.90|+0.00",
                "position": {"x": 1.0, "y": 0.9, "z": 0.0},
            },
        ],
        "agent": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
        },
    }


class _MockController:
    """Stub controller: every .step() returns the same static two-apple scene.

    Static metadata means the agent never moves, so the cached scene_model (and its
    fingerprint) is identical across both turns — the fingerprint gate CANNOT be the
    thing that breaks, which is the point we prove below.
    """

    def __init__(self, metadata: dict[str, Any]) -> None:
        self._metadata = metadata
        self.calls: list[dict[str, Any]] = []

    def step(self, **kwargs: Any) -> _Event:
        self.calls.append(kwargs)
        return _Event(metadata=self._metadata)


def _make_session() -> OperatorStationSession:
    controller = _MockController(_two_apples_metadata())
    package = build_ai2thor_runtime_package(controller=controller, scene_id="FloorPlan1")
    return OperatorStationSession(runtime_package=package, seed=0)


# ── Link 3 (PERSIST): grounding records a surviving claim on AI2-THOR ──────────


def test_turn1_grounding_persists_active_claims_on_ai2thor() -> None:
    """PERSIST link FIRES: turn-1 grounding populates active_claims with the winner.

    Disproves SESSION_STATE's stale "active_claims is None (selection never
    recorded)" finding — that pre-dated the F13 merge (b5851f2).
    """
    session = _make_session()

    session.handle_utterance("which apple is closest?")

    claims = session.active_claims
    assert claims is not None, "PERSIST link broken: active_claims is None after grounding"
    entry = claims.last_grounded_target
    assert entry is not None
    assert entry.object_type == "apple"
    # The near apple (distance ~1) must be the closest winner, not scan order.
    assert entry.x == pytest.approx(1.0)


def test_fingerprint_gate_survives_to_mint_on_ai2thor() -> None:
    """FINGERPRINT theory DISPROVEN: claims stay valid at mint time.

    The plan predicted AI2-THOR floats + step_count=0 would break the exact
    tuple-equality fingerprint check. They do not: the agent doesn't move between
    turns, so the cached scene_model's (agent_x, agent_y, step_count) tuple is
    identical, and equality holds even for floats.
    """
    session = _make_session()
    session.handle_utterance("which apple is closest?")

    assert session._claims_valid_for_current_environment() is True, (
        "fingerprint gate unexpectedly invalidated the claim"
    )


# ── Link 4 (STAMP): the colour-match gate no-ops on colourless AI2-THOR objects ─


def test_stamp_target_ref_no_ops_at_mint_due_to_colour_gate() -> None:
    """STAMP link now FIRES at the colour gate after the parser fix (plan 013).

    Drive the REAL two-turn dispatch: spy on ``_stamp_target_ref`` so we observe it
    exactly when turn-2's ticket mint calls it. Claims are present and valid, and —
    now that the domain helper emits ``color=None`` instead of ``""`` for a colourless
    request — the colour gate (operator_station.py:5297) passes and target_ref IS
    written, carrying the winner's object_id (A1 now emits it).

    This is the "fixed-state" counterpart to A3's original broken-state assertion,
    updated per the owner's DECISION to fix the parser (option 3).
    """
    session = _make_session()
    session.handle_utterance("which apple is closest?")

    captures: list[dict[str, Any]] = []
    original_stamp = session._stamp_target_ref

    def spy(params: dict[str, Any]) -> None:
        claims_present = session.active_claims is not None
        claims_valid = session._claims_valid_for_current_environment()
        original_stamp(params)
        captures.append(
            {
                "target_ref": params.get("target_ref"),
                "claims_present": claims_present,
                "claims_valid": claims_valid,
            }
        )

    session._stamp_target_ref = spy  # type: ignore[assignment]
    session.handle_utterance("go to the apple")

    assert captures, "turn-2 mint never called _stamp_target_ref"
    mint = captures[0]
    assert mint["claims_present"] is True
    assert mint["claims_valid"] is True
    # FIXED: a target_ref carrying the near apple's native object_id is stamped.
    assert mint["target_ref"] is not None, "stamp should now fire with color=None"
    assert mint["target_ref"]["object_id"] == "Apple|+1.00|+0.90|+0.00"


def test_colourless_request_now_carries_color_none() -> None:
    """Parser fix: a colourless AI2-THOR request carries color=None, not "".

    This pins the exact fix (ai2thor_domain_helper.py: match.group("color") or None).
    With color=None the stamp's colour gate passes and target_ref is written with the
    winner's object_id; the int-truncated coord is retained for shape-parity but is
    dead on real AI2-THOR floats.
    """
    session = _make_session()
    session.handle_utterance("which apple is closest?")
    assert session._claims_valid_for_current_environment() is True

    task = session.compose_known_task("go to the apple")
    assert task.params.get("color") is None, (
        "parser must emit color=None for a colourless request (fix: 'or None')"
    )
    session._stamp_target_ref(task.params)
    stamped = task.params.get("target_ref")
    assert stamped is not None, "stamp should fire once color is None"
    # A1: object_id now flows from metadata → SceneObject → GroundedObjectEntry → ref.
    assert stamped.get("object_id") == "Apple|+1.00|+0.90|+0.00"
    # Shape-parity coord remains, int()-truncated — dead on AI2-THOR floats.
    assert "coord" in stamped
    assert all(isinstance(v, int) for v in stamped["coord"])


# ── ACCEPTANCE (the one that proves F13 moved): resolve BY object_id ────────────


def test_target_ref_resolves_to_specific_apple_by_object_id() -> None:
    """SEAM (turn1 → stamp → hand-fed sense): grounding + A1 + A2 make Sense pick the
    SPECIFIC apple the kernel chose — by object_id, not scan order, not coord.

    This isolates the stamp+resolution logic by feeding the stamped ref into sense
    directly. The TRUE end-to-end dispatch (target_ref reaching the LIVE runner sense)
    is proven separately in test_target_ref_reaches_live_sense_through_real_dispatch.

    The two apples are description-identical (colourless). The FAR apple is listed
    FIRST in metadata, so a scan-order / last-write-wins resolver would pick it.
    Turn 1 grounds "closest" → the NEAR apple wins and its object_id is persisted.
    Turn 2 stamps that object_id into target_ref; Ai2thorSense's target_ref-first
    pass (A2) must resolve target_object to the NEAR apple by matching object_id.

    Asserted BY object_id — the whole point of F13. A coord/scan-order fallback would
    resolve to the far apple (first in scan order) and fail this test.
    """
    from jeenom.schemas import EvidenceFrame, ExecutionContext

    session = _make_session()

    # Turn 1: ground closest. Winner is the NEAR apple (distance ~1), NOT scan order.
    session.handle_utterance("which apple is closest?")
    winner = session.active_claims.last_grounded_target
    assert winner.object_id == "Apple|+1.00|+0.90|+0.00", "closest winner must be near apple"

    # Turn 2: compose + stamp the go-to task exactly as the mint path does.
    task = session.compose_known_task("go to the apple")
    session._stamp_target_ref(task.params)
    target_ref = task.params.get("target_ref")
    assert target_ref is not None and target_ref.get("object_id") == winner.object_id

    # Now drive Ai2thorSense with that stamped target_ref and prove it resolves the
    # SPECIFIC apple by object_id — the far apple is first in scan order, so a
    # description-only resolver would wrongly pick it.
    sense = session.sense
    observation = _Event(_two_apples_metadata())
    evidence_frame = EvidenceFrame(needs=["target_location"])
    execution_context = ExecutionContext(
        active_skill="locate_object",
        params={
            "object_type": "apple",
            "color": None,
            "target_ref": target_ref,
        },
    )

    _, _, sample, _, _ = sense.tick(observation, evidence_frame, execution_context)

    assert sample.target_visible is True
    assert sample.target_object is not None
    # BY OBJECT_ID: resolved to the near apple the kernel chose, not the far
    # (scan-order-first) apple.
    assert sample.target_object["object_id"] == "Apple|+1.00|+0.90|+0.00", (
        "Sense must resolve the kernel-chosen apple by object_id, not scan order"
    )
    assert sample.target_location == (1.0, 0.0)


def test_target_ref_first_beats_scan_order_directly() -> None:
    """Unit-level A2 proof: with two apples (far first) and a target_ref naming the
    near one, Ai2thorSense resolves the near apple — isolating the target_ref-first
    pass from the grounding machinery."""
    from jeenom.schemas import EvidenceFrame, ExecutionContext

    session = _make_session()
    sense = session.sense
    observation = _Event(_two_apples_metadata())  # far apple first in scan order
    evidence_frame = EvidenceFrame(needs=["target_location"])
    execution_context = ExecutionContext(
        active_skill="locate_object",
        params={
            "object_type": "apple",
            "color": None,
            "target_ref": {"object_id": "Apple|+1.00|+0.90|+0.00", "coord": (1, 0)},
        },
    )

    _, _, sample, _, _ = sense.tick(observation, evidence_frame, execution_context)

    assert sample.target_object["object_id"] == "Apple|+1.00|+0.90|+0.00"


def test_object_id_flows_metadata_to_scene_object() -> None:
    """A1 assertion: native objectId reaches SceneObject.object_id (no schema edit)."""
    session = _make_session()
    session.handle_utterance("which apple is closest?")
    scene = session.memory.scene_model
    ids = {o.object_id for o in scene.objects}
    assert ids == {"Apple|+1.00|+0.90|+0.00", "Apple|+3.00|+0.90|+0.00"}


# ── TRUE END-TO-END: target_ref reaches the LIVE sense through real dispatch ────


class _NavEvent:
    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata


def _nav_apples() -> list[dict[str, Any]]:
    """Far apple FIRST in scan order, near apple second — both colourless, id-bearing."""
    return [
        {"objectType": "Apple", "objectId": "Apple|+0.75",
         "position": {"x": 0.75, "y": 0.9, "z": 0.0}},
        {"objectType": "Apple", "objectId": "Apple|+0.25",
         "position": {"x": 0.25, "y": 0.9, "z": 0.0}},
    ]


class _NavMockController:
    """Navigation-capable mock: supports GetReachablePositions + MoveAhead/Rotate so
    the real episode actually reaches the sense (locate) step — a static mock fails
    navigation before sense ever runs, so it cannot prove the propagation seam."""

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
            "objects": _nav_apples(),
            "agent": {
                "position": {"x": self.ax, "y": 0.0, "z": self.az},
                "rotation": {"x": 0.0, "y": self.yaw, "z": 0.0},
            },
        })


def test_target_ref_reaches_live_sense_through_real_dispatch(monkeypatch) -> None:
    """LOAD-BEARING: the stamped target_ref actually reaches Ai2thorSense through the
    REAL turn-2 dispatch (ticket → cortex → locate step → execution_context.params),
    and the LIVE sense resolves the kernel-chosen apple by object_id.

    This is the anti-"green wire that no-ops in the real loop" check. It does NOT
    hand-build the ExecutionContext; it spies on Ai2thorSense.tick during a genuine
    handle_utterance("go to the apple") and asserts on the target_ref the runner
    actually delivered. run_task_episode constructs its OWN Ai2thorSense, so we patch
    at the class level.
    """
    from jeenom.ai2thor_substrate_adapter import build_ai2thor_runtime_package
    import jeenom.ai2thor_sense as ai2thor_sense_module

    controller = _NavMockController()
    package = build_ai2thor_runtime_package(controller=controller, scene_id="FloorPlan1")
    session = OperatorStationSession(runtime_package=package, seed=0)

    # Turn 1: ground closest. Near apple (0.25) wins over far (0.75, first in scan).
    session.handle_utterance("which apple is closest?")
    assert session.active_claims.last_grounded_target.object_id == "Apple|+0.25"

    captures: list[dict[str, Any]] = []
    original_tick = ai2thor_sense_module.Ai2thorSense.tick

    def spy_tick(self, observation, evidence_frame, execution_context, *args, **kwargs):
        ctx_params = getattr(execution_context, "params", {}) or {}
        frame_ctx = getattr(evidence_frame, "context", {}) or {}
        delivered_ref = ctx_params.get("target_ref") or frame_ctx.get("target_ref")
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

    # Turn 2: the real front door. Ticket mint stamps target_ref; the runner must
    # thread it to the live sense.
    session.handle_utterance("go to the apple")

    # The runner DID invoke the live sense, and it received the id-bearing ref.
    refs_delivered = [
        c for c in captures if isinstance(c["delivered_target_ref"], dict)
    ]
    assert refs_delivered, (
        "target_ref never reached the live Ai2thorSense — A2 wiring hole; F13 has NOT "
        "moved despite the stamp firing"
    )
    delivered = refs_delivered[0]["delivered_target_ref"]
    assert delivered.get("object_id") == "Apple|+0.25", (
        "the kernel-chosen (near) apple's object_id must reach sense, not scan order"
    )
    # And the live sense resolved BY that object_id, not the far scan-order-first apple.
    resolved = [
        c["resolved_object_id"] for c in captures
        if c["resolved_object_id"] is not None
    ]
    assert "Apple|+0.25" in resolved
    assert "Apple|+0.75" not in resolved, (
        "sense resolved the far (scan-order-first) apple — target_ref-first pass failed"
    )
