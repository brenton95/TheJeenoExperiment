"""Granular semantic normalization of grounding-query answer_fields (the easy win).

The LLM emits answer-field vocabulary that drifts from the canonical set the deterministic
executor recognizes (e.g. "distances" vs "distance"). Per the tool-call-discipline principle,
we normalize the tool-call BEFORE dispatch: repair near-misses to canonical (granular — the
rest of the intent is kept), and fail CLOSED on a genuinely-unknown value (→ regex fallback,
else "I didn't understand"). It is substrate-INDEPENDENT (shared vocabulary), the counterpart
to the per-substrate domain helper. Wiring into the grounding-plan parse chokepoint is covered
by the compose integration test; this pins the canonicalizer itself.
"""
from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pytest

from jeenom.schemas import (
    GROUNDING_QUERY_ANSWER_FIELDS,
    SchemaValidationError,
    _ensure_canonical_answer_fields,
)


def test_near_miss_is_repaired_to_canonical():
    assert _ensure_canonical_answer_fields(["distances"], "x") == ["distance"]


def test_synonym_aliases_canonicalize():
    assert _ensure_canonical_answer_fields(["nearest", "furthest"], "x") == ["closest", "farthest"]


def test_casing_is_normalized():
    assert _ensure_canonical_answer_fields(["Distances", "CLOSEST"], "x") == ["distance", "closest"]


def test_canonical_values_pass_through_unchanged():
    canonical = ["distance", "ranked_doors", "closest", "farthest", "exists", "target"]
    assert _ensure_canonical_answer_fields(canonical, "x") == canonical
    for field in canonical:
        assert field in GROUNDING_QUERY_ANSWER_FIELDS


def test_ordinal_pattern_answer_fields_pass_through():
    assert _ensure_canonical_answer_fields(["second_closest", "third_farthest"], "x") == [
        "second_closest",
        "third_farthest",
    ]


def test_unknown_answer_field_fails_closed():
    with pytest.raises(SchemaValidationError):
        _ensure_canonical_answer_fields(["frobnicate"], "x")


# ---- compose integration: the LLM's answer+distances shape now produces a real answer ----

def _answer_distances_intent():
    """A status_query whose grounding plan uses operation=answer + answer_fields=['distances']
    — the exact LLM shape that previously dead-ended in compose."""
    return {
        "intent_type": "status_query",
        "canonical_instruction": None,
        "task_type": None,
        "target": None,
        "knowledge_update": None,
        "reference": None,
        "status_query": "ground_target",
        "claim_reference": None,
        "control": None,
        "target_selector": None,
        "grounding_query_plan": {
            "object_type": "door",
            "operation": "answer",
            "primitive_handle": "grounding.all_doors.ranked.manhattan.agent",
            "metric": "manhattan",
            "reference": "agent",
            "order": "ascending",
            "ordinal": None,
            "color": None,
            "exclude_colors": [],
            "distance_value": None,
            "comparison": None,
            "tie_policy": "clarify",
            "answer_fields": ["distances"],
            "required_capabilities": ["grounding.all_doors.ranked.manhattan.agent"],
            "preserved_constraints": [],
        },
        "primitive_definition": None,
        "capability_status": "executable",
        "required_capabilities": ["grounding.all_doors.ranked.manhattan.agent"],
        "clear_memory": False,
        "confidence": 1.0,
        "reason": "test: answer the distances to all doors",
        "concept_name": None,
        "concept_utterance": None,
        "concept_steps": None,
        "utterance_steps": None,
        "action_name": None,
        "repeat_count": None,
        "mission_steps": None,
        "selection_objective": None,
        "steering_directive": None,
    }


def test_answer_distances_composes_instead_of_dead_ending():
    from jeenom.llm_compiler import LLMCompiler
    from jeenom.minigrid_runtime_package import build_minigrid_runtime_package
    from jeenom.operator_station import OperatorStationSession

    intent = _answer_distances_intent()

    def transport(request):
        assert request.get("method_name") == "compile_operator_intent"
        return copy.deepcopy(intent)

    # Full observation so doors are visible deterministically (compose runs, not needs_evidence).
    pkg = build_minigrid_runtime_package(
        env_id="MiniGrid-GoToDoor-16x16-v0", render_mode="none", observability="full"
    )
    session = OperatorStationSession(
        compiler=LLMCompiler(api_key="test-key", transport=transport),
        compiler_name="llm",
        runtime_package=pkg,
        env_id="MiniGrid-GoToDoor-16x16-v0",
        seed=8,
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
    )
    try:
        command = session.command_from_llm_intent("what is the distance to the doors")
        message = str((command.payload or {}).get("message", ""))
        # the near-miss was canonicalized at the parse chokepoint...
        assert session.last_operator_intent.grounding_query_plan["answer_fields"] == ["distance"]
        # ...and compose produced a real ranked-distance answer, not the dead-end.
        assert "could not compose" not in message.lower()
        assert command.kind != "ambiguous"
        assert "door" in message.lower()
    finally:
        session.close()
