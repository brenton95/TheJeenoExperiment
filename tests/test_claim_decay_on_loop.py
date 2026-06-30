"""Step 2 of the partial-observability fix: the decay machine runs on the loop.

Step 1 made the hot-path claim type carry/respect freshness. This step lights the
Phase 13B decay machine (`next_freshness` / `UNVERIFIABLE_DECAY_STEPS`) on the
cortex loop, clocked by `world_sample.step_count`:

- A claim observed this tick is stamped `last_observed_tick = step_count` and read
  as ``current``.
- A claim NOT re-observed this tick ages: ``current -> unverifiable`` on the first
  look-away, then ``unverifiable -> unknown`` once
  ``step_count - last_observed_tick >= UNVERIFIABLE_DECAY_STEPS``.

No spatial map yet (Step 3). The point here is to prove decay transitions a claim
on the real loop, against the pure-function contract pinned by
``evals/claim_custody_unverifiable_freshness_probe.py``.

Decay is intra-task and uniform across kinds by deliberate, flagged debt — see the
``# TECH-DEBT(...)`` tags in jeenom/cortex.py.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

from jeenom.claim_freshness import UNVERIFIABLE_DECAY_STEPS
from jeenom.cortex import Cortex
from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.memory import OperationalMemory
from jeenom.schemas import OperationalEvidence


def _cortex() -> Cortex:
    cortex = Cortex(
        memory=OperationalMemory(root=Path(tempfile.mkdtemp())),
        compiler=SmokeTestCompiler(),
    )
    # Seat mid-procedure on navigate_to_object so step advancement stays inert.
    cortex.procedure = SimpleNamespace(
        steps=["locate_object", "navigate_to_object", "verify_adjacent", "done"]
    )
    cortex.execution_state["step_index"] = 1
    cortex.resolved_task_params = {"color": "red", "object_type": "door"}
    return cortex


def _ev(claims: dict) -> OperationalEvidence:
    return OperationalEvidence(claims=claims, confidence=1.0, source="sense")


def _ws(step_count: int) -> SimpleNamespace:
    return SimpleNamespace(step_count=step_count)


def test_observed_claim_is_current_and_stamped_with_tick():
    cortex = _cortex()
    cortex.update_from_evidence(_ev({"target_location": (0, 2)}), world_sample=_ws(0))

    claim = cortex._claims["target_location"]
    assert claim.freshness == "current"
    assert claim.last_observed_tick == 0
    assert cortex.get_claim("target_location") == (0, 2)


def test_lookaway_drops_carried_claim_to_unverifiable_but_still_usable():
    cortex = _cortex()
    cortex.update_from_evidence(_ev({"target_location": (0, 2)}), world_sample=_ws(0))
    # Next tick the target is not re-observed (not in evidence).
    cortex.update_from_evidence(_ev({"agent_pose": {"x": 1, "y": 1, "dir": 0}}), world_sample=_ws(1))

    claim = cortex._claims["target_location"]
    assert claim.freshness == "unverifiable"
    # unverifiable is still our best belief — the hot path still acts on it.
    assert cortex.get_claim("target_location") == (0, 2)


def test_carried_claim_decays_to_unknown_at_the_constant():
    cortex = _cortex()
    cortex.update_from_evidence(_ev({"target_location": (0, 2)}), world_sample=_ws(0))
    cortex.update_from_evidence(_ev({}), world_sample=_ws(1))  # -> unverifiable
    assert cortex._claims["target_location"].freshness == "unverifiable"

    cortex.update_from_evidence(_ev({}), world_sample=_ws(UNVERIFIABLE_DECAY_STEPS))
    claim = cortex._claims["target_location"]
    assert claim.freshness == "unknown"
    assert cortex.get_claim("target_location") is None


def test_decay_to_unknown_is_parameterized_on_the_constant_not_a_literal():
    cortex = _cortex()
    cortex.update_from_evidence(_ev({"target_location": (0, 2)}), world_sample=_ws(0))
    cortex.update_from_evidence(_ev({}), world_sample=_ws(1))
    # One tick short of the threshold: still unverifiable.
    cortex.update_from_evidence(_ev({}), world_sample=_ws(UNVERIFIABLE_DECAY_STEPS - 1))
    assert cortex._claims["target_location"].freshness == "unverifiable"


def test_reobservation_snaps_belief_back_to_current():
    cortex = _cortex()
    cortex.update_from_evidence(_ev({"target_location": (0, 2)}), world_sample=_ws(0))
    cortex.update_from_evidence(_ev({}), world_sample=_ws(1))
    assert cortex._claims["target_location"].freshness == "unverifiable"

    cortex.update_from_evidence(_ev({"target_location": (0, 2)}), world_sample=_ws(2))
    claim = cortex._claims["target_location"]
    assert claim.freshness == "current"
    assert claim.last_observed_tick == 2


def test_decay_is_uniform_across_claim_kinds_tech_debt():
    """Uniform-decay debt: a non-spatial-named claim still ages as an observation.

    Locks the deliberate simplification (uniform UNVERIFIABLE_DECAY_STEPS, no
    per-kind rates) so it is visible and intentional, not silent.
    """
    cortex = _cortex()
    cortex.update_from_evidence(_ev({"some_named_fact": "x"}), world_sample=_ws(0))
    cortex.update_from_evidence(_ev({}), world_sample=_ws(1))
    assert cortex._claims["some_named_fact"].freshness == "unverifiable"


def test_no_step_count_means_no_decay():
    """Backward compatibility: callers that pass no world_sample drive no decay."""
    cortex = _cortex()
    cortex.update_from_evidence(_ev({"target_location": (0, 2)}), world_sample=_ws(0))
    cortex.update_from_evidence(_ev({"agent_pose": {"x": 1, "y": 1, "dir": 0}}))  # no world_sample
    assert cortex._claims["target_location"].freshness == "current"
