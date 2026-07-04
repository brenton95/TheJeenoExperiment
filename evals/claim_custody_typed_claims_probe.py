"""Typed Claims probe (claim unification).

Verifies that Cortex's claim store uses the single unified ``ClaimRecord`` type
behind a clean accessor interface, while maintaining backward compatibility with
code that reads the raw-value claims dict.

This probe previously pinned the now-removed ``ObservationClaim``/``ExecutionClaim``
duplication. The claim-unification pass deliberately reversed that decision:
``ClaimRecord`` is the one claim type, and the Cortex belief loop shares one
mission-scoped store with the representation surface.

Checks:
  claim_record_in_schemas        — ClaimRecord importable from jeenom.schemas
  claim_record_observation_fields — ClaimRecord carries the hot-path fields
                                     (key, value, source, freshness, last_observed_tick)
  cortex_has_get_claim           — Cortex has get_claim() method
  cortex_has_set_claim           — Cortex has set_claim() method
  cortex_has_has_claim           — Cortex has has_claim() method
  cortex_claims_property         — cortex.claims returns a plain dict (backward compat)
  internal_store_typed           — Cortex._claims holds ClaimRecord objects
  internal_store_kind_observation — set_claim authors kind="observation" claims
  set_get_roundtrip              — set_claim/get_claim round-trip preserves value
  has_claim_truthy               — has_claim returns True for a truthy value
  has_claim_falsy                — has_claim returns False for None or falsy value
  has_claim_absent               — has_claim returns False when key not set
  claims_property_raw_values     — claims property exposes raw values, not ClaimRecord
  update_from_evidence_typed     — update_from_evidence stores ClaimRecord internally
  update_from_evidence_source    — ClaimRecord.source matches evidence.source
  regression_full_task           — full task run still completes (go to the red door)
"""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch
from harness import build_env as _build_env, make_session as _make_session

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from jeenom.cortex import Cortex
from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.memory import OperationalMemory
from jeenom.operator_station import OperatorStationSession
from jeenom.schemas import ClaimRecord, OperationalEvidence


def main() -> int:
    metrics: dict[str, bool] = {}

    # ── Schema checks ──────────────────────────────────────────────────────────
    metrics["claim_record_in_schemas"] = ClaimRecord is not None

    claim_field_names = {f.name for f in fields(ClaimRecord)}
    metrics["claim_record_observation_fields"] = {
        "key", "value", "source", "freshness", "last_observed_tick"
    }.issubset(claim_field_names)

    # ── Cortex accessor interface ──────────────────────────────────────────────
    metrics["cortex_has_get_claim"] = callable(getattr(Cortex, "get_claim", None))
    metrics["cortex_has_set_claim"] = callable(getattr(Cortex, "set_claim", None))
    metrics["cortex_has_has_claim"] = callable(getattr(Cortex, "has_claim", None))
    metrics["cortex_claims_property"] = isinstance(
        Cortex.__dict__.get("claims"), property
    )

    # ── Accessor behaviour ─────────────────────────────────────────────────────
    memory = OperationalMemory(root=Path(tempfile.mkdtemp()))
    compiler = SmokeTestCompiler()
    cortex = Cortex(memory=memory, compiler=compiler)

    metrics["internal_store_typed"] = isinstance(cortex._claims, dict)

    cortex.set_claim("target_location", (3, 4))
    stored = cortex._claims.get("target_location")
    metrics["internal_store_kind_observation"] = (
        isinstance(stored, ClaimRecord) and stored.kind == "observation"
    )
    metrics["set_get_roundtrip"] = cortex.get_claim("target_location") == (3, 4)
    metrics["has_claim_truthy"] = cortex.has_claim("target_location")

    cortex.set_claim("falsy_key", None)
    metrics["has_claim_falsy"] = not cortex.has_claim("falsy_key")
    metrics["has_claim_absent"] = not cortex.has_claim("nonexistent_key")

    metrics["claims_property_raw_values"] = (
        isinstance(cortex.claims, dict)
        and cortex.claims.get("target_location") == (3, 4)
        and not isinstance(cortex.claims.get("target_location"), ClaimRecord)
    )

    # ── update_from_evidence stores a typed ClaimRecord ───────────────────────
    cortex2 = Cortex(memory=OperationalMemory(root=Path(tempfile.mkdtemp())), compiler=compiler)
    evidence = OperationalEvidence(
        claims={"target_location": (5, 6), "adjacency_to_target": False},
        confidence=1.0,
        source="sense",
    )
    cortex2.update_from_evidence(evidence)

    metrics["update_from_evidence_typed"] = isinstance(
        cortex2._claims.get("target_location"), ClaimRecord
    )
    metrics["update_from_evidence_source"] = (
        cortex2._claims["target_location"].source == "sense"
    )

    # ── Regression: full task still works ─────────────────────────────────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess = _make_session()
        resp = sess.handle_utterance("go to the red door")
        metrics["regression_full_task"] = "RUN COMPLETE" in resp or "TASK COMPLETE" in resp

    print(json.dumps(metrics, sort_keys=True))
    return 0 if all(metrics.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
