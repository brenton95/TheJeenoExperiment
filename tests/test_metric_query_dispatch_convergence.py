"""TECH-DEBT(metric-query-dispatch-bypass) closure — metric queries converge through dispatch.

Pressure: `classify_utterance` minted a `metric_query` ApprovedCommand directly, skipping
the IntentVerifier/dispatch convergence chain entirely. The stated fear ("routing them
through dispatch causes capability-matching regressions") turned out to be stale for
defined metrics — `metric_query_summary` already re-dispatches a grounding intent through
the same gate, so both routes converge. The real gap was the undefined-metric path: it
never constructed an OperatorIntent at all, so the verifier never ran and
`last_operator_intent` stayed None — a deterministic/LLM parity hole and an unverified
semantic route.

Fix under test: metric-query patterns move into the IntentCache (whose entries produce
OperatorIntent and route through dispatch + IntentVerifier like LLM intents), the classify
bypass is deleted, and dispatch exempts `metric_query` from the premature capability gate
because the metric owns its handle resolution downstream (defined -> re-dispatched
grounding intent through the same gate; undefined -> typed CUSTOM METRIC MISSING
definition flow).

Red bars: classify must return `unresolved` for metric queries (bypass gone), and an
undefined-metric turn must have a verified `metric_query` intent on record. Behavior pins:
registered, synthesizable, and undefined metrics keep today's operator-visible outcomes.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jeenom.operator_station import classify_utterance
from tests.test_heterogeneous_sequence import _build_env, _make_session


def _fresh_session():
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        return _make_session()


def _turn(sess, utterance: str) -> str:
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        result = sess.handle_utterance(utterance)
    return result.message if hasattr(result, "message") else str(result)


def test_classify_no_longer_bypasses_dispatch_for_metric_queries():
    """The classify-level bypass is gone: metric queries are not resolved into a raw
    ApprovedCommand before the convergence chain. (They hit the IntentCache, which
    routes through dispatch, before classify is ever consulted.)"""
    sess = _fresh_session()
    command = classify_utterance(
        "rank all doors by manhattan",
        sess.capability_registry,
        domain_helper=sess.domain_helper,
        planning_semantics=sess.planning_semantics,
    )
    assert command.kind == "unresolved"


def test_undefined_metric_runs_through_verifier_and_still_guides_definition():
    """The undefined-metric turn must carry a verified metric_query intent (previously
    last_operator_intent stayed None — the verifier never saw the utterance) while
    preserving the typed CUSTOM METRIC MISSING definition flow, not a generic
    capability refusal."""
    sess = _fresh_session()
    message = _turn(sess, "rank all doors by convenientdist")
    assert "CUSTOM METRIC MISSING" in message
    assert "convenientdist" in message
    assert sess.last_operator_intent is not None
    assert sess.last_operator_intent.intent_type == "metric_query"


def test_registered_metric_behavior_preserved():
    """Behavior pin: a metric with a registered ranked handle still answers directly."""
    sess = _fresh_session()
    message = _turn(sess, "rank all doors by manhattan")
    assert "RANKED BY MANHATTAN" in message


def test_synthesizable_metric_behavior_preserved():
    """Behavior pin: a metric whose ranked handle is synthesizable still produces the
    synthesis proposal via the inner grounding dispatch — the same gate both routes
    always converged through for defined metrics."""
    sess = _fresh_session()
    message = _turn(sess, "rank all doors by euclidean")
    assert "SYNTHESIS PROPOSAL" in message
    assert "grounding.all_doors.ranked.euclidean.agent" in message
