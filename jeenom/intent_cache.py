"""IntentCache: precompiled regex patterns that produce OperatorIntent on match.

Fast-path NLU that routes through the same dispatch pipeline as LLM-compiled intents.
This is a parallel lookup to the LLM compiler, not a replacement — both paths run
IntentVerifier and dispatch after matching.

Also exports SEQUENCE_STEP_PREFIX / SEQUENCE_STEP_SUFFIX for use by
OperatorStationSession._try_natural_sequence (avoids re.compile in method bodies).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

# Used by operator_station._try_natural_sequence
SEQUENCE_STEP_PREFIX: re.Pattern[str] = re.compile(
    r"^\s*(?:(?:do|execute|run|perform|also)\s+)?(?:a\s+|the\s+|an\s+)?(?:first\s+)?"
)
SEQUENCE_STEP_SUFFIX: re.Pattern[str] = re.compile(r"\s+(?:first|next|also|too)\s*$")

_VERB_PREFIX = (
    r"^(?:i\s+(?:want\s+to|would\s+like\s+to|need\s+to|'d\s+like\s+to)\s+)?"
    r"(?:please\s+)?(?:define|make|create|synthesize|build)\s+"
)

_PRIM_DEF_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        _VERB_PREFIX
        + r"(?:a\s+|an\s+)?(?:new\s+)?(?:distance\s+)?metric\s+"
        r"(?:(?:called|named)\s+)?(?P<name>[A-Za-z][A-Za-z0-9_ ]*?)\s*=\s*"
        r"(?P<formula>.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        _VERB_PREFIX
        + r"(?:a\s+|an\s+)?(?:new\s+)?(?:distance\s+)?metric\s+"
        r"(?:called|named)\s+(?P<name>[A-Za-z][A-Za-z0-9_ ]*?)\s+"
        r"(?:as|to be|which is|that is|that|using|where|based on)\s+(?P<formula>.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        _VERB_PREFIX
        + r"(?:a\s+|an\s+)?(?:new\s+)?distance\s+metric\s+"
        r"(?:which is|that is|as)\s+(?P<formula>.+?)\s+"
        r"(?:and\s+)?call\s+it\s+(?P<name>[A-Za-z][A-Za-z0-9_ ]*)$",
        re.IGNORECASE,
    ),
)

_METRIC_QUERY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:rank|list|show)\s+(?:all\s+)?(?:the\s+)?doors\s+by\s+"
        r"(?P<metric>[A-Za-z][A-Za-z0-9_]*)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:what\s+is|whats|what's|show|list)\s+(?:the\s+)?"
        r"(?P<metric>[A-Za-z][A-Za-z0-9_]*)\s+"
        r"(?:distance\s+)?(?:to|for|of)\s+all\s+(?:the\s+)?doors\b",
        re.IGNORECASE,
    ),
)

_METRIC_QUERY_STOPWORDS = frozenset({
    "distance", "distances", "closest", "farthest", "furthest",
    "door", "doors", "all", "the", "manhattan", "euclidean",
})

_CONCEPT_TEACH_PAT: re.Pattern[str] = re.compile(
    r"^(?:remember|teach|define)\s+(.+?)\s+(?:means|as|is shorthand for)\s+(.+)$",
    re.IGNORECASE,
)

_CONCEPT_FORGET_PAT: re.Pattern[str] = re.compile(
    r"^forget(?:\s+concept)?\s+(?P<name>.+)$",
    re.IGNORECASE,
)

_DELIVERY_TARGET_PAT: re.Pattern[str] = re.compile(
    r"^(?:go\s+to|reach|find|get\s+to|head\s+to|navigate\s+to)\s+(?:the\s+)?delivery\s+target$",
    re.IGNORECASE,
)


@dataclass
class IntentCache:
    """Precompiled regex → OperatorIntent lookup.

    Entries may use match (anchored) or search (substring) mode depending on the
    pattern's semantics.  Builders return OperatorIntent | ApprovedCommand | None;
    None means "no match" and the next entry is tried.
    """

    # (pattern, builder, use_search)
    _entries: list[tuple[re.Pattern[str], Callable[[re.Match[str]], Any], bool]] = field(
        default_factory=list, repr=False
    )

    def register(
        self,
        pattern: str,
        intent_builder: Callable[[re.Match[str]], Any],
        flags: int = re.IGNORECASE,
        search: bool = False,
    ) -> None:
        self._entries.append((re.compile(pattern, flags), intent_builder, search))

    def _register_compiled(
        self,
        compiled: re.Pattern[str],
        intent_builder: Callable[[re.Match[str]], Any],
        search: bool = False,
    ) -> None:
        self._entries.append((compiled, intent_builder, search))

    def lookup(self, utterance: str) -> Any | None:
        text = utterance.strip()
        for compiled, builder, use_search in self._entries:
            m = compiled.search(text) if use_search else compiled.match(text)
            if m is not None:
                result = builder(m)
                if result is not None:
                    return result
        return None


def _build_primitive_definition_from_match(
    m: re.Match[str],
    registry: Any,
) -> Any:
    """Build OperatorIntent or ApprovedCommand(unsupported) from a matched prim-def pattern."""
    from .mission_cortex import (
        _metric_dependencies,
        _normalize_metric_name,
        _parse_metric_expression,
    )
    from .schemas import ApprovedCommand, OperatorIntent, PrimitiveDefinitionRequest

    name = m.group("name")
    formula = m.group("formula").strip()
    normalized_name = _normalize_metric_name(name)
    known_metrics = list(registry.ranked_metric_handles().keys())
    expression = _parse_metric_expression(formula, known_metrics)
    if expression is None:
        return ApprovedCommand(
            command_type="unsupported",
            utterance=m.string,
            payload={
                "message": (
                    "I could not parse that metric formula. Use a query-only formula "
                    "such as min(euclidean, manhattan), euclidean mod 5, or manhattan plus 3."
                )
            },
        )
    if expression.get("op") == "unsafe":
        return ApprovedCommand(
            command_type="unsupported",
            utterance=m.string,
            payload={
                "message": (
                    "REFUSE\nMetric definitions must be query-only. I will not build a metric "
                    "that contains actuation, movement, controller, or motor side effects."
                )
            },
        )
    dependencies = list(dict.fromkeys(_metric_dependencies(formula, known_metrics)))
    dependency_handles = [registry.ranked_handle_for(metric) for metric in dependencies]
    prim_def = PrimitiveDefinitionRequest(
        definition_type="distance_metric",
        name=name,
        normalized_name=normalized_name,
        expression=expression,
        dependencies=dependencies,
        dependency_handles=dependency_handles,
        proposed_handle=registry.ranked_handle_for(normalized_name),
        safety_class="query",
        authority_level="operator",
        provenance={"operator_utterance": m.string, "formula": formula},
    )
    return OperatorIntent(intent_type="primitive_definition", primitive_definition=prim_def)


def _build_metric_query_from_match(m: re.Match[str]) -> Any | None:
    from .mission_cortex import _normalize_metric_name
    from .schemas import OperatorIntent

    metric = m.group("metric")
    normalized = _normalize_metric_name(metric)
    if not normalized or normalized in _METRIC_QUERY_STOPWORDS:
        return None
    return OperatorIntent(intent_type="metric_query", status_query=normalized)


def parse_metric_query(text: str) -> str | None:
    """Return normalized metric name if text is a metric-query utterance, else None.

    Exported for use by classify_utterance (avoids re.compile in operator_station.py).
    """
    from .mission_cortex import _normalize_metric_name

    for pat in _METRIC_QUERY_PATTERNS:
        m = pat.search(text.strip())
        if not m:
            continue
        metric = m.group("metric")
        normalized = _normalize_metric_name(metric)
        if normalized and normalized not in _METRIC_QUERY_STOPWORDS:
            return normalized
    return None


def seed_intent_cache(cache: IntentCache, registry: Any) -> None:
    """Register fast-path NLU patterns into cache at station startup.

    Metric-definition, metric-query, concept-teach, and concept-forget patterns are
    registered here; cache hits produce OperatorIntent and route through dispatch
    (IntentVerifier + knowledge-type routing) like LLM intents. Metric queries own their
    metric->handle resolution downstream in metric_query_summary, so dispatch exempts
    them from the premature capability gate rather than arbitrating an unresolved metric
    name. Delivery-target remains in classify_utterance as an exact continuation
    pattern.
    """
    from .schemas import OperatorIntent

    # Metric-query patterns (substring search) → metric_query intent through dispatch.
    for pat in _METRIC_QUERY_PATTERNS:
        cache._register_compiled(pat, _build_metric_query_from_match, search=True)

    # Metric-definition patterns (anchored match, 3 variants) → primitive_definition intent
    for pat in _PRIM_DEF_PATTERNS:
        def _make_prim_builder(registry: Any = registry) -> Callable[[re.Match[str]], Any]:
            def builder(m: re.Match[str]) -> Any:
                return _build_primitive_definition_from_match(m, registry)
            return builder

        cache._register_compiled(pat, _make_prim_builder(), search=False)

    # Concept-teach: anchored match against raw utterance (preserves commas in expansion)
    def _concept_teach_builder(m: re.Match[str]) -> OperatorIntent:
        cname = m.group(1).strip().strip("'\"")
        cutterance = m.group(2).strip().strip("'\"")
        return OperatorIntent(
            intent_type="concept_teach",
            concept_name=cname,
            concept_utterance=cutterance,
        )

    cache._register_compiled(_CONCEPT_TEACH_PAT, _concept_teach_builder, search=False)

    # Concept-forget: anchored match → control path.
    # Hard-reset phrases ("everything", "memory", etc.) must fall through to
    # classify_utterance which routes them as reset(clear_memory=True).
    _FORGET_HARD_RESET_NAMES: frozenset[str] = frozenset({
        "everything", "memory", "all", "all memory",
    })

    def _concept_forget_builder(m: re.Match[str]) -> OperatorIntent | None:
        cname = m.group("name").strip().strip("'\"")
        if cname.lower() in _FORGET_HARD_RESET_NAMES:
            return None  # let classify_utterance handle as hard reset
        return OperatorIntent(intent_type="concept_forget", concept_name=cname)

    cache._register_compiled(_CONCEPT_FORGET_PAT, _concept_forget_builder, search=False)
