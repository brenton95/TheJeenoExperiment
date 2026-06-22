from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from typing import Callable

from .schemas import OperationalContext, OperatorIntent


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


@dataclass(frozen=True)
class PlanningSemantics:
    """Context-driven meaning used by the planner and verifier boundary."""

    operational_context: OperationalContext

    @property
    def object_types(self) -> tuple[str, ...]:
        return tuple(self.operational_context.object_vocabulary)

    @property
    def default_object_type(self) -> str | None:
        return self.object_types[0] if self.object_types else None

    @property
    def motor_object_terms(self) -> tuple[str, ...]:
        """Object words that, following a pickup/toggle verb, mark an utterance as
        an object task rather than a bare motor command. Domain-configured; falls
        back to the navigation object vocabulary when unspecified."""
        terms = self.operational_context.grounding_semantics.get("motor_object_terms")
        if isinstance(terms, list):
            configured = tuple(str(term) for term in terms if isinstance(term, str))
            if configured:
                return configured
        return self.object_types

    @property
    def metrics(self) -> tuple[str, ...]:
        metrics = self.operational_context.grounding_semantics.get("distance_metrics", [])
        if not isinstance(metrics, list):
            return ()
        return tuple(str(metric) for metric in metrics if isinstance(metric, str))

    @property
    def default_metric(self) -> str | None:
        closest = self.operational_context.reference_semantics.get("closest", {})
        if isinstance(closest, dict) and isinstance(closest.get("default_metric"), str):
            return closest["default_metric"]
        return self.metrics[0] if self.metrics else None

    @property
    def ranked_claims_output(self) -> str:
        output = self.operational_context.grounding_semantics.get("ranked_claims_output")
        if isinstance(output, str) and output:
            return output
        object_type = self.default_object_type or "object"
        return f"active_claims.ranked_{self.pluralize(object_type)}"

    def attribute_values(self, attribute: str) -> tuple[str, ...]:
        values = self.operational_context.grounding_semantics.get("attribute_values", {})
        if not isinstance(values, dict):
            return ()
        raw = values.get(attribute, [])
        if not isinstance(raw, list):
            return ()
        return tuple(str(item) for item in raw if isinstance(item, str))

    def pluralize(self, object_type: str) -> str:
        irregular = self.operational_context.grounding_semantics.get("object_type_plurals", {})
        if isinstance(irregular, dict) and isinstance(irregular.get(object_type), str):
            return str(irregular[object_type])
        if object_type.endswith("s"):
            return object_type
        return f"{object_type}s"

    def object_type_from_text(self, text: str) -> str | None:
        normalized = _normalize(text)
        for object_type in self.object_types:
            plural = self.pluralize(object_type)
            if re.search(rf"\b(?:{re.escape(object_type)}|{re.escape(plural)})\b", normalized):
                return object_type
        return None

    def metric_from_text_or_plan(self, text: str, plan: dict[str, Any] | None) -> str | None:
        if plan is not None:
            metric = plan.get("metric")
            return str(metric) if metric else None
        normalized = _normalize(text)
        for metric in self.metrics:
            if re.search(rf"\b{re.escape(metric)}\b", normalized):
                return metric
        return self.default_metric

    def metric_supported(self, metric: str | None) -> bool:
        return metric is not None and metric in set(self.metrics)

    def capability_handle(self, key: str, **values: Any) -> str | None:
        handles = self.operational_context.grounding_semantics.get("capability_handles", {})
        if not isinstance(handles, dict):
            handles = {}
        template = handles.get(key)
        if not isinstance(template, str):
            template = self._default_handle_template(key)
        if not template:
            return None

        object_type = str(values.get("object_type") or self.default_object_type or "object")
        render_values = {
            "object_type": object_type,
            "object_type_plural": self.pluralize(object_type),
            "metric": values.get("metric"),
            "reference": values.get("reference") or "agent",
        }
        try:
            return template.format(**render_values)
        except Exception:  # noqa: BLE001
            return None

    def _default_handle_template(self, key: str) -> str | None:
        defaults = {
            "ranked": "grounding.all_{object_type_plural}.ranked.{metric}.agent",
            "closest": "grounding.closest_{object_type}.{metric}.{reference}",
            "unique": "grounding.unique_{object_type}.color_filter",
            "filter_threshold": "claims.filter.threshold.{metric}",
            "task_go_to_object": "task.go_to_object.{object_type}",
        }
        return defaults.get(key)

    def ranked_handle(self, metric: str | None, *, object_type: str | None = None) -> str | None:
        if not self.metric_supported(metric):
            return None
        return self.capability_handle("ranked", metric=metric, object_type=object_type)

    def filter_handle(self, metric: str | None) -> str | None:
        if not self.metric_supported(metric):
            return None
        return self.capability_handle("filter_threshold", metric=metric)

    def target_handle(self, intent: OperatorIntent) -> str | None:
        selector = intent.target_selector or {}
        relation = selector.get("relation")
        metric = selector.get("distance_metric")
        reference = selector.get("distance_reference")
        object_type = selector.get("object_type") or self.default_object_type
        if relation == "closest":
            if metric is None or reference is None or not self.metric_supported(str(metric)):
                return None
            return self.capability_handle(
                "closest",
                metric=str(metric),
                reference=reference,
                object_type=object_type,
            )
        if selector:
            return self.capability_handle("unique", object_type=object_type)
        target = intent.target or {}
        target_object = target.get("object_type")
        if target_object:
            return self.capability_handle("unique", object_type=target_object)
        return None

    def task_handle(self, task_type: str | None, object_type: str | None) -> str | None:
        if task_type != "go_to_object" or object_type is None:
            return None
        return self.capability_handle("task_go_to_object", object_type=object_type)

    def preservation_signals(self, text: str) -> list[str]:
        normalized = _normalize(text)
        signals: list[str] = []
        checks = {
            "superlative.closest": ("closest", "nearest", "shortest"),
            "superlative.farthest": ("farthest", "furthest", "most distant", "least close"),
            "ordinal": ("first", "second", "third", "fourth", "fifth"),
            "cardinality.all": ("all", "each", "every"),
            "negation": ("not", "except", "other than"),
            "threshold": ("above", "below", "within", "at least", "at most"),
            "reference": ("that", "same", "previous", "last"),
        }
        for signal, terms in checks.items():
            if any(term in normalized for term in terms):
                signals.append(signal)
        for metric in self.metrics:
            if re.search(rf"\b{re.escape(metric)}\b", normalized):
                signals.append(f"metric.{metric}")
        for color in self.attribute_values("color"):
            if re.search(rf"\b{re.escape(color)}\b", normalized):
                signals.append(f"color.{color}")
        object_type = self.object_type_from_text(normalized)
        if object_type is not None:
            signals.append(f"object_type.{object_type}")
        return signals


# Domain factory registry — registered by the domain adapter at init.
_DEFAULT_SEMANTICS_FACTORY: Callable[[], "PlanningSemantics"] | None = None


def register_default_planning_semantics(factory: Callable[[], "PlanningSemantics"]) -> None:
    global _DEFAULT_SEMANTICS_FACTORY
    _DEFAULT_SEMANTICS_FACTORY = factory


def default_planning_semantics() -> "PlanningSemantics":
    if _DEFAULT_SEMANTICS_FACTORY is None:
        raise RuntimeError(
            "No default planning semantics registered. "
            "Call register_default_planning_semantics() before using the planner."
        )
    return _DEFAULT_SEMANTICS_FACTORY()
