from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from .schemas import GroundedObjectEntry, OperationalContext


def _normalize_text(text: str) -> str:
    normalized = " ".join(text.lower().strip().split())
    normalized = re.sub(r"[?!]+$", "", normalized)
    normalized = re.sub(r"[.,;:]+", " ", normalized)
    return " ".join(normalized.split())


@dataclass(frozen=True)
class MiniGridDomainHelper:
    """MiniGrid meaning helper bound to an OperationalContext."""

    operational_context: OperationalContext

    DEFAULT_COLORS = ("red", "green", "blue", "yellow", "purple", "grey", "gray")

    @property
    def object_types(self) -> tuple[str, ...]:
        return tuple(self.operational_context.object_vocabulary)

    @property
    def supported_colors(self) -> tuple[str, ...]:
        semantics = self.operational_context.grounding_semantics
        values = (
            semantics.get("attribute_values", {})
            if isinstance(semantics.get("attribute_values"), dict)
            else {}
        )
        color_values = values.get("color", self.DEFAULT_COLORS)
        aliases = (
            semantics.get("attribute_aliases", {})
            if isinstance(semantics.get("attribute_aliases"), dict)
            else {}
        )
        color_aliases = aliases.get("color", {})
        if not isinstance(color_values, list):
            color_values = list(self.DEFAULT_COLORS)
        if not isinstance(color_aliases, dict):
            color_aliases = {}
        ordered: list[str] = []
        for color in [*color_values, *color_aliases.keys()]:
            if isinstance(color, str) and color not in ordered:
                ordered.append(color)
        return tuple(ordered or self.DEFAULT_COLORS)

    @property
    def default_object_type(self) -> str:
        return self.object_types[0] if self.object_types else "object"

    def object_type_pattern(self) -> str:
        return "|".join(re.escape(object_type) for object_type in self.object_types)

    def pluralize_object_type(self, object_type: str) -> str:
        plurals = self.operational_context.grounding_semantics.get(
            "object_type_plurals",
            {},
        )
        if isinstance(plurals, dict) and isinstance(plurals.get(object_type), str):
            return str(plurals[object_type])
        return object_type if object_type.endswith("s") else f"{object_type}s"

    @property
    def default_metric(self) -> str:
        closest = self.operational_context.reference_semantics.get("closest", {})
        if isinstance(closest, dict) and isinstance(closest.get("default_metric"), str):
            return closest["default_metric"]
        return "manhattan"

    def normalize_color(self, color: str) -> str:
        aliases = self.operational_context.grounding_semantics.get("attribute_aliases", {})
        color_aliases = aliases.get("color", {}) if isinstance(aliases, dict) else {}
        if isinstance(color_aliases, dict) and color in color_aliases:
            return str(color_aliases[color])
        return "grey" if color == "gray" else color

    def color_pattern(self) -> str:
        return "|".join(re.escape(color) for color in self.supported_colors)

    def parse_target_fact(self, normalized: str) -> dict[str, Any] | None:
        object_type_pattern = self.object_type_pattern()
        if not object_type_pattern:
            return None
        color_pattern = self.color_pattern()
        patterns = [
            rf"^(?:please )?(?:your |the |my |our )?delivery target is (?:the )?(?P<color>{color_pattern}) (?P<object_type>{object_type_pattern})$",
            rf"^(?:please )?(?:the )?(?P<color>{color_pattern}) (?P<object_type>{object_type_pattern}) is (?:your |the |my |our )?delivery target$",
            rf"^(?:please )?target is (?:the )?(?P<color>{color_pattern}) (?P<object_type>{object_type_pattern})$",
            rf"^(?:please )?remember (?:that )?(?:the )?(?P<color>{color_pattern}) (?P<object_type>{object_type_pattern})$",
            rf"^(?:please )?set (?:the )?delivery target to (?:the )?(?P<color>{color_pattern}) (?P<object_type>{object_type_pattern})$",
            rf"^(?:please )?use (?:the )?(?P<color>{color_pattern}) (?P<object_type>{object_type_pattern}) as (?:your |the |my |our )?delivery target$",
        ]
        for pattern in patterns:
            match = re.match(pattern, normalized)
            if match:
                color = self.normalize_color(match.group("color"))
                object_type = match.group("object_type")
                return {
                    "target_color": color,
                    "target_type": object_type,
                    "delivery_target": {
                        "color": color,
                        "object_type": object_type,
                    },
                }
        return None

    def parse_go_to_object_utterance(self, utterance: str) -> dict[str, str] | None:
        normalized = _normalize_text(utterance)
        object_type_pattern = self.object_type_pattern()
        if not object_type_pattern:
            return None
        match = re.search(
            rf"\b(?P<verb>go to|go the|reach|find|get to|head to|navigate to)\s+"
            rf"(?:the )?(?P<color>{self.color_pattern()}) (?P<object_type>{object_type_pattern})\b",
            normalized,
        )
        if not match:
            return None
        return {
            "verb": match.group("verb"),
            "color": self.normalize_color(match.group("color")),
            "object_type": match.group("object_type"),
        }

    def parse_exact_go_to_object_utterance(self, utterance: str) -> dict[str, str] | None:
        normalized = _normalize_text(utterance)
        object_type_pattern = self.object_type_pattern()
        if not object_type_pattern:
            return None
        match = re.match(
            rf"^(?P<verb>go to|reach|find|get to|head to|navigate to)\s+"
            rf"(?:the )?(?P<color>{self.color_pattern()}) (?P<object_type>{object_type_pattern})$",
            normalized,
        )
        if not match:
            return None
        return {
            "verb": match.group("verb"),
            "color": self.normalize_color(match.group("color")),
            "object_type": match.group("object_type"),
        }

    def canonicalize_task_instruction(self, utterance: str) -> str:
        match = self.parse_go_to_object_utterance(utterance)
        if not match:
            return utterance
        verb = match["verb"]
        if verb in {"go the", "head to", "navigate to"}:
            verb = "go to"
        return f"{verb} the {match['color']} {match['object_type']}"

    def color_reference_in_utterance(self, normalized: str) -> str | None:
        object_type_pattern = self.object_type_pattern()
        if not object_type_pattern:
            return None
        match = re.search(
            rf"\b(?P<color>{self.color_pattern()})\s+(?:one|{object_type_pattern})\b",
            normalized,
        )
        if match:
            return self.normalize_color(match.group("color"))
        return None

    def bare_color_reference(self, normalized: str) -> str | None:
        match = re.search(rf"\b(?P<color>{self.color_pattern()})\b", normalized)
        if not match:
            return None
        return self.normalize_color(match.group("color"))

    def color_answer(self, normalized: str) -> str | None:
        object_type_pattern = self.object_type_pattern()
        match = re.match(
            rf"^(?:the )?(?P<color>{self.color_pattern()})(?: one| {object_type_pattern})?$",
            normalized,
        )
        if not match:
            return None
        return self.normalize_color(match.group("color"))

    def is_metric_answer(self, normalized: str, metric: str) -> bool:
        return normalized in {
            metric,
            f"use {metric}",
            f"{metric} distance",
            f"by {metric} distance",
            f"use {metric} distance",
        }

    def metric_from_grounding_handle(self, handle: str) -> str:
        metrics = self.operational_context.grounding_semantics.get("distance_metrics", [])
        if not isinstance(metrics, list):
            metrics = []
        for metric in metrics:
            if isinstance(metric, str) and f".{metric}." in handle:
                return metric
        return self.default_metric

    def entry_target_dict(self, entry: GroundedObjectEntry) -> dict[str, Any]:
        return {
            "type": entry.object_type,
            "color": entry.color,
            "x": entry.x,
            "y": entry.y,
        }

    def entry_label(self, entry: GroundedObjectEntry | dict[str, Any]) -> str:
        if isinstance(entry, dict):
            color = entry.get("color")
            object_type = entry.get("object_type") or entry.get("type") or self.default_object_type
            x = entry.get("x")
            y = entry.get("y")
            distance = entry.get("distance")
        else:
            color = entry.color
            object_type = entry.object_type
            x = entry.x
            y = entry.y
            distance = entry.distance
        return f"{color} {object_type}@({x},{y}) distance={distance}"

    def task_utterance_for_entry(self, entry: GroundedObjectEntry | dict[str, Any]) -> str:
        if isinstance(entry, dict):
            color = entry.get("color")
            object_type = entry.get("object_type") or entry.get("type") or self.default_object_type
        else:
            color = entry.color
            object_type = entry.object_type
        return f"go to the {color} {object_type}"

    def format_ranked_objects_from_entries(
        self,
        entries: Iterable[GroundedObjectEntry | dict[str, Any]],
        *,
        metric: str,
        include_navigation_hint: bool = True,
    ) -> str:
        ranked_entries = list(entries)
        first = ranked_entries[0] if ranked_entries else None
        if isinstance(first, dict):
            object_type = (
                first.get("object_type")
                or first.get("type")
                or self.default_object_type
            )
        elif first is not None:
            object_type = first.object_type
        else:
            object_type = self.default_object_type
        object_type = str(object_type)
        plural = self.pluralize_object_type(object_type)
        lines = [
            f"{plural.upper()} RANKED BY {metric.upper()} DISTANCE FROM AGENT"
        ]
        for i, entry in enumerate(ranked_entries):
            lines.append(f"  {i + 1}. {self.entry_label(entry)}")
        if include_navigation_hint:
            lines.append(
                f"\n(I can navigate to any specific {object_type} - tell me which color.)"
            )
        return "\n".join(lines)

    def format_ranked_doors_from_entries(
        self,
        entries: Iterable[GroundedObjectEntry | dict[str, Any]],
        *,
        metric: str,
        include_navigation_hint: bool = True,
    ) -> str:
        """Backward-compatible alias for the former MiniGrid-specific name."""
        return self.format_ranked_objects_from_entries(
            entries,
            metric=metric,
            include_navigation_hint=include_navigation_hint,
        )

    def format_color_plan_answer(
        self,
        *,
        color: str,
        matches: list[GroundedObjectEntry],
        answer_fields: set[str],
    ) -> str:
        if not matches:
            if "exists" in answer_fields:
                return (
                    "GROUNDING ANSWER\n"
                    "exists=false\n"
                    f"color={color}\n"
                    f"object_type={self.default_object_type}"
                )
            return f"No matching {color} {self.default_object_type} found."
        if "exists" in answer_fields and "distance" not in answer_fields:
            return f"GROUNDING ANSWER\nexists=true\ntarget={self.entry_label(matches[0])}"
        if "distance" in answer_fields:
            lines = ["GROUNDING ANSWER"]
            for entry in matches:
                lines.append(f"target={self.entry_label(entry)}")
            return "\n".join(lines)
        return "GROUNDING ANSWER\n" + "\n".join(
            f"target={self.entry_label(entry)}" for entry in matches
        )
