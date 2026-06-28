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


_RANKING_TERMS = frozenset({
    "nearest", "closest", "farthest", "furthest",
    "most distant", "least close",
})


@dataclass(frozen=True)
class Ai2thorDomainHelper:
    """AI2-THOR meaning helper bound to an OperationalContext."""

    operational_context: OperationalContext

    @property
    def object_types(self) -> tuple[str, ...]:
        return tuple(self.operational_context.object_vocabulary or ("apple",))

    @property
    def default_object_type(self) -> str:
        return self.object_types[0] if self.object_types else "apple"

    def object_type_pattern(self) -> str:
        return "|".join(re.escape(t) for t in self.object_types)

    @property
    def supported_colors(self) -> tuple[str, ...]:
        return ()

    @property
    def default_metric(self) -> str:
        closest = self.operational_context.reference_semantics.get("closest", {})
        if isinstance(closest, dict) and isinstance(closest.get("default_metric"), str):
            return closest["default_metric"]
        return "euclidean"

    def normalize_color(self, color: str) -> str:
        return color

    def parse_target_fact(self, normalized: str) -> dict[str, Any] | None:
        return None

    def parse_go_to_object_utterance(self, utterance: str) -> dict[str, str] | None:
        normalized = _normalize_text(utterance)
        object_type_pattern = self.object_type_pattern()
        if not object_type_pattern:
            return None
        match = re.search(
            rf"\b(?P<verb>go to|go the|reach|find|get to|head to|navigate to)\s+"
            rf"(?:the )?(?:(?P<color>[a-z]+) )?(?P<object_type>{object_type_pattern})\b",
            normalized,
        )
        if not match:
            return None
        color = match.group("color") or ""
        if color in _RANKING_TERMS:
            return None
        return {
            "verb": match.group("verb"),
            "color": color,
            "object_type": match.group("object_type"),
        }

    def canonicalize_task_instruction(self, utterance: str) -> str:
        match = self.parse_go_to_object_utterance(utterance)
        if not match:
            return utterance
        verb = match["verb"]
        if verb in {"go the", "head to", "navigate to"}:
            verb = "go to"
        color_part = f"{match['color']} " if match["color"] else ""
        return f"{verb} the {color_part}{match['object_type']}"

    def color_reference_in_utterance(self, normalized: str) -> str | None:
        return None

    def entry_target_dict(self, entry: GroundedObjectEntry) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": entry.object_type,
            "x": entry.x,
            "y": entry.y,
        }
        if entry.color is not None:
            result["color"] = entry.color
        return result

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
        if color:
            return f"{color} {object_type}@({x},{y}) distance={distance}"
        return f"{object_type}@({x},{y}) distance={distance}"

    def task_utterance_for_entry(self, entry: GroundedObjectEntry | dict[str, Any]) -> str:
        if isinstance(entry, dict):
            color = entry.get("color")
            object_type = entry.get("object_type") or entry.get("type") or self.default_object_type
        else:
            color = entry.color
            object_type = entry.object_type
        if color:
            return f"go to the {color} {object_type}"
        return f"go to the {object_type}"

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
        plural = object_type if object_type.endswith("s") else f"{object_type}s"
        lines = [
            f"{plural.upper()} RANKED BY {metric.upper()} DISTANCE FROM AGENT"
        ]
        for i, entry in enumerate(ranked_entries):
            lines.append(f"  {i + 1}. {self.entry_label(entry)}")
        if include_navigation_hint:
            lines.append(
                f"\n(I can navigate to any specific {object_type} - tell me which one.)"
            )
        return "\n".join(lines)

    def bare_color_reference(self, normalized: str) -> str | None:
        return None

    def metric_from_grounding_handle(self, handle: str) -> str:
        metrics = self.operational_context.grounding_semantics.get("distance_metrics", [])
        if not isinstance(metrics, list):
            metrics = []
        for metric in metrics:
            if isinstance(metric, str) and f".{metric}." in handle:
                return metric
        return self.default_metric

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

    def parse_exact_go_to_object_utterance(self, utterance: str) -> dict[str, str] | None:
        normalized = _normalize_text(utterance)
        object_type_pattern = self.object_type_pattern()
        if not object_type_pattern:
            return None
        match = re.match(
            rf"^(?P<verb>go to|reach|find|get to|head to|navigate to)\s+"
            rf"(?:the )?(?:(?P<color>[a-z]+) )?(?P<object_type>{object_type_pattern})$",
            normalized,
        )
        if not match:
            return None
        color = match.group("color") or ""
        if color in _RANKING_TERMS:
            return None
        return {
            "verb": match.group("verb"),
            "color": color,
            "object_type": match.group("object_type"),
        }
