from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import OperationalContext


def _normalize_text(text: str) -> str:
    normalized = " ".join(text.lower().strip().split())
    normalized = re.sub(r"[?!]+$", "", normalized)
    normalized = re.sub(r"[.,;:]+", " ", normalized)
    return " ".join(normalized.split())


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
        return {
            "verb": match.group("verb"),
            "color": color,
            "object_type": match.group("object_type"),
        }
