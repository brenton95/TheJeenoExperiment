"""Phase 7.59 — Intent Readiness Requirement Matching.

CapabilityMatcher is a pure, substrate-independent function:
  match(intent, registry) -> CapabilityMatchResult

It operates only on CapabilityRegistry entries and OperatorIntent handles.
No MiniGrid, no gymnasium, no sense, no env, no memory.

No-weakening rule: a handle must match exactly. closest_door does not
satisfy ranked_doors; go_to_object does not satisfy pickup. The registry
is checked by exact handle string — no fuzzy prefix matching.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .capability_registry import CapabilityRegistry
    from .schemas import OperatorIntent

VERDICT_EXECUTABLE = "executable"
VERDICT_NEEDS_CLARIFICATION = "needs_clarification"
VERDICT_SYNTHESIZABLE = "synthesizable"
VERDICT_MISSING_SKILLS = "missing_skills"
VERDICT_UNSUPPORTED = "unsupported"
VERDICT_SKIPPED = "skipped"  # no required_capabilities declared — matcher defers


@dataclass
class CapabilityMatchResult:
    verdict: str
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    synthesizable_handles: list[str] = field(default_factory=list)
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict == VERDICT_EXECUTABLE

    def compact(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "matched": self.matched,
            "missing": self.missing,
            "synthesizable": self.synthesizable_handles,
            "detail": self.detail,
        }

    def operator_message(self) -> str:
        if self.verdict == VERDICT_EXECUTABLE:
            return ""
        if self.verdict == VERDICT_MISSING_SKILLS:
            handles = ", ".join(self.missing)
            # Canonical deterministic refusal — matches the `missing_skills` verdict and is
            # the single operator-facing wording used by both the regex/smoke and LLM paths
            # (the arbitrator only decides to refuse; this owns the text).
            return (
                f"MISSING SKILLS: {handles}\n"
                f"I do not have these capabilities yet. "
                f"Use 'what can you do' to see what is available."
            )
        if self.verdict == VERDICT_SYNTHESIZABLE:
            handles = ", ".join(self.synthesizable_handles)
            return (
                f"CAPABILITY NOT YET IMPLEMENTED\n"
                f"synthesizable={handles}\n"
                f"This capability is marked safe to generate but has not been synthesized. "
                f"Synthesis is not yet active (Phase 7.7)."
            )
        if self.verdict == VERDICT_UNSUPPORTED:
            return f"UNSUPPORTED\n{self.detail}"
        if self.verdict == VERDICT_SKIPPED:
            return ""
        return f"CAPABILITY CHECK\nverdict={self.verdict}\n{self.detail}"


class CapabilityMatcher:
    """Deterministic, substrate-independent capability requirement matcher.

    Checks whether every handle in intent.required_capabilities is present
    in the registry and implemented. Applies exact handle matching — no
    capability subsumption, no prefix relaxation.
    """

    def match(
        self,
        intent: "OperatorIntent",
        registry: "CapabilityRegistry",
    ) -> CapabilityMatchResult:
        required = intent.required_capabilities
        if not required:
            return CapabilityMatchResult(
                verdict=VERDICT_SKIPPED,
                detail="No required_capabilities declared — matcher defers to LLM capability_status.",
            )

        matched: list[str] = []
        missing: list[str] = []
        synthesizable_handles: list[str] = []

        for handle in required:
            entry = registry.lookup(handle)
            if entry is None:
                # Handle not in registry at all — hard missing
                missing.append(handle)
            elif entry.implementation_status == "implemented":
                matched.append(handle)
            elif entry.implementation_status in {"synthesizable"} or entry.safe_to_synthesize:
                synthesizable_handles.append(handle)
            else:
                # planned / missing / unsupported in registry
                missing.append(handle)

        if missing:
            return CapabilityMatchResult(
                verdict=VERDICT_MISSING_SKILLS,
                matched=matched,
                missing=missing,
                synthesizable_handles=synthesizable_handles,
                detail=f"Required capabilities not available: {', '.join(missing)}",
            )
        if synthesizable_handles:
            return CapabilityMatchResult(
                verdict=VERDICT_SYNTHESIZABLE,
                matched=matched,
                missing=missing,
                synthesizable_handles=synthesizable_handles,
                detail=(
                    f"Required capabilities are absent but synthesizable: "
                    f"{', '.join(synthesizable_handles)}"
                ),
            )
        return CapabilityMatchResult(
            verdict=VERDICT_EXECUTABLE,
            matched=matched,
            detail="All required capabilities are implemented.",
        )


# Module-level singleton — stateless, safe to reuse
default_matcher = CapabilityMatcher()
