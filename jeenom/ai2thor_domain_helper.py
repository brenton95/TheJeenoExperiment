from __future__ import annotations

from dataclasses import dataclass

from .schemas import OperationalContext


@dataclass(frozen=True)
class Ai2thorDomainHelper:
    """AI2-THOR meaning helper bound to an OperationalContext.

    Skeleton for the boundary-test spike: only enough surface to satisfy
    RuntimePackage's `domain_helper.operational_context is operational_context`
    binding check. Parsing/canonicalization behaviour (modelled on
    MiniGridDomainHelper) is out of scope for this plan.
    """

    operational_context: OperationalContext

    @property
    def object_types(self) -> tuple[str, ...]:
        return tuple(self.operational_context.object_vocabulary or ("apple",))

    @property
    def default_object_type(self) -> str:
        return self.object_types[0] if self.object_types else "apple"
