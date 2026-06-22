from __future__ import annotations

from .minigrid_domain_helper import MiniGridDomainHelper
from .minigrid_operational_context import MiniGridOperationalContext
from .minigrid_substrate_adapter import MiniGridSubstrateAdapter
from .runtime_package import RuntimePackage


def default_minigrid_domain_helper() -> MiniGridDomainHelper:
    context = MiniGridOperationalContext.default()
    return MiniGridDomainHelper(context)


def build_minigrid_runtime_package(
    *,
    env_id: str,
    render_mode: str,
    observability: str = "partial",
) -> RuntimePackage:
    context = MiniGridOperationalContext.default(env_id=env_id)
    substrate = MiniGridSubstrateAdapter(
        env_id=env_id,
        render_mode=render_mode,
        observability=observability,
        operational_context=context,
    )
    return RuntimePackage(
        substrate=substrate,
        operational_context=context,
        domain_helper=MiniGridDomainHelper(context),
        orpi_manifest=substrate.orpi_manifest(),
    )
