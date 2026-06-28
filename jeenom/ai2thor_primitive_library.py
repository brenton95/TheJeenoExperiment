from __future__ import annotations

from typing import Any

from . import geometry
from .primitive_library import PrimitiveSpec


def ground_all_apples_ranked_euclidean(
    scene: Any,
    filter_dict: dict[str, Any],
) -> list[tuple[float, Any]]:
    object_type = filter_dict.get("object_type", "apple")
    color = filter_dict.get("color")
    exclude_colors = filter_dict.get("exclude_colors") or []
    objects = scene.find(object_type=object_type, color=color)
    if exclude_colors:
        objects = [o for o in objects if o.color not in exclude_colors]
    return sorted(
        [
            (geometry.euclidean(obj.coord, scene.agent_coord), obj)
            for obj in objects
        ],
        key=lambda pair: (pair[0], pair[1].color or "", pair[1].x, pair[1].y),
    )


AI2THOR_GROUNDING_PRIMITIVES: dict[str, PrimitiveSpec] = {
    "all_apples.ranked.euclidean.agent": PrimitiveSpec(
        name="all_apples.ranked.euclidean.agent",
        consumes=("scene.grid_objects", "agent_pose"),
        produces=("ranked_apple_list", "distances"),
        description=(
            "List all visible apples ranked by Euclidean distance from the agent. "
            "Query only — no target is selected and no motion occurs."
        ),
        implementation_status="implemented",
        runtime_kind="python",
        runtime_value="ground_all_apples_ranked_euclidean",
    ),
}
