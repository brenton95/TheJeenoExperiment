from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .minigrid_primitive_library import MINIGRID_GROUNDING_PRIMITIVES
from .primitive_library import (
    ACTION_PRIMITIVES,
    CLAIMS_FILTER_PRIMITIVES,
    GROUNDING_PRIMITIVES,
    SENSING_PRIMITIVES,
    TASK_PRIMITIVES,
    PrimitiveSpec as RuntimePrimitiveSpec,
)
from .schemas import PrimitiveManifest, PrimitiveSpec, TargetSelector


def _orpi_postcondition_primitive(layer: str, spec: RuntimePrimitiveSpec) -> str | None:
    if layer != "action":
        return None
    if spec.safety_class == "query":
        return None
    return "sensing.parse_grid_objects"


_SYNTHESIZED_ALIASES = {
    "grounding.closest_door.euclidean.agent": (
        "grounding.all_doors.ranked.euclidean.agent",
    ),
    "grounding.all_doors.ranked.euclidean.agent": (
        "grounding.closest_door.euclidean.agent",
    ),
}


def _runtime_binding(spec: RuntimePrimitiveSpec) -> dict[str, Any] | None:
    binding: dict[str, Any] = {}
    if spec.runtime_kind is not None:
        binding["kind"] = spec.runtime_kind
    if spec.runtime_value is not None:
        binding["value"] = spec.runtime_value
    if spec.required_action_primitives:
        binding["required_action_primitives"] = list(spec.required_action_primitives)
    return binding or None


def _manifest_spec(
    *,
    layer: str,
    source_name: str,
    spec: RuntimePrimitiveSpec,
) -> dict[str, Any]:
    return {
        "name": f"{layer}.{source_name}",
        "primitive_type": layer,
        "layer": layer,
        "description": spec.description,
        "inputs": list(spec.consumes),
        "outputs": list(spec.produces),
        "side_effects": list(spec.side_effects),
        "implementation_status": spec.implementation_status,
        "safe_to_synthesize": spec.safe_to_synthesize,
        "runtime_binding": _runtime_binding(spec),
        "preconditions": list(spec.preconditions),
        "postconditions": list(spec.postconditions),
        "required_claims": list(spec.required_claims),
        "produced_claims": list(spec.produced_claims),
        "units": dict(spec.units),
        "frame_id": spec.frame_id,
        "required_frames": list(spec.required_frames),
        "safety_class": spec.safety_class,
        "authority_level": spec.authority_level,
        "failure_modes": list(spec.failure_modes),
        "validation_hooks": list(spec.validation_hooks),
        "substrate_fingerprint": spec.substrate_fingerprint,
        "postcondition_primitive": _orpi_postcondition_primitive(layer, spec),
    }


def _top_level_task_capability(
    *,
    name: str,
    description: str,
    inputs: list[str],
    outputs: list[str],
    side_effects: list[str],
    implementation_status: str,
    runtime_binding: dict[str, Any] | None,
    safety_class: str = "query",
    authority_level: str = "none",
    validation_hooks: list[str] | None = None,
    failure_modes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "primitive_type": "task",
        "layer": "task",
        "description": description,
        "inputs": inputs,
        "outputs": outputs,
        "side_effects": side_effects,
        "implementation_status": implementation_status,
        "safe_to_synthesize": False,
        "runtime_binding": runtime_binding,
        "preconditions": list(inputs),
        "postconditions": list(outputs),
        "required_claims": list(inputs),
        "produced_claims": list(outputs),
        "units": {},
        "frame_id": None,
        "required_frames": [],
        "safety_class": safety_class,
        "authority_level": authority_level,
        "failure_modes": failure_modes or [],
        "validation_hooks": validation_hooks or [],
        "substrate_fingerprint": None,
        "postcondition_primitive": None,
    }


def minigrid_manifest_dict() -> dict[str, Any]:
    primitives: list[dict[str, Any]] = [
        _top_level_task_capability(
            name="task.go_to_object.door",
            description="Run the known go_to_object recipe for a grounded door target.",
            inputs=["target.color", "target.object_type", "target_location"],
            outputs=["task_complete", "execution_report"],
            side_effects=["moves_agent"],
            implementation_status="implemented",
            runtime_binding={
                "kind": "understanding_recipe",
                "value": "go_to_object",
                "procedure_steps": [
                    "locate_object",
                    "navigate_to_object",
                    "verify_adjacent",
                    "done",
                ],
            },
            safety_class="actuation",
            authority_level="operator",
            validation_hooks=["minigrid_task_preflight"],
            failure_modes=["no_path_found", "target_missing"],
        ),
        _top_level_task_capability(
            name="task.go_to_object.key",
            description="Run the known go_to_object recipe for a grounded key target.",
            inputs=["target.color", "target.object_type", "target_location"],
            outputs=["task_complete", "execution_report"],
            side_effects=["moves_agent"],
            implementation_status="implemented",
            runtime_binding={
                "kind": "understanding_recipe",
                "value": "go_to_object",
                "procedure_steps": [
                    "locate_object",
                    "navigate_to_object",
                    "verify_adjacent",
                    "done",
                ],
            },
        ),
        _top_level_task_capability(
            name="task.pickup.key",
            description=(
                "Pick up a key task. Low-level pickup exists, but key grounding, "
                "inventory semantics, and task recipe are not implemented."
            ),
            inputs=["target.object_type", "key_location"],
            outputs=["inventory"],
            side_effects=["changes_environment", "changes_inventory"],
            implementation_status="unsupported",
            runtime_binding=None,
        ),
        _top_level_task_capability(
            name="task.open_or_unlock.door",
            description=(
                "Open or unlock a door task. Low-level toggle exists, but door "
                "state/inventory semantics and task recipe are not implemented."
            ),
            inputs=["target.object_type", "door_location", "inventory"],
            outputs=["door_state"],
            side_effects=["changes_environment"],
            implementation_status="unsupported",
            runtime_binding=None,
        ),
    ]
    _all_grounding = {**GROUNDING_PRIMITIVES, **MINIGRID_GROUNDING_PRIMITIVES}
    for layer, library in (
        ("task", TASK_PRIMITIVES),
        ("grounding", _all_grounding),
        ("sensing", SENSING_PRIMITIVES),
        ("action", ACTION_PRIMITIVES),
        ("claims", CLAIMS_FILTER_PRIMITIVES),
    ):
        primitives.extend(
            _manifest_spec(layer=layer, source_name=name, spec=spec)
            for name, spec in sorted(library.items())
        )
    return {
        "name": "minigrid_primitive_registry_v1",
        "primitives": primitives,
    }


class CapabilityRegistry:
    def __init__(self, manifest: PrimitiveManifest) -> None:
        self.manifest = manifest
        self._by_name = {spec.name: spec for spec in manifest.primitives}
        self._synthesized_callables: dict[str, Any] = {}

    @classmethod
    def minigrid_default(cls) -> CapabilityRegistry:
        return cls(PrimitiveManifest.from_dict(minigrid_manifest_dict()))

    def compact_summary(self) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for spec in self.manifest.primitives:
            grouped.setdefault(spec.layer, []).append(
                {
                    "name": spec.name,
                    "layer": spec.layer,
                    "status": spec.implementation_status,
                    "safe_to_synthesize": spec.safe_to_synthesize,
                    "inputs": list(spec.inputs),
                    "outputs": list(spec.outputs),
                    "side_effects": list(spec.side_effects),
                    "runtime_binding": spec.runtime_binding,
                    "description": spec.description,
                    "required_claims": list(spec.required_claims),
                    "produced_claims": list(spec.produced_claims),
                    "frame_id": spec.frame_id,
                    "required_frames": list(spec.required_frames),
                    "safety_class": spec.safety_class,
                    "authority_level": spec.authority_level,
                    "failure_modes": list(spec.failure_modes),
                    "validation_hooks": list(spec.validation_hooks),
                    "substrate_fingerprint": spec.substrate_fingerprint,
                    "postcondition_primitive": spec.postcondition_primitive,
                    "mode": spec.mode,
                    "cadence": spec.cadence,
                    "invariant_level": spec.invariant_level,
                }
            )
        return {"name": self.manifest.name, "primitives": grouped}

    def primitive(self, name: str) -> PrimitiveSpec | None:
        return self._by_name.get(name)

    def lookup(self, handle: str) -> PrimitiveSpec | None:
        """Exact handle lookup — no weakening, no prefix relaxation."""
        return self._by_name.get(handle)

    def primitive_names(self, *, layer: str | None = None) -> list[str]:
        return sorted(
            spec.name
            for spec in self.manifest.primitives
            if layer is None or spec.layer == layer
        )

    def candidates_for_postcondition(
        self,
        postcondition: str,
        *,
        orpi_manifest: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Return primitive/procedure candidates that declare a postcondition."""

        candidates: list[dict[str, Any]] = []
        for spec in self.manifest.primitives:
            if postcondition in set(spec.postconditions) | set(spec.outputs):
                candidates.append(
                    {
                        "kind": "primitive",
                        "name": spec.name,
                        "postconditions": list(spec.postconditions or spec.outputs),
                        "provenance": "registry",
                    }
                )
        if orpi_manifest is not None:
            for procedure in getattr(orpi_manifest, "bundled_procedures", []):
                if postcondition in procedure.declared_postconditions:
                    candidates.append(
                        {
                            "kind": "procedure",
                            "name": procedure.name,
                            "postconditions": list(procedure.declared_postconditions),
                            "preconditions": list(procedure.declared_preconditions),
                            "provenance": procedure.provenance,
                            "steps": procedure.primitive_step_names(),
                        }
                    )
        return candidates

    def expand_procedure_candidate(
        self,
        procedure_name: str,
        *,
        orpi_manifest: Any,
    ) -> list[str]:
        """Expand a bundled ORPI procedure to primitive handles before execution."""

        for procedure in getattr(orpi_manifest, "bundled_procedures", []):
            if procedure.name == procedure_name:
                return procedure.primitive_step_names()
        return []

    def register_synthesized(self, handle: str, fn: Any) -> bool:
        """Promote a synthesized primitive from synthesizable to implemented.

        Updates the spec in-place so lookup() and compact_summary() reflect
        the new status immediately. Stores the callable for dispatch.
        Returns True if the primitive was known and synthesizable, False otherwise.
        """
        spec = self._by_name.get(handle)
        if spec is None:
            return False
        if spec.implementation_status not in {"synthesizable"}:
            return False
        from .primitive_library import PrimitiveSpec as RuntimeSpec
        promoted = RuntimeSpec(
            name=spec.name,
            consumes=tuple(spec.inputs),
            produces=tuple(spec.outputs),
            description=spec.description + " [synthesized]",
            side_effects=tuple(spec.side_effects),
            implementation_status="implemented",
            safe_to_synthesize=False,
            runtime_kind="python_synthesized",
            runtime_value=handle,
        )
        from .schemas import PrimitiveSpec as SchemaSpec
        promoted_schema = SchemaSpec(
            name=promoted.name,
            primitive_type=spec.primitive_type,
            layer=spec.layer,
            description=promoted.description,
            inputs=list(promoted.consumes),
            outputs=list(promoted.produces),
            side_effects=list(promoted.side_effects),
            implementation_status="implemented",
            safe_to_synthesize=False,
            runtime_binding={"kind": "python_synthesized", "value": handle},
            postcondition_primitive=spec.postcondition_primitive,
            mode=spec.mode,
            cadence=spec.cadence,
            invariant_level=spec.invariant_level,
        )
        self._promote_synthesized_schema(handle, promoted_schema, fn)
        for alias in _SYNTHESIZED_ALIASES.get(handle, ()):
            alias_spec = self._by_name.get(alias)
            if alias_spec is not None and alias_spec.implementation_status == "synthesizable":
                alias_schema = SchemaSpec(
                    name=alias,
                    primitive_type=alias_spec.primitive_type,
                    layer=alias_spec.layer,
                    description=alias_spec.description + " [synthesized alias]",
                    inputs=list(alias_spec.inputs),
                    outputs=list(alias_spec.outputs),
                    side_effects=list(alias_spec.side_effects),
                    implementation_status="implemented",
                    safe_to_synthesize=False,
                    runtime_binding={"kind": "python_synthesized", "value": handle},
                    postcondition_primitive=alias_spec.postcondition_primitive,
                    mode=alias_spec.mode,
                    cadence=alias_spec.cadence,
                    invariant_level=alias_spec.invariant_level,
                )
                self._promote_synthesized_schema(alias, alias_schema, fn)
        return True

    def _promote_synthesized_schema(
        self,
        handle: str,
        schema: PrimitiveSpec,
        fn: Any,
    ) -> None:
        self._by_name[handle] = schema
        self._synthesized_callables[handle] = fn
        # Keep manifest.primitives in sync so compact_summary() and help_text()
        # immediately reflect the promoted status.
        for i, p in enumerate(self.manifest.primitives):
            if p.name == handle:
                self.manifest.primitives[i] = schema
                break

    def get_synthesized_callable(self, handle: str) -> Any | None:
        """Return the validated callable for a synthesized primitive, or None."""
        return self._synthesized_callables.get(handle)

    def register_dynamic(
        self,
        handle: str,
        description: str,
        fn: Any,
        *,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        side_effects: list[str] | None = None,
        safety_class: str = "query",
        authority_level: str = "none",
        validation_hooks: list[str] | None = None,
    ) -> bool:
        """Register a brand-new synthesized primitive that was not pre-declared.

        Unlike register_synthesized, this creates the spec from scratch rather than
        promoting an existing synthesizable entry. Used when the arbitrator proposes
        a handle that does not exist in the registry yet.
        Returns True on success, False if the handle already exists.
        """
        if handle in self._by_name:
            return False
        from .schemas import PrimitiveSpec as SchemaSpec
        parts = handle.split(".")
        layer = parts[0] if parts else "grounding"
        new_spec = SchemaSpec(
            name=handle,
            primitive_type=layer,
            layer=layer,
            description=description + " [synthesized]",
            inputs=inputs or ["scene.door_candidates", "agent_pose"],
            outputs=outputs or ["grounded_target", "distance"],
            side_effects=side_effects or [],
            implementation_status="implemented",
            safe_to_synthesize=False,
            runtime_binding={"kind": "python_synthesized", "value": handle},
            safety_class=safety_class,
            authority_level=authority_level,
            validation_hooks=validation_hooks or [],
            mode="deterministic",
        )
        self._by_name[handle] = new_spec
        self.manifest.primitives.append(new_spec)
        self._synthesized_callables[handle] = fn
        return True

    def readiness_for_selector(self, selector: dict[str, Any] | None) -> dict[str, Any]:
        if selector is None:
            return {
                "status": "unsupported",
                "layer": "grounding",
                "primitive": None,
                "reason": "No target selector was provided.",
            }
        raw_object_type = selector.get("object_type")
        target_selector = TargetSelector.from_dict(
            selector,
            object_types=(
                [raw_object_type]
                if isinstance(raw_object_type, str)
                else []
            ),
        )
        primitive_name = self._primitive_name_for_selector(target_selector)
        if primitive_name is None:
            return {
                "status": "unsupported",
                "layer": "grounding",
                "primitive": None,
                "reason": "No matching grounding primitive is declared for this selector.",
            }
        return self._readiness_for_primitive(primitive_name)

    def readiness_for_task(
        self,
        *,
        task_type: str | None,
        object_type: str | None,
    ) -> dict[str, Any]:
        if task_type is None or object_type is None:
            return {
                "status": "unsupported",
                "layer": "task",
                "primitive": None,
                "reason": (
                    f"No task primitive supports task_type={task_type} "
                    f"object_type={object_type}."
                ),
            }
        handle_task_type = (
            "open_or_unlock"
            if task_type in {"open", "unlock", "open_or_unlock"}
            else task_type
        )
        primitive_name = f"task.{handle_task_type}.{object_type}"
        if self.lookup(primitive_name) is not None:
            return self._readiness_for_primitive(primitive_name)
        return {
            "status": "unsupported",
            "layer": "task",
            "primitive": None,
            "reason": f"No task primitive supports task_type={task_type} object_type={object_type}.",
        }

    def help_text(self) -> str:
        summary = self.compact_summary()["primitives"]
        executable_tasks = [
            item["name"]
            for item in summary.get("task", [])
            if item["status"] == "implemented" and item["name"].startswith("task.")
        ]
        executable_grounding = [
            item["name"]
            for item in summary.get("grounding", [])
            if item["status"] == "implemented"
        ]
        sensing = [
            item["name"]
            for item in summary.get("sensing", [])
            if item["status"] == "implemented"
        ]
        actions = [
            item["name"]
            for item in summary.get("action", [])
            if item["status"] == "implemented"
        ]
        claims = [
            item["name"]
            for item in summary.get("claims", [])
            if item["status"] == "implemented"
        ]
        synthesizable = [
            item["name"]
            for items in summary.values()
            for item in items
            if item["status"] == "synthesizable" or item["safe_to_synthesize"]
        ]
        unsupported = [
            item["name"]
            for items in summary.values()
            for item in items
            if item["status"] == "unsupported"
        ]
        return (
            "CAPABILITIES\n"
            f"registry={self.manifest.name}\n"
            f"tasks={', '.join(executable_tasks) or 'none'}\n"
            f"grounding={', '.join(executable_grounding) or 'none'}\n"
            f"sensing={', '.join(sensing) or 'none'}\n"
            f"actions={', '.join(actions) or 'none'}\n"
            f"claims={', '.join(claims) or 'none'}\n"
            f"synthesizable={', '.join(synthesizable) or 'none'}\n"
            f"unsupported={', '.join(unsupported) or 'none'}"
        )

    def _readiness_for_primitive(self, primitive_name: str) -> dict[str, Any]:
        spec = self.primitive(primitive_name)
        if spec is None:
            return {
                "status": "missing_primitive",
                "layer": None,
                "primitive": primitive_name,
                "reason": "Primitive is not present in the registry.",
            }
        if spec.implementation_status == "implemented":
            return {
                "status": "executable",
                "layer": spec.layer,
                "primitive": spec.name,
                "reason": spec.description,
            }
        if spec.implementation_status == "synthesizable" or spec.safe_to_synthesize:
            return {
                "status": "synthesizable_missing_primitive",
                "layer": spec.layer,
                "primitive": spec.name,
                "reason": spec.description,
            }
        if spec.implementation_status in {"missing", "planned"}:
            return {
                "status": "missing_primitive",
                "layer": spec.layer,
                "primitive": spec.name,
                "reason": spec.description,
            }
        return {
            "status": "unsupported",
            "layer": spec.layer,
            "primitive": spec.name,
            "reason": spec.description,
        }

    def _primitive_name_for_selector(self, selector: TargetSelector) -> str | None:
        if selector.relation == "closest":
            if selector.distance_metric is None or selector.distance_reference is None:
                return None
            return (
                f"grounding.closest_{selector.object_type}.{selector.distance_metric}."
                f"{selector.distance_reference}"
            )
        if selector.relation in {None, "unique"}:
            return f"grounding.unique_{selector.object_type}.color_filter"
        return None

    def ranked_metric_handles(self) -> dict[str, str]:
        """Return {metric_name: handle} for all registered ranked-grounding primitives.

        Identifies primitives by the typed handle segment ``.ranked.<metric>.``
        and extracts the metric name. Output claim names are intentionally not
        used because legacy MiniGrid claims still contain ``door`` vocabulary.
        """
        result: dict[str, str] = {}
        for spec in self.manifest.primitives:
            if spec.layer != "grounding":
                continue
            parts = spec.name.split(".")
            try:
                idx = parts.index("ranked")
                if idx + 1 < len(parts) - 1:
                    metric = parts[idx + 1]
                    result[metric] = spec.name
            except ValueError:
                continue
        return result

    def ranked_handle_for(self, metric_name: str) -> str:
        """Return the capability handle for a metric name.

        For known metrics returns the registered handle.
        For new operator-defined metrics derives the handle by following the
        naming convention of existing ranked primitives, so the convention is
        defined once (in primitive_library) not scattered across callers.
        """
        known = self.ranked_metric_handles()
        if metric_name in known:
            return known[metric_name]
        for handle in known.values():
            parts = handle.split(".")
            try:
                idx = parts.index("ranked")
                prefix = ".".join(parts[: idx + 1])
                suffix = ".".join(parts[idx + 2 :])
                return f"{prefix}.{metric_name}.{suffix}"
            except ValueError:
                continue
        return f"grounding.ranked.{metric_name}.agent"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.manifest.name,
            "primitives": [asdict(spec) for spec in self.manifest.primitives],
        }
