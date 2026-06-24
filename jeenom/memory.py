from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import ClaimRecord, MemoryUpdate, SceneModel


class OperationalMemory:
    def __init__(self, root: Path | None = None):
        project_root = root or Path(__file__).resolve().parent.parent
        self.memory_dir = project_root / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_path = self.memory_dir / "knowledge.yaml"

        self.understanding = {
            "go_to_object": {
                "description": "Navigate to a target object described by the operator or mission.",
                "default_recipe": [
                    "locate_object",
                    "navigate_to_object",
                    "verify_adjacent",
                    "done",
                ],
            },
            "search_for_object": {
                "description": "Search for a target object and navigate to it.",
                "default_recipe": [
                    "locate_object",
                    "navigate_to_object",
                    "verify_adjacent",
                    "done",
                ],
            },
        }

        self.embodied_intelligence = {
            "minigrid": {
                "substrate": "MiniGrid",
                "action_space": "discrete",
                "observation_space": "fully_observable_grid_encoding",
                "notes": "The runtime executes validated action primitives deterministically.",
            }
        }

        self.knowledge = {
            "target_color": None,
            "target_type": None,
            "delivery_target": None,
            "last_task_type": None,
            "last_instruction": None,
            "primitive_definitions": {},
        }
        self.knowledge.update(self._load_knowledge())
        self._normalize_target_knowledge()

        self.scene_model: SceneModel | None = None

        # The single mission-scoped claim store. Both the per-task Cortex belief
        # loop and the station's RepresentationStore read/write this one dict, so
        # belief survives across run_episode task runs and is cleared only on the
        # typed-reset path (reset_episode(clear_reference_context=True)).
        self.claims: dict[str, ClaimRecord] = {}

        self.episodic_memory = {
            "known_target_location": None,
            "last_world_sample": None,
            "last_trace_length": 0,
            "last_target": None,
            "last_task": None,
            "last_successful_instruction": None,
        }

    def resolve_target_params(self, task_params: dict[str, Any]) -> dict[str, Any]:
        resolved = dict(task_params)
        delivery_target = self.knowledge.get("delivery_target")
        if isinstance(delivery_target, dict):
            if resolved.get("color") is None and delivery_target.get("color") is not None:
                resolved["color"] = delivery_target["color"]
            if resolved.get("object_type") is None and delivery_target.get("object_type") is not None:
                resolved["object_type"] = delivery_target["object_type"]
        if resolved.get("color") is None and self.knowledge.get("target_color") is not None:
            resolved["color"] = self.knowledge["target_color"]
        if resolved.get("object_type") is None and self.knowledge.get("target_type") is not None:
            resolved["object_type"] = self.knowledge["target_type"]
        return resolved

    def update_knowledge(self, key: str, value: Any, persist: bool = True) -> None:
        self.knowledge[key] = value
        if key in {"target_color", "target_type", "delivery_target"}:
            self._normalize_target_knowledge(prefer_delivery_target=(key == "delivery_target"))
        if persist:
            self._persist_knowledge()

    def update_scene_model(self, model: SceneModel) -> None:
        self.scene_model = model

    def update_episodic_memory(self, key: str, value: Any) -> None:
        self.episodic_memory[key] = value

    def reset_episode(self, *, clear_reference_context: bool = True) -> None:
        self.scene_model = None
        # Belief is mission-scoped: task admission (clear_reference_context=False)
        # preserves the claim store; only a typed reset (the episode boundary)
        # clears it.
        if clear_reference_context:
            self.claims = {}
        last_target = None
        last_task = None
        last_successful_instruction = None
        if not clear_reference_context:
            last_target = self.episodic_memory.get("last_target")
            last_task = self.episodic_memory.get("last_task")
            last_successful_instruction = self.episodic_memory.get("last_successful_instruction")

        self.episodic_memory = {
            "known_target_location": None,
            "last_world_sample": None,
            "last_trace_length": 0,
            "last_target": last_target,
            "last_task": last_task,
            "last_successful_instruction": last_successful_instruction,
        }

    def apply_memory_updates(self, updates: list[MemoryUpdate]) -> None:
        should_persist = False
        for update in updates:
            if update.scope == "knowledge":
                self.knowledge[update.key] = update.value
                should_persist = True
            elif update.scope == "episodic_memory":
                self.episodic_memory[update.key] = update.value
            else:
                raise ValueError(f"Unknown memory update scope: {update.scope}")

        if should_persist:
            self._normalize_target_knowledge()
            self._persist_knowledge()

    def serializable_snapshot(self) -> dict[str, Any]:
        return {
            "understanding": self.understanding,
            "embodied_intelligence": self.embodied_intelligence,
            "knowledge": self.knowledge,
            "episodic_memory": self.episodic_memory,
        }

    def _load_knowledge(self) -> dict[str, Any]:
        if not self.knowledge_path.exists():
            return {}

        loaded: dict[str, Any] = {}
        for line in self.knowledge_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or ":" not in stripped:
                continue
            key, raw_value = stripped.split(":", 1)
            value_text = raw_value.strip()
            try:
                loaded[key.strip()] = json.loads(value_text)
            except json.JSONDecodeError:
                loaded[key.strip()] = value_text or None
        return loaded

    def _normalize_target_knowledge(self, prefer_delivery_target: bool = True) -> None:
        delivery_target = self.knowledge.get("delivery_target")
        if isinstance(delivery_target, dict) and prefer_delivery_target:
            color = delivery_target.get("color")
            object_type = delivery_target.get("object_type")
            if color is not None:
                self.knowledge["target_color"] = color
            if object_type is not None:
                self.knowledge["target_type"] = object_type
            return

        color = self.knowledge.get("target_color")
        object_type = self.knowledge.get("target_type")
        if color is None and object_type is None:
            self.knowledge["delivery_target"] = None
            return
        self.knowledge["delivery_target"] = {
            "color": color,
            "object_type": object_type,
        }

    def _persist_knowledge(self) -> None:
        lines = [
            f"{key}: {json.dumps(self.knowledge.get(key))}"
            for key in [
                "target_color",
                "target_type",
                "delivery_target",
                "last_task_type",
                "last_instruction",
                "primitive_definitions",
            ]
        ]
        self.knowledge_path.write_text("\n".join(lines) + "\n")
