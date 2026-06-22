from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Observation:
    raw: dict[str, Any]
    step_count: int


class MiniGridAdapter:
    def __init__(self, env):
        self.env = env
        self.obs: dict[str, Any] | None = None
        self.info: dict[str, Any] | None = None
        self.step_count = 0

    def reset(self, seed: int | None = None) -> Observation:
        self.obs, self.info = self.env.reset(seed=seed)
        self.step_count = 0
        self.obs = self._annotate_observation(self.obs)
        return Observation(raw=self.obs, step_count=self.step_count)

    def observe(self) -> Observation:
        if self.obs is None:
            raise RuntimeError("Environment has not been reset yet.")
        return Observation(raw=self.obs, step_count=self.step_count)

    def list_grid_objects(self) -> list[dict[str, Any]]:
        raw_env = self.env.unwrapped
        objects: list[dict[str, Any]] = []
        for x in range(raw_env.width):
            for y in range(raw_env.height):
                cell = raw_env.grid.get(x, y)
                if cell is None:
                    continue
                cell_type = getattr(cell, "type", None)
                if cell_type is None:
                    continue
                objects.append(
                    {
                        "type": cell_type,
                        "color": getattr(cell, "color", None),
                        "x": x,
                        "y": y,
                    }
                )
        return objects

    def find_matching_objects(
        self,
        object_type: str | None = None,
        color: str | None = None,
    ) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for obj in self.list_grid_objects():
            if object_type is not None and obj["type"] != object_type:
                continue
            if color is not None and obj["color"] != color:
                continue
            matches.append(obj)
        return matches

    def retarget_to_object(self, target_object: dict[str, Any]) -> dict[str, Any]:
        raw_env = self.env.unwrapped
        target_pos = (int(target_object["x"]), int(target_object["y"]))

        if hasattr(raw_env, "target_pos"):
            raw_env.target_pos = target_pos
        if hasattr(raw_env, "target_color") and target_object.get("color") is not None:
            raw_env.target_color = target_object["color"]

        color = target_object.get("color")
        object_type = target_object.get("type")
        if color and object_type:
            raw_env.mission = f"go to the {color} {object_type}"
        elif object_type:
            raw_env.mission = f"go to the {object_type}"

        if self.obs is not None:
            self.obs["mission"] = getattr(raw_env, "mission", self.obs.get("mission"))

        return {
            "type": object_type,
            "color": color,
            "x": target_pos[0],
            "y": target_pos[1],
        }

    def act(self, action: int):
        if self.obs is None:
            raise RuntimeError("Environment has not been reset yet.")
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        self.info = info
        self.obs = self._annotate_observation(self.obs)
        observation = Observation(raw=self.obs, step_count=self.step_count)
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()

    def _annotate_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Attach substrate-owned pose/FOV metadata to a MiniGrid observation.

        The cognition layer should not reverse-engineer MiniGrid's egocentric
        image transform. The adapter owns that HOW detail and passes a compact
        visible-cell map to Sense.
        """

        raw = dict(obs)
        raw_env = self.env.unwrapped
        image = raw.get("image")
        width = int(getattr(raw_env, "width", 0) or 0)
        height = int(getattr(raw_env, "height", 0) or 0)
        agent_pos = getattr(raw_env, "agent_pos", None)
        agent_dir = int(getattr(raw_env, "agent_dir", raw.get("direction", 0)) or 0)
        if agent_pos is not None:
            raw["_jeenom_agent_pos"] = (int(agent_pos[0]), int(agent_pos[1]))
        raw["_jeenom_agent_dir"] = agent_dir
        raw["_jeenom_grid_size"] = (width, height)

        image_shape = tuple(int(v) for v in getattr(image, "shape", (0, 0))[:2])
        full_shape = (width, height)
        observation_model = "full_grid" if image_shape == full_shape else "agent_fov"
        raw["_jeenom_observation_model"] = observation_model
        raw["_jeenom_visible_cells"] = self._visible_cell_records(
            observation_model=observation_model,
            image_shape=image_shape,
        )
        return raw

    def _visible_cell_records(
        self,
        *,
        observation_model: str,
        image_shape: tuple[int, int],
    ) -> list[dict[str, int]]:
        raw_env = self.env.unwrapped
        width = int(getattr(raw_env, "width", 0) or 0)
        height = int(getattr(raw_env, "height", 0) or 0)
        if observation_model == "full_grid":
            return [
                {"view_x": x, "view_y": y, "x": x, "y": y}
                for x in range(width)
                for y in range(height)
            ]

        try:
            _, vis_mask = raw_env.gen_obs_grid()
        except Exception:  # noqa: BLE001
            vis_mask = None

        records: list[dict[str, int]] = []
        for x in range(width):
            for y in range(height):
                rel = raw_env.relative_coords(x, y)
                if rel is None:
                    continue
                view_x, view_y = int(rel[0]), int(rel[1])
                if not (0 <= view_x < image_shape[0] and 0 <= view_y < image_shape[1]):
                    continue
                if vis_mask is not None and not bool(vis_mask[view_x, view_y]):
                    continue
                records.append({"view_x": view_x, "view_y": view_y, "x": x, "y": y})
        return records
