"""AI2-THOR golden-path eval — two-tier design.

Tier 2a (default, runs here): mock controller with an Apple, full
handle_utterance pipeline. Proves the wiring + assertions are correct
without Unity.

Tier 2b (--live, runs on Colab): real ai2thor.controller.Controller,
same run_golden core, same assertions.

On Colab (one-time):
    !pip install ai2thor
    !python -c "from ai2thor.controller import Controller; Controller(platform='CloudRendering')"
Then:
    !python evals/eval_golden_ai2thor.py --live
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jeenom.ai2thor_substrate_adapter import build_ai2thor_runtime_package
from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.operator_station import OperatorStationSession


def run_golden(controller: Any) -> dict[str, Any]:
    rp = build_ai2thor_runtime_package(controller=controller)
    session = OperatorStationSession(
        compiler=SmokeTestCompiler(),
        compiler_name="smoke_test",
        env_id="FloorPlan1",
        seed=42,
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
        runtime_package=rp,
    )
    session.handle_utterance("go to the red apple")
    lr = session.last_result
    if lr is None:
        return {
            "task_complete": False,
            "runtime_llm_calls_during_render": -1,
            "cache_miss_during_render": -1,
            "reason": "no last_result",
        }
    fs = lr.get("final_state", {})
    return {
        "task_complete": fs.get("task_complete", False),
        "runtime_llm_calls_during_render": lr.get("runtime_llm_calls_during_render", -1),
        "cache_miss_during_render": lr.get("cache_miss_during_render", -1),
        "loop_count": len(lr.get("loop_records", [])),
        "final_skill_plan": fs.get("current_skill"),
        "reason": None,
    }


class _FakeEvent:
    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata


class _GoldenMockController:
    """Mock controller seeded with a reachable grid and an Apple a few cells away."""

    def __init__(self) -> None:
        self.agent_x = 0.0
        self.agent_z = 0.0
        self.agent_yaw = 0.0
        self.grid_size = 0.25
        self.reachable = [
            {"x": x * 0.25, "y": 0.0, "z": z * 0.25}
            for x in range(4)
            for z in range(4)
        ]

    def step(self, **kwargs: Any) -> _FakeEvent:
        action = kwargs.get("action", "")
        if action == "GetReachablePositions":
            return _FakeEvent(metadata={"actionReturn": self.reachable})

        last_action_success = True
        if action == "MoveAhead":
            yaw = int(self.agent_yaw) % 360
            dx, dz = 0.0, 0.0
            if yaw == 0:
                dz = self.grid_size
            elif yaw == 90:
                dx = self.grid_size
            elif yaw == 180:
                dz = -self.grid_size
            elif yaw == 270:
                dx = -self.grid_size
            self.agent_x = round(self.agent_x + dx, 6)
            self.agent_z = round(self.agent_z + dz, 6)
        elif action == "RotateRight":
            self.agent_yaw = (self.agent_yaw + 90) % 360
        elif action == "RotateLeft":
            self.agent_yaw = (self.agent_yaw - 90) % 360

        return _FakeEvent(metadata={
            "lastActionSuccess": last_action_success,
            "objects": [
                {"objectType": "Apple", "position": {"x": 0.5, "y": 0.9, "z": 0.5}},
            ],
            "agent": {
                "position": {"x": self.agent_x, "y": 0.0, "z": self.agent_z},
                "rotation": {"x": 0.0, "y": self.agent_yaw, "z": 0.0},
            },
        })


def main() -> int:
    parser = argparse.ArgumentParser(description="AI2-THOR golden-path eval.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use a real ai2thor Controller (requires GPU + ai2thor installed).",
    )
    parser.add_argument(
        "--scene",
        default="FloorPlan1",
        help="Scene ID for --live mode (default: FloorPlan1).",
    )
    args = parser.parse_args()

    if args.live:
        from ai2thor.controller import Controller  # type: ignore[import-untyped]

        controller = Controller(
            scene=args.scene,
            platform="CloudRendering",
            renderImage=False,
        )
        tier = "LIVE"
    else:
        controller = _GoldenMockController()
        tier = "MOCK"

    print(f"AI2-THOR GOLDEN PATH EVAL (tier: {tier})\n")

    result = run_golden(controller)

    checks: dict[str, bool] = {}
    checks["task_complete"] = result["task_complete"] is True
    checks["runtime_llm_calls_zero"] = result["runtime_llm_calls_during_render"] == 0
    checks["cache_miss_zero"] = result["cache_miss_during_render"] == 0

    print("CHECKS")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")

    n_pass = sum(checks.values())
    print(f"\n{n_pass}/{len(checks)} passed")

    if not all(checks.values()):
        print(f"\nDiagnostics: {result}")

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
