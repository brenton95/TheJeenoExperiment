"""Phase 8 probe: Key object type support.

Verifies that 'key' is a recognised object type and that the station
handles key-related utterances correctly.

Checks:
  key_in_object_types         — 'key' in OPERATOR_OBJECT_TYPES
  pickup_object_in_primitives — 'pickup_object' in TASK_PRIMITIVES
  pickup_object_is_planned    — pickup_object.implementation_status == 'planned'
  task_pickup_key_in_registry — 'task.pickup.key' in capability registry
  task_pickup_key_unsupported — task.pickup.key implementation_status == 'unsupported'
  smoke_compile_go_to_key     — SmokeTestCompiler compiles 'go to the red key'
                                as task_instruction with object_type=key
  go_to_key_reports_gap       — handle_utterance('go to the red key') reports MISSING SKILLS
                                for task.go_to_object.key (capability gap surfaced correctly)
  golden_path_no_regression   — 'go to the red door' still returns RUN COMPLETE
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jeenom.capability_registry import CapabilityRegistry
from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.memory import OperationalMemory
from jeenom.operator_station import OperatorStationSession
from jeenom.primitive_library import TASK_PRIMITIVES
from jeenom.schemas import OPERATOR_OBJECT_TYPES


def _make_session(memory_root: Path | None = None) -> OperatorStationSession:
    return OperatorStationSession(
        compiler=SmokeTestCompiler(),
        compiler_name="smoke_test",
        env_id="MiniGrid-GoToDoor-8x8-v0",
        seed=42,
        render_mode="none",
        memory_root=memory_root or Path(tempfile.mkdtemp()),
    )


def main() -> int:
    metrics: dict[str, bool] = {}

    # ── Schema checks ─────────────────────────────────────────────────────────
    metrics["key_in_object_types"] = "key" in OPERATOR_OBJECT_TYPES

    # ── Primitive library checks ───────────────────────────────────────────────
    metrics["pickup_object_in_primitives"] = "pickup_object" in TASK_PRIMITIVES
    spec = TASK_PRIMITIVES.get("pickup_object")
    metrics["pickup_object_is_planned"] = (
        spec is not None and spec.implementation_status == "planned"
    )

    # ── Capability registry checks ─────────────────────────────────────────────
    registry = CapabilityRegistry.minigrid_default()
    key_cap = registry.lookup("task.pickup.key")
    metrics["task_pickup_key_in_registry"] = key_cap is not None
    metrics["task_pickup_key_unsupported"] = (
        key_cap is not None and key_cap.implementation_status == "unsupported"
    )

    # ── SmokeTestCompiler compiles key utterances ──────────────────────────────
    compiler = SmokeTestCompiler()
    memory = OperationalMemory(root=Path(tempfile.mkdtemp()))
    intent = compiler.compile_operator_intent("go to the red key", memory=memory)
    metrics["smoke_compile_go_to_key"] = (
        intent.intent_type == "task_instruction"
        and intent.target is not None
        and intent.target.get("object_type") == "key"
        and intent.target.get("color") == "red"
    )

    # ── Station handles key utterances ────────────────────────────────────────
    session = _make_session()
    result = session.handle_utterance("go to the red key")
    metrics["go_to_key_reports_gap"] = (
        "MISSING SKILLS" in result and "task.go_to_object.key" in result
    )

    # ── Golden path regression ────────────────────────────────────────────────
    session2 = _make_session()
    result2 = session2.handle_utterance("go to the red door")
    metrics["golden_path_no_regression"] = "RUN COMPLETE" in result2

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Phase 8 Key Object Probe")
    print(f"{'='*60}")
    all_passed = True
    for check, passed in metrics.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {check}")
        if not passed:
            all_passed = False

    print(f"{'='*60}")
    print(f"Result: {'ALL PASS' if all_passed else 'SOME FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
