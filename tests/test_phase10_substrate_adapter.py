from __future__ import annotations

import unittest
from unittest.mock import patch

from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.memory import OperationalMemory
from jeenom.minigrid_substrate_adapter import MiniGridSubstrateAdapter
from jeenom.plan_cache import PlanCache
from jeenom.schemas import ExecutionContract, Percepts, PrimitiveCall
from jeenom.sense import MiniGridSense
from jeenom.spine import MiniGridSpine


class TestPhase10SubstrateAdapter(unittest.TestCase):
    def test_minigrid_substrate_creates_role_bindings(self):
        substrate = MiniGridSubstrateAdapter(
            env_id="MiniGrid-GoToDoor-8x8-v0",
            render_mode="none",
        )
        memory = OperationalMemory()
        compiler = SmokeTestCompiler()
        plan_cache = PlanCache(enabled=True)

        self.assertIsInstance(
            substrate.create_sense(memory, compiler, plan_cache),
            MiniGridSense,
        )
        self.assertIsInstance(
            substrate.create_spine(memory, compiler, plan_cache),
            MiniGridSpine,
        )
        self.assertIn("turn_right", substrate.known_action_names())

    def test_minigrid_substrate_raw_motor_execution(self):
        substrate = MiniGridSubstrateAdapter(
            env_id="MiniGrid-GoToDoor-8x8-v0",
            render_mode="none",
        )

        result = substrate.run_motor_actions(seed=42, actions=["turn_right"])

        self.assertTrue(result["success"])
        self.assertEqual(result["steps_taken"], 1)
        self.assertEqual(result["actions_executed"], ["turn_right"])

    def test_human_task_episode_reuses_open_task_adapter_without_reset(self):
        substrate = MiniGridSubstrateAdapter(
            env_id="MiniGrid-GoToDoor-8x8-v0",
            render_mode="human",
        )
        open_adapter = object()
        substrate.task_adapter = open_adapter
        captured: dict[str, object] = {}

        def fake_run_episode(**kwargs):
            captured.update(kwargs)
            return {"_render_adapter": kwargs["render_adapter"], "ok": True}

        with patch("jeenom.run_demo.run_episode", side_effect=fake_run_episode):
            result = substrate.run_task_episode(
                instruction="go to the grey door",
                compiler_name="smoke",
                compiler=SmokeTestCompiler(),
                seed=18,
                max_loops=64,
                memory=OperationalMemory(),
                plan_cache=PlanCache(enabled=True),
                progress_callback=None,
            )

        self.assertTrue(result["ok"])
        self.assertIs(captured["render_adapter"], open_adapter)
        self.assertIs(substrate.task_adapter, open_adapter)
        self.assertIs(captured["skip_reset"], True)

    def test_task_episode_reuses_any_open_adapter_without_reset(self):
        for render_mode, adapter_role in (
            ("human", "preview_adapter"),
            ("none", "task_adapter"),
            ("rgb_array", "task_adapter"),
        ):
            with self.subTest(render_mode=render_mode, adapter_role=adapter_role):
                substrate = MiniGridSubstrateAdapter(
                    env_id="MiniGrid-GoToDoor-8x8-v0",
                    render_mode=render_mode,
                )
                open_adapter = object()
                setattr(substrate, adapter_role, open_adapter)
                captured: dict[str, object] = {}

                def fake_run_episode(**kwargs):
                    captured.update(kwargs)
                    return {"_render_adapter": kwargs["render_adapter"], "ok": True}

                with patch("jeenom.run_demo.run_episode", side_effect=fake_run_episode):
                    result = substrate.run_task_episode(
                        instruction="go to the grey door",
                        compiler_name="smoke",
                        compiler=SmokeTestCompiler(),
                        seed=18,
                        max_loops=64,
                        memory=OperationalMemory(),
                        plan_cache=PlanCache(enabled=True),
                        progress_callback=None,
                    )

                self.assertTrue(result["ok"])
                self.assertIs(captured["render_adapter"], open_adapter)
                self.assertIs(captured["skip_reset"], True)
                self.assertIs(captured["keep_render_open"], True)
                self.assertIs(substrate.task_adapter, open_adapter)
                self.assertIsNone(substrate.preview_adapter)

    def test_spine_treats_zero_length_navigation_path_as_already_at_goal(self):
        spine = MiniGridSpine(
            OperationalMemory(),
            adapter=None,
            compiler=SmokeTestCompiler(),
            plan_cache=PlanCache(enabled=False),
        )
        contract = ExecutionContract(
            skill="navigate_to_object",
            params={"object_type": "door", "target_location": (4, 5)},
        )
        plan = [
            PrimitiveCall(name="plan_grid_path"),
            PrimitiveCall(name="execute_next_path_action"),
        ]
        percepts = Percepts(
            cues={
                "agent_pose": {"x": 4, "y": 4, "dir": 1},
                "target_location": (4, 5),
                "target_object": {"type": "door", "color": "grey"},
                "passable_positions": {(4, 4)},
                "grid_size": (8, 8),
            },
            source="test",
        )

        report = spine.execute_plan(contract, plan, percepts)

        self.assertEqual(report.status, "succeeded")
        self.assertIsNone(report.reason)
        self.assertTrue(report.progress["already_at_navigation_goal"])
        self.assertNotEqual(report.reason, "no_path_found")


if __name__ == "__main__":
    unittest.main()
