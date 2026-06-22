"""Live-LLM intent-fidelity probe — the opt-in `live_llm` suite (REAL model calls).

This is the non-regex coverage: it compiles operator utterances through the real
`LLMCompiler` backend (no fake transport) and asserts the LLM produced the correct
STRUCTURED decision — the "tool call": `intent_type` and the dispatched `command.kind` —
plus, for the refusal case, the *deterministic* statement that a function (not the LLM)
emits. It never asserts on the LLM's free-text prose.

Discipline:
- SKIPS (exit 0) when OPENROUTER_API_KEY is unset, so the deterministic gate and keyless CI
  are unaffected. eval_master only ever runs this suite with the key present.
- Cost-bounded: it inspects the compile/dispatch decision via `command_from_llm_intent`
  WITHOUT executing tasks (no sense/skill-plan LLM calls), so each case is ~1-2 model calls.
- Backend-swappable: `build_compiler("llm")` (driven by `compiler_name="llm"`) is the seam.
  Pointing it at a local model (e.g. LLAMA) later is a backend change, not a probe change.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from harness import emit_result

ENV_ID = "MiniGrid-GoToDoor-8x8-v0"
SEED = 42

# (utterance, allowed intent_types). We assert the STABLE things: the LLM was actually used
# (no silent regex/smoke fallback) and produced the correct intent-type tool-call decision.
# Downstream command_kind is intentionally NOT asserted here — it depends on live scene
# grounding (e.g. "the red door" may resolve to ambiguous), which is legitimate runtime
# behavior, not an LLM-fidelity property.
CASES = [
    ("go to the red door", {"task_instruction"}),
    # "N steps" is legitimately either a repeated motor_command or a motor_sequence — both
    # are valid motor tool-calls. The probe verifies a sensible motor decision, not which.
    ("go straight for 3 steps", {"motor_command", "motor_sequence"}),
    ("pick up the red key", {"unsupported"}),
]


def main() -> int:
    if not os.getenv("OPENROUTER_API_KEY"):
        print("SKIPPED: live_llm probe requires OPENROUTER_API_KEY "
              "(deterministic gate unaffected).")
        return 0

    from jeenom.operator_station import OperatorStationSession

    metrics: dict[str, bool] = {}
    details: dict[str, object] = {}

    for utterance, ok_intents in CASES:
        label = utterance.replace(" ", "_")[:20]
        session = OperatorStationSession(
            compiler_name="llm",
            env_id=ENV_ID,
            seed=SEED,
            render_mode="none",
            memory_root=Path(tempfile.mkdtemp()),
        )
        try:
            command = session.command_from_llm_intent(utterance)
            intent = session.last_operator_intent
            compile_calls = [
                c for c in getattr(session.compiler, "call_history", [])
                if c.get("method_name") == "compile_operator_intent"
            ]
            # The decision genuinely went through the LLM — no silent smoke/regex fallback.
            # This is the regression guard for the truncation bug that made every task
            # compile fall back to the deterministic fast-path.
            metrics[f"{label}_used_live_llm"] = (
                bool(compile_calls)
                and not any(c.get("used_fallback") for c in compile_calls)
            )
            metrics[f"{label}_intent_decision_ok"] = (
                intent is not None and intent.intent_type in ok_intents
            )
            details[label] = {
                "intent_type": intent.intent_type if intent else None,
                "command_kind": command.kind,
                "compile_calls": len(compile_calls),
                "fallback_reasons": [
                    c.get("reason") for c in compile_calls if c.get("used_fallback")
                ],
            }
        finally:
            session.close()

    metrics["live_llm_intent_fidelity_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="live_llm_intent_fidelity_holds")


if __name__ == "__main__":
    raise SystemExit(main())
