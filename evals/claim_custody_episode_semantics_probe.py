"""Interactive episode continuity and explicit-reset semantics.

Task turns continue the current live environment. A fresh seeded episode is
created only when the operator explicitly requests reset.
"""
from __future__ import annotations

from harness import emit_result, first_line, make_session, patched_env_builder


def _human_session():
    return make_session(render_mode="human", max_loops=64)


def main() -> int:
    metrics: dict[str, bool] = {}
    details: dict[str, str] = {}

    with patched_env_builder():
        delivery = _human_session()
        delivery.handle_utterance("the red door is the delivery target")
        delivery.handle_utterance("go to the red door")
        reset_response = delivery.handle_utterance("reset")
        delivery_response = delivery.handle_utterance("go to the delivery target")
        metrics["explicit_reset_starts_fresh_episode"] = "RESET:" in reset_response
        metrics["durable_target_after_reset_completes"] = (
            "RUN COMPLETE" in delivery_response
        )
        metrics["task_after_explicit_reset_does_not_no_path"] = (
            "no_path_found" not in delivery_response
        )
        details["reset_response"] = first_line(reset_response)
        details["delivery_target_response"] = first_line(delivery_response)

    metrics["explicit_reset_episode_semantics_hold"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="explicit_reset_episode_semantics_hold")


if __name__ == "__main__":
    raise SystemExit(main())
