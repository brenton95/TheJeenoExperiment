"""Explicit eval manifest for eval_master.py.

Suites:
  all          - every probe in EVAL_SPECS (the all-green gate)
  architecture - invariant and architecture probes
  cleanup      - Phase 9 cleanup red-bar probes
  llm_path     - fake-transport LLMCompiler route/semantic parity probes
  orpi         - Phase 12 ORPI-v0 conformance probes
  smoke        - historical smoke/regression probes
  expected_fail - eval-first red-bar probes whose FAILURE is the expected (clean) state;
                  kept OUT of EVAL_SPECS so `all` stays green. eval_master inverts the
                  verdict for this suite. A probe here that PASSES is the graduation signal:
                  move its spec into EVAL_SPECS and it joins the main green count.
"""
from __future__ import annotations

EVAL_SPECS: list[dict[str, object]] = [
    # ── authority ─────────────────────────────────────────────────────────────
    {"file": "authority_capability_arbitrator_probe.py", "suites": ["architecture"]},
    {"file": "authority_clarification_resume_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "authority_command_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "authority_command_registry_probe.py", "suites": ["architecture"]},
    {"file": "authority_execution_ticket_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "authority_kb_channel_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "authority_memory_write_gate_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "authority_mission_child_ticket_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "authority_motor_ticket_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "authority_request_plan_gate_probe.py", "suites": ["cleanup"]},
    {"file": "authority_side_effect_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "authority_substrate_contract_probe.py", "suites": ["architecture", "cleanup"]},
    # ── claim custody ─────────────────────────────────────────────────────────
    {"file": "claim_custody_context_planning_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "claim_custody_env_assumption_probe.py", "suites": ["architecture"]},
    {"file": "claim_custody_episode_semantics_probe.py", "suites": ["cleanup"]},
    {"file": "claim_custody_knowledge_base_probe.py", "suites": ["architecture"]},
    {"file": "claim_custody_knowledge_scope_probe.py", "suites": ["architecture", "orpi"]},
    {"file": "claim_custody_mission_contract_probe.py", "suites": ["architecture"]},
    {"file": "claim_custody_mission_flow_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "claim_custody_operational_context_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "claim_custody_plan_reuse_probe.py", "suites": ["architecture"]},
    {"file": "claim_custody_representation_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "claim_custody_stale_claim_probe.py", "suites": ["architecture"]},
    {"file": "claim_custody_typed_claims_probe.py", "suites": ["architecture"]},
    {"file": "claim_custody_unified_abstraction_probe.py", "suites": ["architecture"]},
    {"file": "claim_custody_unverifiable_freshness_probe.py", "suites": ["architecture"]},
    # ── intent fidelity ───────────────────────────────────────────────────────
    {"file": "intent_fidelity_cache_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "intent_fidelity_concept_probe.py", "suites": ["architecture"]},
    {"file": "intent_fidelity_llm_motor_sequence_probe.py", "suites": ["architecture", "llm_path"]},
    {"file": "intent_fidelity_llm_operator_matrix_probe.py", "suites": ["architecture", "llm_path"]},
    {"file": "intent_fidelity_llm_path_parity_probe.py", "suites": ["architecture", "llm_path"]},
    {"file": "intent_fidelity_llm_schema_strict_probe.py", "suites": ["architecture", "llm_path"]},
    {"file": "intent_fidelity_motor_command_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "intent_fidelity_motor_count_probe.py", "suites": ["architecture"]},
    {"file": "intent_fidelity_motor_implicit_probe.py", "suites": ["architecture"]},
    {"file": "intent_fidelity_operator_intent_probe.py", "suites": ["architecture"]},
    {"file": "intent_fidelity_sequential_probe.py", "suites": ["architecture"]},
    {"file": "intent_fidelity_sequence_instruction_probe.py", "suites": ["architecture"]},
    {"file": "intent_fidelity_verifier_probe.py", "suites": ["architecture"]},
    # ── pipeline ──────────────────────────────────────────────────────────────
    {"file": "pipeline_dispatch_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "pipeline_orpi_labelled_episode_probe.py", "suites": ["architecture", "orpi"]},
    {"file": "substrate_partial_observability_needs_evidence_probe.py", "suites": ["architecture"]},
    {"file": "pipeline_steering_directive_probe.py", "suites": ["architecture", "orpi"]},
    {"file": "pipeline_procedure_selection_probe.py", "suites": ["architecture", "orpi"]},
    {"file": "pipeline_request_plan_probe.py", "suites": ["architecture"]},
    {"file": "pipeline_turn_orchestrator_probe.py", "suites": ["architecture", "cleanup"]},
    # ── regression ────────────────────────────────────────────────────────────
    {"file": "regression_golden_probe.py", "suites": ["smoke"]},
    {"file": "regression_jit_prewarm_probe.py", "suites": ["smoke"]},
    {"file": "regression_live_operator_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "regression_orpi_primitive_type_migration_probe.py", "suites": ["architecture", "orpi"]},
    {"file": "regression_phase7_5_probe.py", "suites": ["architecture"]},
    {"file": "regression_phase7_59_probe.py", "suites": ["architecture"]},
    {"file": "regression_phase7_6_probe.py", "suites": ["architecture"]},
    {"file": "regression_phase7_8_probe.py", "suites": ["architecture"]},
    {"file": "regression_phase7_95_probe.py", "suites": ["architecture"]},
    # ── repair ────────────────────────────────────────────────────────────────
    {"file": "repair_mismatch_probe.py", "suites": ["architecture"]},
    {"file": "repair_operational_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "repair_truthfulness_probe.py", "suites": ["cleanup"]},
    # ── substrate ─────────────────────────────────────────────────────────────
    {"file": "substrate_adapter_probe.py", "suites": ["architecture"]},
    {"file": "substrate_capability_matcher_probe.py", "suites": ["architecture"]},
    {"file": "substrate_capability_registry_probe.py", "suites": ["architecture"]},
    {"file": "substrate_cortex_invariant_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "substrate_hardware_schema_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "substrate_domain_helper_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "substrate_minigrid_fov_probe.py", "suites": ["architecture"]},
    {"file": "substrate_orpi_cadence_probe.py", "suites": ["architecture", "orpi"]},
    {"file": "substrate_orpi_contract_coverage_probe.py", "suites": ["architecture", "orpi"]},
    {"file": "substrate_orpi_manifest_registration_probe.py", "suites": ["architecture", "orpi"]},
    {"file": "substrate_orpi_no_llm_in_loop_probe.py", "suites": ["architecture", "orpi"]},
    {"file": "substrate_orpi_postcondition_probe.py", "suites": ["architecture", "orpi"]},
    {"file": "substrate_runtime_package_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "substrate_schema_enforcement_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "substrate_static_architecture_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "substrate_vocabulary_advertising_probe.py", "suites": ["architecture", "llm_path"]},
    # ── synthesis ─────────────────────────────────────────────────────────────
    {"file": "synthesis_primitive_composition_probe.py", "suites": ["architecture"]},
    {"file": "synthesis_primitive_ladder_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "synthesis_user_defined_metric_probe.py", "suites": ["architecture", "cleanup"]},
    # ── naming contract ───────────────────────────────────────────────────────
    {"file": "eval_naming_contract_probe.py", "suites": ["architecture"]},
]


# Eval-first red-bar probes that are EXPECTED to fail until their phase implements them.
# Kept separate from EVAL_SPECS so the all-green gate never includes a known-red probe.
# eval_master runs this suite with an inverted verdict (failure = clean). A pass here means
# the feature landed — graduate the spec into EVAL_SPECS.
EXPECTED_FAIL_SPECS: list[dict[str, object]] = []

EXPECTED_FAIL_SUITE = "expected_fail"


# Genuine live-LLM probes: they make REAL model calls (the only suite eval_master lets reach
# the network). Kept OUT of EVAL_SPECS so the deterministic "all" gate never makes a network
# call. Opt-in via `--suite live_llm`; each probe SKIPS (exit 0) when no OPENROUTER_API_KEY is
# present, so keyless CI stays green. Assertions target the LLM's STRUCTURED decision (the
# "tool call" — intent_type / command_kind / graph_status), never its free-text prose.
# Backend-swappable: pointing build_compiler at a local model (e.g. LLAMA) is a compiler-backend
# change, not a probe change.
LIVE_LLM_SPECS: list[dict[str, object]] = [
    {"file": "intent_fidelity_live_llm_probe.py", "suites": ["live_llm"]},
]

LIVE_LLM_SUITE = "live_llm"


def select_eval_specs(suite: str) -> list[dict[str, object]]:
    if suite == EXPECTED_FAIL_SUITE:
        return list(EXPECTED_FAIL_SPECS)
    if suite == LIVE_LLM_SUITE:
        return list(LIVE_LLM_SPECS)
    if suite == "all":
        return list(EVAL_SPECS)
    return [
        spec
        for spec in EVAL_SPECS
        if suite in set(spec.get("suites", []))
    ]
