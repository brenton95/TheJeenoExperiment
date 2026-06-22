import argparse
import os
import sys
import subprocess
from pathlib import Path

from manifest import EVAL_SPECS, EXPECTED_FAIL_SUITE, LIVE_LLM_SUITE, select_eval_specs

# LIVE_LLM_SUITE (from manifest) is the ONLY suite permitted to make real LLM calls. Every
# other suite is the deterministic gate and runs with the live-LLM key stripped (below), so
# no probe can flake on a network call regardless of whether its author neutralized it.


def main():
    parser = argparse.ArgumentParser(description="Run JEENOM eval probes.")
    parser.add_argument(
        "--suite",
        default="all",
        help="Eval suite to run: all, architecture, cleanup, llm_path, orpi, smoke.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List selected evals without running them.",
    )
    args = parser.parse_args()

    evals_dir = Path(__file__).parent
    selected_specs = select_eval_specs(args.suite)
    expected_fail = args.suite == EXPECTED_FAIL_SUITE
    if not selected_specs:
        if expected_fail:
            print("Found 0 eval scripts to run for suite=expected_fail.")
            print("No expected-fail probes are currently registered. ✅")
            sys.exit(0)
        known = sorted(
            {"all"}
            | {
                suite
                for spec in EVAL_SPECS
                for suite in spec.get("suites", [])
            }
        )
        print(f"Unknown or empty suite: {args.suite}")
        print(f"Known suites: {', '.join(known)}")
        sys.exit(2)

    eval_files = [evals_dir / str(spec["file"]) for spec in selected_specs]

    missing = [path.name for path in eval_files if not path.exists()]
    if missing:
        print("Manifest references missing eval files:")
        for name in missing:
            print(f"  - {name}")
        sys.exit(2)

    if args.list:
        print(f"Selected {len(eval_files)} eval scripts for suite={args.suite}:")
        for spec, path in zip(selected_specs, eval_files):
            suites = ", ".join(spec.get("suites", []))
            print(f"  - {path.name} [{suites}]")
        sys.exit(0)
    
    # Expected-fail suite: a probe's FAILURE is the clean state; a PASS means the feature
    # landed and the probe should graduate into EVAL_SPECS.
    print(f"Found {len(eval_files)} eval scripts to run for suite={args.suite}.")
    if expected_fail:
        print("(expected-fail suite: a failing probe is the EXPECTED state; a pass graduates.)")

    failures = []
    graduates = []
    fallback_enabled = []

    # Deterministic-gate guarantee: strip the live-LLM key for every suite except live_llm,
    # so gate probes cannot make a real OpenRouter call (flaky / networked / costly). The
    # live_llm suite passes the environment through; its probes skip when no key is present.
    run_env = os.environ.copy()
    if args.suite != LIVE_LLM_SUITE:
        run_env.pop("OPENROUTER_API_KEY", None)

    for eval_file in eval_files:
        print(f"\n{'='*60}")
        cmd = [sys.executable, str(eval_file)]

        # Check if script accepts --allow-fallback
        script_content = eval_file.read_text()
        if "--allow-fallback" in script_content:
            cmd.append("--allow-fallback")
            fallback_enabled.append(eval_file.name)

        mode = "offline fallback allowed" if "--allow-fallback" in cmd else "strict/offline"
        print(f"Running {eval_file.name} ({mode})...")
        print(f"{'='*60}")

        result = subprocess.run(cmd, capture_output=True, text=True, env=run_env)
        passed = result.returncode == 0

        if expected_fail:
            if passed:
                print(f"🎓 GRADUATE → {eval_file.name} now PASSES; move its spec into EVAL_SPECS.")
                graduates.append(eval_file.name)
            else:
                print(f"🔴 expected-fail ✓: {eval_file.name} (red as designed)")
        elif not passed:
            print(f"❌ FAILED: {eval_file.name}")
            print(result.stdout)
            print(result.stderr)
            failures.append((eval_file.name, result.returncode))
        else:
            print(f"✅ PASSED: {eval_file.name}")

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total run: {len(eval_files)}")
    print(f"Fallback-enabled probes: {len(fallback_enabled)}")

    if expected_fail:
        print(f"Expected-fail (red as designed): {len(eval_files) - len(graduates)}")
        print(f"Graduated (now passing): {len(graduates)}")
        if graduates:
            print("\nProbes ready to graduate into EVAL_SPECS:")
            for name in graduates:
                print(f"  - {name}")
            sys.exit(1)
        print("\nAll expected-fail probes are red as designed. ✅")
        sys.exit(0)

    print(f"Passed: {len(eval_files) - len(failures)}")
    print(f"Failed: {len(failures)}")
    if failures:
        print("\nFailed scripts:")
        for name, code in failures:
            print(f"  - {name} (exit code: {code})")
        sys.exit(1)
    else:
        print("\nAll evals passed successfully! 🎉")
        sys.exit(0)

if __name__ == "__main__":
    main()
