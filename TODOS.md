# TODOs

## Refactor — parametric object type handling (Phase 8.5)
The key object implementation in Phase 8 mirrors the door implementation with explicit
`if object_type == "door"` / `if object_type == "key"` branches in multiple files:
- `jeenom/capability_registry.py` — `readiness_for_task()`, `readiness_for_selector()`, manifest entries
- `jeenom/operator_station.py` — routing guard and regex patterns
- `jeenom/sense.py` — adjacency check

These should be replaced with a parametric approach: a single `SUPPORTED_OBJECT_TYPES`
config that drives capability registration, routing, and adjacency rules automatically.
Adding a new object type should require changes in one place only.
This is a refactor (no new capability), so it belongs in its own phase/PR.

## SmokeTestCompiler — generalize object type matching
Currently `llm_compiler.py` hardcodes `door|key` in the regex.
Should dynamically build from `OPERATOR_OBJECT_TYPES` in `schemas.py`:
```python
object_types = "|".join(OPERATOR_OBJECT_TYPES)
pattern = rf"go to the (?P<color>\w+) (?P<object_type>{object_types})"
```
This way adding a new object type to `schemas.py` automatically works in the smoke test with no extra changes.
