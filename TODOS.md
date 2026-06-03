# TODOs

## SmokeTestCompiler — generalize object type matching
Currently `llm_compiler.py` hardcodes `door|key` in the regex.
Should dynamically build from `OPERATOR_OBJECT_TYPES` in `schemas.py`:
```python
object_types = "|".join(OPERATOR_OBJECT_TYPES)
pattern = rf"go to the (?P<color>\w+) (?P<object_type>{object_types})"
```
This way adding a new object type to `schemas.py` automatically works in the smoke test with no extra changes.
