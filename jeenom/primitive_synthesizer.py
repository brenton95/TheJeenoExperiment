"""Phase 7.8 — Primitive Synthesizer.

Generates Python function bodies for missing pure grounding primitives via LLM.
Validated primitives are registered in the CapabilityRegistry at runtime.

Architecture:
  CapabilityArbitrator returns synthesize decision
    → PrimitiveSynthesizer.synthesize(spec)
    → SynthesisResult (code, function_name, status)
    → PrimitiveValidator.validate(result, contract)
    → CapabilityRegistry.register_synthesized(handle, fn)
    → station grounds through registered primitive

Hard rules:
  - Only pure grounding/query primitives. No actuation, no I/O, no env access.
  - Generated code may only use: math, SceneModel, SceneObject, and primitive inputs.
  - Failed synthesis or failed validation never unblocks execution.
  - No substrate imports in this module (AST-verified by probe).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


SYNTHESIS_STATUSES = ("success", "refused", "failed", "unsupported")

GROUNDING_FUNCTION_SIGNATURE = (
    "def {function_name}(scene, selector):\n"
    "    # scene: SceneModel — agent_x, agent_y, find(), manhattan_distance_from_agent()\n"
    "    # selector: dict — object_type, color, exclude_colors, relation, distance_metric\n"
    "    # Returns: list[tuple[float, SceneObject]] sorted ascending by distance\n"
)

CLAIMS_FILTER_FUNCTION_SIGNATURE = (
    "def {function_name}(entries, condition):\n"
    "    # entries: list[GroundedObjectEntry] — typed ActiveClaims entries with fields:\n"
    "    #   .color (str|None), .x (int), .y (int), .distance (float),\n"
    "    #   .object_type (str), .metric (str|None), .provenance (str|None)\n"
    "    # condition: dict — threshold (float), comparison (str)\n"
    "    #   comparison values: 'above', 'below', 'within', 'at_least', 'at_most'\n"
    "    # Returns: list[GroundedObjectEntry] — filtered subset, order preserved\n"
    "    # MUST NOT access scene, env, filesystem, or network. MUST NOT import anything.\n"
)

CLAIMS_FILTER_EXAMPLE = (
    "# Example: parametric distance threshold filter over ActiveClaims entries\n"
    "def claims_filter_threshold(entries, condition):\n"
    "    threshold = float(condition.get('threshold', 0))\n"
    "    comparison = condition.get('comparison', 'above')\n"
    "    if comparison == 'above':\n"
    "        return [e for e in entries if e.distance > threshold]\n"
    "    elif comparison == 'at_least':\n"
    "        return [e for e in entries if e.distance >= threshold]\n"
    "    elif comparison == 'below':\n"
    "        return [e for e in entries if e.distance < threshold]\n"
    "    elif comparison in ('at_most', 'within'):\n"
    "        return [e for e in entries if e.distance <= threshold]\n"
    "    return entries\n"
)

ALLOWED_IMPORTS = frozenset({"typing"})


@dataclass
class SynthesisResult:
    handle: str
    function_name: str
    code: str
    status: str  # one of SYNTHESIS_STATUSES
    error_message: str = ""

    def __post_init__(self) -> None:
        if self.status not in SYNTHESIS_STATUSES:
            raise ValueError(f"Invalid synthesis status: {self.status}")


def synthesis_response_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "function_name": {"type": "string"},
            "function_body": {"type": "string"},
            "description": {"type": "string"},
        },
        "required": ["function_name", "function_body", "description"],
        "additionalProperties": False,
    }


class SynthesizerBackend(ABC):
    """Abstract synthesis backend. No substrate imports permitted."""

    @abstractmethod
    def synthesize(
        self,
        handle: str,
        description: str,
        consumes: tuple[str, ...],
        produces: tuple[str, ...],
        existing_example: str | None = None,
        previous_code: str | None = None,
        validation_error: str | None = None,
    ) -> SynthesisResult:
        raise NotImplementedError


class SmokeTestSynthesizer(SynthesizerBackend):
    """Deterministic synthesizer — always refuses. Used when no LLM is available."""

    def synthesize(
        self,
        handle: str,
        description: str,
        consumes: tuple[str, ...],
        produces: tuple[str, ...],
        existing_example: str | None = None,
        previous_code: str | None = None,
        validation_error: str | None = None,
    ) -> SynthesisResult:
        return SynthesisResult(
            handle=handle,
            function_name="",
            code="",
            status="refused",
            error_message="SmokeTestSynthesizer does not generate code.",
        )


class LLMSynthesizer(SynthesizerBackend):
    """LLM-backed synthesizer.

    Asks OpenRouter to produce a Python function body for the missing primitive.
    Falls back to SmokeTestSynthesizer when the API key is unavailable.
    """

    MANHATTAN_EXAMPLE = (
        "# Example: Manhattan distance grounding (ground_closest_door_manhattan)\n"
        "def ground_closest_door_manhattan(scene, selector):\n"
        "    doors = scene.find(\n"
        "        object_type=selector['object_type'],\n"
        "        color=selector.get('color'),\n"
        "        exclude_colors=selector.get('exclude_colors') or [],\n"
        "    )\n"
        "    if not doors:\n"
        "        return []\n"
        "    return sorted(\n"
        "        [\n"
        "            (geometry.manhattan(d.coord, scene.agent_coord), d)\n"
        "            for d in doors\n"
        "        ],\n"
        "        key=lambda pair: (pair[0], pair[1].color or ''),\n"
        "    )\n"
    )

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        transport: Callable[[dict[str, Any]], Any] | None = None,
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        self.timeout = timeout
        self.transport = transport or self._chat_completions_transport
        self._fallback_reason: str | None = None
        if not self.api_key:
            self._fallback_reason = "OPENROUTER_API_KEY not set; synthesis unavailable."

    def synthesize(
        self,
        handle: str,
        description: str,
        consumes: tuple[str, ...],
        produces: tuple[str, ...],
        existing_example: str | None = None,
        previous_code: str | None = None,
        validation_error: str | None = None,
    ) -> SynthesisResult:
        if self._fallback_reason:
            return SynthesisResult(
                handle=handle,
                function_name="",
                code="",
                status="refused",
                error_message=self._fallback_reason,
            )

        function_name = _handle_to_function_name(handle)
        is_claims_filter = handle.startswith("claims.")

        if is_claims_filter:
            signature = CLAIMS_FILTER_FUNCTION_SIGNATURE.format(function_name=function_name)
            example = existing_example or CLAIMS_FILTER_EXAMPLE
            system_prompt = (
                "You are the JEENOM primitive synthesizer. Your task is to generate a pure Python "
                "claims-filter function that operates exclusively on typed ActiveClaims entries.\n\n"
                "HARD RULES:\n"
                "  - Only use: standard Python built-ins and the entries/condition inputs.\n"
                "  - Do NOT import anything — no math, no gymnasium, no minigrid, no numpy.\n"
                "  - Do NOT access SceneModel, env, filesystem, or network.\n"
                "  - Do NOT call env.step() or any robot action.\n"
                "  - The function must be pure and deterministic.\n"
                "  - Return a list[GroundedObjectEntry] — filtered subset, order preserved.\n"
                "  - entries may be empty — return [] in that case.\n\n"
                f"Function signature to implement:\n{signature}\n\n"
                f"Reference implementation:\n{example}\n\n"
                "Put the complete function in function_body: start with exactly the def line, "
                "then all indented body lines. Return only syntactically valid Python code. "
                "Do not include Markdown fences or import statements.\n"
            )
        else:
            example = existing_example or self.MANHATTAN_EXAMPLE
            signature = GROUNDING_FUNCTION_SIGNATURE.format(function_name=function_name)
            system_prompt = (
                "You are the JEENOM primitive synthesizer. Your task is to generate a pure Python "
                "function body for a missing grounding primitive.\n\n"
                "HARD RULES:\n"
                "  - Only use: math, geometry, standard Python built-ins, and the scene/selector inputs.\n"
                "  - For distance, prefer geometry.manhattan/geometry.euclidean over coord pairs\n"
                "    (e.g. geometry.manhattan(d.coord, scene.agent_coord)) — these are N-dimensional\n"
                "    and float-capable, so they work on 2D and 3D substrates without change.\n"
                "  - Do NOT import gymnasium, minigrid, numpy, torch, or any external library.\n"
                "  - Do NOT access the filesystem or network.\n"
                "  - Do NOT call env.step() or any robot action.\n"
                "  - The function must be pure and deterministic.\n"
                "  - Return a list[tuple[float, SceneObject]] sorted ascending by distance.\n"
                "  - An empty scene returns [].\n\n"
                f"Function signature to implement:\n{signature}\n\n"
                f"Reference implementation (Manhattan distance):\n{example}\n\n"
                "Put the complete function in function_body: start with exactly this def line, "
                "then all indented body lines. Return only syntactically valid Python code in "
                "function_body. Do not include Markdown fences. Do not include import statements — "
                "use math.sqrt / geometry.euclidean etc. inline.\n"
            )
        if validation_error and previous_code:
            system_prompt += (
                "\nThe previous candidate failed validation. Repair it instead of changing "
                "the requested primitive.\n"
                f"Validation failure:\n{validation_error}\n\n"
                f"Previous candidate:\n{previous_code}\n\n"
                "Return a complete corrected function only.\n"
            )
        user_payload = {
            "handle": handle,
            "description": description,
            "consumes": list(consumes),
            "produces": list(produces),
            "function_name": function_name,
        }
        request_payload = {
            "method_name": "synthesize_primitive",
            "system_prompt": system_prompt,
            "user_payload": user_payload,
            "schema_name": "jeenom_synthesis_result",
            "schema": synthesis_response_json_schema(),
            "max_tokens": 512,
        }

        try:
            raw = self.transport(request_payload)
            return self._parse_result(handle, function_name, raw)
        except Exception as exc:  # noqa: BLE001
            return SynthesisResult(
                handle=handle,
                function_name=function_name,
                code="",
                status="failed",
                error_message=str(exc),
            )

    def _parse_result(
        self,
        handle: str,
        function_name: str,
        raw: Any,
    ) -> SynthesisResult:
        if not isinstance(raw, dict):
            return SynthesisResult(
                handle=handle,
                function_name=function_name,
                code="",
                status="failed",
                error_message="Synthesis response was not a JSON object.",
            )
        body = raw.get("function_body", "")
        if not isinstance(body, str) or not body.strip():
            return SynthesisResult(
                handle=handle,
                function_name=function_name,
                code="",
                status="failed",
                error_message="Synthesis response missing function_body.",
            )
        # Normalize: LLMs sometimes emit only the indented body without the def line.
        # If the code doesn't start with a def, prepend the appropriate signature.
        # Preserve original indentation — only strip leading blank lines.
        if not body.lstrip("\n").startswith("def "):
            if handle.startswith("claims."):
                sig = CLAIMS_FILTER_FUNCTION_SIGNATURE.format(function_name=function_name)
            else:
                sig = GROUNDING_FUNCTION_SIGNATURE.format(function_name=function_name)
            body = sig + body.lstrip("\n")
        # Reject any import statements the LLM might have snuck in
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                token = stripped.split()[1].split(".")[0]
                if token not in ALLOWED_IMPORTS:
                    return SynthesisResult(
                        handle=handle,
                        function_name=function_name,
                        code="",
                        status="failed",
                        error_message=f"Generated code contains disallowed import: {token}",
                    )
        return SynthesisResult(
            handle=handle,
            function_name=raw.get("function_name", function_name),
            code=body,
            status="success",
        )

    def _chat_completions_transport(self, request_payload: dict[str, Any]) -> Any:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request_payload["system_prompt"]},
                {
                    "role": "user",
                    "content": json.dumps(request_payload["user_payload"], indent=2),
                },
            ],
            "max_tokens": request_payload["max_tokens"],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": request_payload["schema_name"],
                    "schema": request_payload["schema"],
                    "strict": True,
                },
            },
        }

        req = urllib.request.Request(
            url="https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                raw_response = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP error {exc.code}: {details}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenRouter network error: {exc}") from exc

        message = raw_response["choices"][0]["message"]
        refusal = message.get("refusal")
        if refusal:
            raise RuntimeError(f"Model refusal: {refusal}")
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Expected string JSON content from OpenRouter")
        return json.loads(content)


def _handle_to_function_name(handle: str) -> str:
    """Convert a capability handle to a Python function name."""
    return handle.replace(".", "_").replace("-", "_")


def build_synthesizer(
    compiler_name: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> SynthesizerBackend:
    if compiler_name == "llm":
        return LLMSynthesizer(api_key=api_key, model=model)
    return SmokeTestSynthesizer()


default_synthesizer: SynthesizerBackend = SmokeTestSynthesizer()
