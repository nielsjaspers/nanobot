"""Base tool interface for Mike."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    _TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        raise NotImplementedError

    def cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            return params
        return self._cast_object(params, schema)

    def _cast_object(self, obj: Any, schema: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(obj, dict):
            return obj
        props = schema.get("properties", {})
        result = {}
        for key, value in obj.items():
            result[key] = self._cast_value(value, props.get(key, {}))
        return result

    def _cast_value(self, value: Any, schema: dict[str, Any]) -> Any:
        kind = schema.get("type")
        if kind == "integer" and isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return value
        if kind == "number" and isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return value
        if kind == "boolean" and isinstance(value, str):
            lowered = value.lower()
            if lowered in {"true", "1", "yes"}:
                return True
            if lowered in {"false", "0", "no"}:
                return False
        if kind == "string":
            return value if value is None else str(value)
        if kind == "array" and isinstance(value, list):
            item_schema = schema.get("items")
            return [self._cast_value(item, item_schema or {}) for item in value]
        if kind == "object" and isinstance(value, dict):
            return self._cast_object(value, schema)
        return value

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        if not isinstance(params, dict):
            return [f"parameters must be an object, got {type(params).__name__}"]
        return self._validate(params, {**(self.parameters or {}), "type": "object"}, "")

    def _validate(self, value: Any, schema: dict[str, Any], path: str) -> list[str]:
        kind = schema.get("type")
        label = path or "parameter"
        if kind == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            return [f"{label} should be integer"]
        if kind == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            return [f"{label} should be number"]
        if kind in self._TYPE_MAP and kind not in {"integer", "number"}:
            if not isinstance(value, self._TYPE_MAP[kind]):
                return [f"{label} should be {kind}"]
        errors: list[str] = []
        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"{label} must be one of {schema['enum']}")
        if kind == "object":
            for key in schema.get("required", []):
                if key not in value:
                    errors.append(f"missing required {path + '.' + key if path else key}")
            for key, item in value.items():
                if key in schema.get("properties", {}):
                    errors.extend(
                        self._validate(
                            item, schema["properties"][key], path + "." + key if path else key
                        )
                    )
        if kind == "array" and "items" in schema:
            for idx, item in enumerate(value):
                errors.extend(
                    self._validate(item, schema["items"], f"{path}[{idx}]" if path else f"[{idx}]")
                )
        return errors

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
