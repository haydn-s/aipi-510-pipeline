"""Validated, immutable JSON configuration with a stable SHA-256 identity."""

import hashlib
import json
import math
from dataclasses import InitVar, dataclass, field
from pathlib import Path


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _validate_json(value: object) -> None:
    if type(value) is dict:
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Configuration keys must be strings")
            _validate_json(item)
    elif type(value) is list:
        for item in value:
            _validate_json(item)
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Configuration numbers must be finite")
    elif value is not None and type(value) not in {str, int, bool}:
        raise ValueError("Configuration values must be JSON-compatible")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class RunConfig:
    data: InitVar[dict]
    _json: str = field(init=False, repr=False)

    def __post_init__(self, data: dict) -> None:
        _validate_json(data)
        if not isinstance(data, dict):
            raise ValueError("Run configuration must be a JSON object")
        allowed = {"schema_version", "name", "model", "data", "split", "preprocessing", "metadata"}
        if unknown := data.keys() - allowed:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        resolved = {
            "schema_version": 1, "name": "pipeline", "data": {},
            "split": {}, "preprocessing": {}, "metadata": {}, **data,
        }
        if type(resolved["schema_version"]) is not int or resolved["schema_version"] != 1:
            raise ValueError("Only configuration schema_version 1 is supported")
        if not isinstance(resolved["name"], str) or not resolved["name"].strip():
            raise ValueError("Configuration name must be a nonempty string")
        for key in ("model", "data", "split", "preprocessing", "metadata"):
            if not isinstance(resolved.get(key), dict):
                raise ValueError(f"Configuration {key} must be an object")
        model = {"params": {}, **resolved["model"]}
        if model.keys() - {"type", "params"}:
            raise ValueError("Model only supports type and params fields")
        if not isinstance(model.get("type"), str) or not model["type"].strip():
            raise ValueError("model.type must be a nonempty string")
        if not isinstance(model["params"], dict):
            raise ValueError("model.params must be an object")
        split = {"test_size": 0.2, "random_state": 42, **resolved["split"]}
        if split.keys() - {"test_size", "random_state"}:
            raise ValueError("Split only supports test_size and random_state fields")
        if type(split["test_size"]) not in {int, float} or not 0 < split["test_size"] < 1:
            raise ValueError("split.test_size must be between 0 and 1")
        if type(split["random_state"]) is not int or not 0 <= split["random_state"] < 2**32:
            raise ValueError("split.random_state must be an integer from 0 to 2**32 - 1")
        preprocessing = {"standardize": True, **resolved["preprocessing"]}
        if preprocessing.keys() != {"standardize"} or type(preprocessing["standardize"]) is not bool:
            raise ValueError("preprocessing only supports a boolean standardize field")
        resolved.update(model=model, split=split, preprocessing=preprocessing)
        # Retain only canonical bytes: caller mutations cannot change this snapshot.
        object.__setattr__(self, "_json", canonical_json(resolved))

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object))

    def to_dict(self) -> dict:
        return json.loads(self._json)

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self._json.encode("utf-8")).hexdigest()

    def save(self, path: Path) -> None:
        # The file's exact UTF-8 bytes hash to config_hash (no trailing newline).
        path.write_text(self._json, encoding="utf-8")
