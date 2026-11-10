"""Persist each execution separately, linked to its immutable configuration."""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .run_config import RunConfig, canonical_json


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunRecord:
    directory: Path
    manifest: dict

    @classmethod
    def create(cls, config: RunConfig, root: Path) -> "RunRecord":
        identity = {
            "schema_version": 1,
            "config_hash": config.config_hash,
            "created_at": _now(),
            "nonce": uuid4().hex,
        }
        run_hash = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
        directory = root / run_hash
        directory.mkdir(parents=True, exist_ok=False)
        config.save(directory / "config.json")
        record = cls(directory, {
            **identity, "run_hash": run_hash, "status": "CREATED",
            "mlflow_run_id": None, "finished_at": None,
        })
        record.update()
        return record

    @property
    def run_hash(self) -> str:
        return self.manifest["run_hash"]

    @property
    def log_file(self) -> Path:
        return self.directory / "events.jsonl"

    def update(self, **changes: object) -> None:
        self.manifest.update(changes)
        if changes.get("status") in {"FINISHED", "FAILED"}:
            self.manifest["finished_at"] = _now()
        temporary = self.directory / "manifest.json.tmp"
        temporary.write_text(json.dumps(self.manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.directory / "manifest.json")
