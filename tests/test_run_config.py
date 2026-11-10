"""Configuration and persistence tests need only Python's standard library."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from aipi_pipeline.run_config import RunConfig, canonical_json
from aipi_pipeline.run_record import RunRecord


def example():
    return {
        "name": "baseline", "model": {"type": "logistic_regression", "params": {"C": 1.0}},
        "metadata": {"owner": "researcher", "labels": ["baseline"]},
    }


class RunConfigTests(unittest.TestCase):
    def test_key_order_whitespace_and_explicit_defaults_do_not_change_hash(self):
        original = RunConfig(example())
        reordered = dict(reversed(list(original.to_dict().items())))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(reordered, indent=4))
            self.assertEqual(original.config_hash, RunConfig.load(path).config_hash)

    def test_parameters_model_and_metadata_are_part_of_hash(self):
        original = RunConfig(example())
        for key, value in (
            ("model", {"type": "random_forest"}),
            ("model", {"type": "logistic_regression", "params": {"C": 2.0}}),
            ("metadata", {"owner": "another researcher"}),
            ("split", {"random_state": 7}),
        ):
            with self.subTest(key=key, value=value):
                self.assertNotEqual(original.config_hash, RunConfig({**example(), key: value}).config_hash)

    def test_input_and_returned_dictionary_mutations_do_not_change_snapshot(self):
        values = example()
        config = RunConfig(values)
        before = config.config_hash
        values["metadata"]["labels"].append("changed")
        config.to_dict()["model"]["params"]["C"] = 10
        self.assertEqual(config.config_hash, before)
        self.assertEqual(config.to_dict()["metadata"]["labels"], ["baseline"])

    def test_saved_bytes_match_config_hash_and_can_be_reloaded(self):
        config = RunConfig({**example(), "metadata": {"description": "café"}})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            config.save(path)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), config.config_hash)
            self.assertEqual(RunConfig.load(path), config)

    def test_invalid_values_are_rejected(self):
        invalid = [
            {"schema_version": 2}, {"schema_version": True}, {"name": ""},
            {"model": {}}, {"model": {"type": "test", "params": []}},
            {"model": {"type": "test", "param": {}}}, {"typo": 1},
            {"split": {"random_state": True}}, {"split": {"random_state": -1}},
            {"split": {"test_size": 1}}, {"split": {"test_size": False}},
            {"metadata": {"score": float("nan")}}, {"metadata": {"score": float("inf")}},
            {"metadata": {1: "not a JSON key"}}, {"metadata": {"value": object()}},
            {"preprocessing": {"standardize": "yes"}}, {"metadata": []},
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                RunConfig({**example(), **values})

    def test_duplicate_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text('{"model":{"type":"first","type":"second"}}')
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                RunConfig.load(path)

    def test_non_object_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "object"):
            RunConfig([])


class RunRecordTests(unittest.TestCase):
    def test_repeated_config_gets_unique_runs_without_overwriting(self):
        config = RunConfig(example())
        with tempfile.TemporaryDirectory() as temporary:
            first = RunRecord.create(config, Path(temporary))
            original = (first.directory / "manifest.json").read_bytes()
            second = RunRecord.create(config, Path(temporary))
            self.assertNotEqual(first.run_hash, second.run_hash)
            self.assertEqual(first.manifest["config_hash"], second.manifest["config_hash"])
            self.assertEqual((first.directory / "manifest.json").read_bytes(), original)
            self.assertEqual(len(list(Path(temporary).iterdir())), 2)
            for record in (first, second):
                identity = {key: record.manifest[key] for key in (
                    "schema_version", "config_hash", "created_at", "nonce",
                )}
                self.assertEqual(
                    record.run_hash, hashlib.sha256(canonical_json(identity).encode()).hexdigest(),
                )
                self.assertEqual(record.directory.name, record.run_hash)
                self.assertEqual(RunConfig.load(record.directory / "config.json"), config)

    def test_manifest_keeps_hash_link_and_final_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = RunConfig(example())
            record = RunRecord.create(config, Path(temporary))
            record.update(status="RUNNING", mlflow_run_id="mlflow-id")
            record.update(status="FAILED", error_type="ValueError")
            saved = json.loads((record.directory / "manifest.json").read_text())
            self.assertEqual(saved["status"], "FAILED")
            self.assertEqual(saved["mlflow_run_id"], "mlflow-id")
            self.assertEqual(saved["config_hash"], config.config_hash)
            self.assertEqual(saved["run_hash"], record.run_hash)
            self.assertIsNotNone(saved["finished_at"])
            self.assertFalse((record.directory / "manifest.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
