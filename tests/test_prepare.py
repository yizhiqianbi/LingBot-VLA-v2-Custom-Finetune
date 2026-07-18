import json
import tempfile
import unittest
from pathlib import Path

import yaml

from lingbot_vla_finetune.contract import contract_sha256
from lingbot_vla_finetune.prepare import (
    RECEIPT_SCHEMA,
    read_prepare_receipt,
    validate_prepared_metadata,
)


class PreparedDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "prepared"
        self.contract_path = Path(self.temp.name) / "contract.yaml"
        self._write_contract()
        self._write_prepared_fixture()

    def tearDown(self):
        self.temp.cleanup()

    def _write_contract(self):
        contract = {
            "schema_version": "test",
            "dataset": {
                "repo_id": "example/test",
                "revision": "2" * 40,
                "lerobot_version": "v2.1",
                "fps": 28,
                "episodes": 2,
                "frames": 12,
                "task": "test task",
            },
            "prepared_dataset": {
                "lerobot_version": "v3.0",
                "lerobot_package_version": "0.4.2",
                "receipt_name": ".prepare.json",
            },
            "raw_schema": {
                "state_key": "observation.state",
                "action_key": "action",
                "state_width": 15,
                "action_width": 15,
                "cameras": ["observation.images.camera"],
            },
            "training_mapping": {},
            "camera_mapping": {},
            "acceptance_assumptions": [],
            "audit_thresholds": {},
        }
        self.contract_path.write_text(
            yaml.safe_dump(contract, sort_keys=False),
            encoding="utf-8",
        )

    def _write_prepared_fixture(self):
        info = {
            "codebase_version": "v3.0",
            "fps": 28,
            "total_episodes": 2,
            "total_frames": 12,
            "features": {
                "observation.state": {"shape": [15]},
                "action": {"shape": [15]},
                "observation.images.camera": {"shape": [8, 8, 3]},
            },
        }
        (self.root / "meta").mkdir(parents=True)
        (self.root / "meta" / "info.json").write_text(
            json.dumps(info), encoding="utf-8"
        )
        required = [
            self.root / "data" / "chunk-000" / "file-000.parquet",
            self.root / "videos" / "camera" / "chunk-000" / "file-000.mp4",
            self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
            self.root / "meta" / "tasks.parquet",
            self.root / "meta" / "stats.json",
        ]
        for path in required:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "dataset_repo_id": "example/test",
            "dataset_revision": "2" * 40,
            "contract_sha256": contract_sha256(self.contract_path),
            "lerobot_package_version": "0.4.2",
            "prepared_dataset_root": str(self.root.resolve()),
        }
        (self.root / ".prepare.json").write_text(json.dumps(receipt), encoding="utf-8")

    def test_prepared_fixture_and_receipt_pass(self):
        metrics = validate_prepared_metadata(self.root, self.contract_path)
        self.assertEqual(metrics["frames"], 12)
        receipt = read_prepare_receipt(self.root, self.contract_path)
        self.assertEqual(receipt["dataset_revision"], "2" * 40)

    def test_changed_contract_rejects_receipt(self):
        payload = yaml.safe_load(self.contract_path.read_text(encoding="utf-8"))
        payload["training_mapping"] = {"changed": True}
        self.contract_path.write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "receipt is stale"):
            read_prepare_receipt(self.root, self.contract_path)


if __name__ == "__main__":
    unittest.main()
