import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from lingbot_vla_finetune.contract import contract_sha256
from lingbot_vla_finetune.norm_stats import (
    MANIFEST_SCHEMA,
    norm_manifest_path,
    validate_norm_stats,
)


class NormStatsManifestTest(unittest.TestCase):
    def test_modified_stats_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = root / "contract.yaml"
            contract.write_text(
                """
schema_version: test
dataset:
  repo_id: example/test
  revision: 3333333333333333333333333333333333333333
prepared_dataset:
  lerobot_version: v3.0
  lerobot_package_version: 0.4.2
  receipt_name: .prepare.json
raw_schema:
  state_width: 1
  action_width: 1
training_mapping: {}
camera_mapping: {}
acceptance_assumptions: []
audit_thresholds: {}
""".lstrip(),
                encoding="utf-8",
            )
            stats = root / "norm.json"
            stats.write_text('{"count": 1}\n', encoding="utf-8")
            digest = hashlib.sha256(stats.read_bytes()).hexdigest()
            manifest = {
                "schema_version": MANIFEST_SCHEMA,
                "contract_sha256": contract_sha256(contract),
                "norm_stats_path": str(stats.resolve()),
                "norm_stats_sha256": digest,
            }
            norm_manifest_path(stats).write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(
                validate_norm_stats(stats, contract)["norm_stats_sha256"],
                digest,
            )
            stats.write_text('{"count": 2}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stale"):
                validate_norm_stats(stats, contract)


if __name__ == "__main__":
    unittest.main()
