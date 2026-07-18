import json
import tempfile
import unittest
from pathlib import Path

from lingbot_vla_finetune.contract import contract_sha256
from lingbot_vla_finetune.render import _read_acceptance, _render_text


class RenderTest(unittest.TestCase):
    def test_template_requires_every_value(self):
        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / "template.yaml"
            template.write_text("path: ${REQUIRED_PATH}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "REQUIRED_PATH"):
                _render_text(template, {})
            self.assertEqual(
                _render_text(template, {"REQUIRED_PATH": "/tmp/value"}),
                "path: /tmp/value\n",
            )

    def test_stale_acceptance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = root / "contract.yaml"
            contract.write_text(
                """
schema_version: test
dataset:
  repo_id: example/test
  revision: 1111111111111111111111111111111111111111
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
            acceptance = root / "acceptance.json"
            acceptance.write_text(
                json.dumps(
                    {
                        "contract_sha256": "stale",
                        "dataset_revision": "1" * 40,
                    }
                ),
                encoding="utf-8",
            )
            self.assertNotEqual(
                json.loads(acceptance.read_text())["contract_sha256"],
                contract_sha256(contract),
            )
            with self.assertRaisesRegex(ValueError, "Stale layout acceptance"):
                _read_acceptance(acceptance, contract)


if __name__ == "__main__":
    unittest.main()
