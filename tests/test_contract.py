import tempfile
import unittest
from pathlib import Path

import yaml

from lingbot_vla_finetune.contract import (
    DEFAULT_CONTRACT_PATH,
    PROJECT_ROOT,
    ContractError,
    contract_sha256,
    load_contract,
)


class ContractTest(unittest.TestCase):
    def test_checked_in_contract_has_pinned_sources(self):
        contract = load_contract(DEFAULT_CONTRACT_PATH)
        self.assertEqual(
            contract["dataset"]["repo_id"],
            "jokeru/take_wrong_item_right_arm",
        )
        self.assertEqual(len(contract["dataset"]["revision"]), 40)
        self.assertTrue(contract["training_mapping"]["owner_confirmation_required"])
        self.assertFalse(contract["camera_wiring"]["raw_names_match_physical_mounts"])
        self.assertEqual(
            contract["camera_mapping"],
            {
                "camera_top": "observation.images.left_eye",
                "camera_wrist_left": "observation.images.right_eye",
                "camera_wrist_right": "observation.images.right_wrist",
                "omitted": ["observation.images.left_wrist"],
            },
        )
        self.assertEqual(len(contract_sha256(DEFAULT_CONTRACT_PATH)), 64)

    def test_v2_policy_uses_native_vision_freeze_flag(self):
        template = yaml.safe_load(
            (
                PROJECT_ROOT
                / "configs"
                / "vla"
                / "take_wrong_item_right_arm.yaml.in"
            ).read_text(encoding="utf-8")
        )
        self.assertFalse(template["train"]["freeze_vit"])
        self.assertTrue(template["train"]["freeze_vision_encoder"])
        self.assertEqual(template["train"]["align_params"]["visual_steps"], 5000)

    def test_invalid_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("schema_version: bad\n", encoding="utf-8")
            with self.assertRaises(ContractError):
                load_contract(path)


if __name__ == "__main__":
    unittest.main()
