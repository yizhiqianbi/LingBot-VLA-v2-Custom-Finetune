import tempfile
import unittest
from pathlib import Path

from lingbot_vla_finetune.contract import (
    DEFAULT_CONTRACT_PATH,
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
        self.assertEqual(len(contract_sha256(DEFAULT_CONTRACT_PATH)), 64)

    def test_invalid_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("schema_version: bad\n", encoding="utf-8")
            with self.assertRaises(ContractError):
                load_contract(path)


if __name__ == "__main__":
    unittest.main()
