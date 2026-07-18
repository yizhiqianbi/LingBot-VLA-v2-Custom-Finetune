import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from lingbot_vla_finetune.audit import audit_dataset, write_layout_acceptance
from lingbot_vla_finetune.contract import contract_sha256


CAMERAS = [
    "observation.images.left_eye",
    "observation.images.right_eye",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
]
TASK = "Use the right arm to put the object in its category."
REVISION = "1" * 40


class DatasetAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "dataset"
        self.contract_path = Path(self.temp.name) / "contract.yaml"
        self._write_fixture()

    def tearDown(self):
        self.temp.cleanup()

    def _contract(self):
        return {
            "schema_version": "lingbot-custom-dataset-contract-v1",
            "dataset": {
                "repo_id": "example/test",
                "revision": REVISION,
                "lerobot_version": "v2.1",
                "fps": 28,
                "episodes": 2,
                "frames": 12,
                "task": TASK,
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
                "cameras": CAMERAS,
            },
            "training_mapping": {
                "owner_confirmation_required": True,
                "states": {},
                "actions": {},
            },
            "camera_mapping": {},
            "acceptance_assumptions": ["fixture assumption"],
            "audit_thresholds": {
                "timestamp_fps_warning_relative_tolerance": 0.02,
                "timestamp_fps_error_relative_tolerance": 0.05,
                "next_state_max_error": 1e-7,
                "independent_action_min_mean_error": 1e-4,
                "minimum_video_bytes": 10,
            },
        }

    def _write_fixture(self):
        (self.root / "meta").mkdir(parents=True)
        (self.root / "data" / "chunk-000").mkdir(parents=True)
        info = {
            "codebase_version": "v2.1",
            "total_episodes": 2,
            "total_frames": 12,
            "chunks_size": 1000,
            "fps": 28,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [15],
                },
                "action": {"dtype": "float32", "shape": [15]},
                **{
                    camera: {
                        "dtype": "video",
                        "shape": [8, 8, 3],
                    }
                    for camera in CAMERAS
                },
            },
        }
        (self.root / "meta" / "info.json").write_text(
            json.dumps(info), encoding="utf-8"
        )
        episodes = []
        for episode_index in range(2):
            length = 6
            episodes.append(
                {
                    "episode_index": episode_index,
                    "length": length,
                    "tasks": [TASK],
                }
            )
            base = episode_index * 0.2
            states = np.stack(
                [
                    np.linspace(base + step, base + step + 0.14, 15)
                    for step in np.arange(length) * 0.01
                ]
            ).astype(np.float32)
            actions = states.copy()
            actions[:-1, :7] = states[1:, :7] + 0.05
            actions[:-1, 7:15] = states[1:, 7:15]
            table = pa.table(
                {
                    "observation.state": states.tolist(),
                    "action": actions.tolist(),
                    "timestamp": [step / 28.0 for step in range(length)],
                    "frame_index": list(range(length)),
                    "episode_index": [episode_index] * length,
                    "task_index": [0] * length,
                }
            )
            pq.write_table(
                table,
                self.root
                / "data"
                / "chunk-000"
                / f"episode_{episode_index:06d}.parquet",
            )
            for camera in CAMERAS:
                path = (
                    self.root
                    / "videos"
                    / "chunk-000"
                    / camera
                    / f"episode_{episode_index:06d}.mp4"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"0" * 32)

        (self.root / "meta" / "episodes.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in episodes),
            encoding="utf-8",
        )
        (self.root / "meta" / "tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task": TASK}) + "\n",
            encoding="utf-8",
        )
        receipt = (
            self.root
            / ".cache"
            / "huggingface"
            / "download"
            / "meta"
            / "info.json.metadata"
        )
        receipt.parent.mkdir(parents=True)
        receipt.write_text(REVISION + "\netag\n", encoding="utf-8")
        self.contract_path.write_text(
            yaml.safe_dump(self._contract(), sort_keys=False), encoding="utf-8"
        )

    def test_valid_fixture_passes_and_can_write_acceptance(self):
        report = audit_dataset(self.root, self.contract_path)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["metrics"]["row_count"], 12)
        self.assertEqual(report["metrics"]["passive_next_state_max_error"], 0.0)
        acceptance_path = Path(self.temp.name) / "acceptance.json"
        acceptance = write_layout_acceptance(
            acceptance_path, report, self.contract_path
        )
        self.assertEqual(
            acceptance["contract_sha256"], contract_sha256(self.contract_path)
        )
        self.assertTrue(acceptance_path.is_file())

    def test_modified_passive_action_is_rejected(self):
        path = self.root / "data" / "chunk-000" / "episode_000000.parquet"
        table = pq.read_table(path)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        actions[0, 8] += 0.01
        replacement = table.set_column(
            table.schema.get_field_index("action"),
            "action",
            pa.array(actions.tolist()),
        )
        pq.write_table(replacement, path)
        report = audit_dataset(self.root, self.contract_path)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(
            any(
                error.startswith("passive_next_state_relation_failed")
                for error in report["errors"]
            )
        )

    def test_small_timestamp_rate_deviation_is_a_warning(self):
        path = self.root / "data" / "chunk-000" / "episode_000000.parquet"
        table = pq.read_table(path)
        replacement = table.set_column(
            table.schema.get_field_index("timestamp"),
            "timestamp",
            pa.array([step / 28.7 for step in range(table.num_rows)]),
        )
        pq.write_table(replacement, path)
        report = audit_dataset(self.root, self.contract_path)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["metrics"]["fps_warning_episodes"], [0])
        self.assertTrue(
            any(
                warning.startswith("timestamp_fps_deviation:0")
                for warning in report["warnings"]
            )
        )


if __name__ == "__main__":
    unittest.main()
