from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _stringify_mapping_list(values: list[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError(f"Expected a mapping, got {value!r}")
        result.append(repr(value))
    return result


def _finite(value: Any) -> bool:
    try:
        import torch

        return bool(torch.isfinite(value).all())
    except (ImportError, TypeError):
        return all(math.isfinite(float(item)) for item in value)


def smoke_loader(
    upstream_root: Path,
    training_config: Path,
    *,
    sample_indices: list[int] | None = None,
) -> dict[str, Any]:
    upstream = upstream_root.expanduser().resolve()
    sys.path.insert(0, str(upstream))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from lingbotvla.data import build_vla_dataset
    from lingbotvla.data.vla_data.utils import FeatureTransform

    payload = yaml.safe_load(training_config.read_text(encoding="utf-8"))
    data = payload["data"]
    train = payload["train"]
    dataset_config = SimpleNamespace(
        data_name=data["data_name"],
        train_path=data["train_path"],
        robot_config_root=data["robot_config_root"],
        joints=_stringify_mapping_list(data["joints"]),
        cameras=list(data["cameras"]),
        norm_type=_stringify_mapping_list(data["norm_type"]),
        prompt_type=data.get("prompt_type", "global"),
        chunk_size=int(train.get("chunk_size", 50)),
        img_size=int(data.get("img_size", 256)),
        image_augment=False,
        use_future_image=False,
    )
    model_shape = SimpleNamespace(
        max_state_dim=int(train["max_state_dim"]),
        max_action_dim=int(train["max_action_dim"]),
        return_image_grid_thw=False,
    )
    robot_config = Path(data["robot_config_root"]) / f"{data['data_name']}.yaml"
    FeatureTransform(
        robot_config,
        dataset_config,
        model_shape,
        processor=None,
        disabled_image_features=True,
        do_nomalize=False,
        chunk_size=dataset_config.chunk_size,
        return_item_befor_padding=False,
        use_future_image=False,
    )
    dataset = build_vla_dataset(
        dataset_config=dataset_config,
        model_config=SimpleNamespace(tokenizer_path=""),
        config=model_shape,
        processor=None,
        do_nomalize=False,
        return_item=True,
        disabled_image_features=True,
        use_depth_align=False,
    )
    indices = sample_indices or [0, len(dataset) // 2, len(dataset) - 1]
    samples = []
    for index in indices:
        item = dataset[index]
        arm_state = item["observation.state.arm.position"]
        gripper_state = item["observation.state.effector.position"]
        arm_action = item["action.arm.position"]
        gripper_action = item["action.effector.position"]
        if not all(
            _finite(value)
            for value in (arm_state, gripper_state, arm_action, gripper_action)
        ):
            raise ValueError(f"Non-finite mapped sample at index {index}")
        if tuple(arm_state.shape) != (7,):
            raise ValueError(f"Unexpected arm state shape: {tuple(arm_state.shape)}")
        if tuple(gripper_state.shape) != (1,):
            raise ValueError(
                f"Unexpected gripper state shape: {tuple(gripper_state.shape)}"
            )
        if tuple(arm_action.shape) != (dataset_config.chunk_size, 7):
            raise ValueError(f"Unexpected arm action shape: {tuple(arm_action.shape)}")
        if tuple(gripper_action.shape) != (dataset_config.chunk_size, 1):
            raise ValueError(
                f"Unexpected gripper action shape: {tuple(gripper_action.shape)}"
            )
        samples.append(
            {
                "index": index,
                "arm_state_shape": list(arm_state.shape),
                "gripper_state_shape": list(gripper_state.shape),
                "arm_action_shape": list(arm_action.shape),
                "gripper_action_shape": list(gripper_action.shape),
                "task": item.get("task"),
            }
        )
    return {
        "status": "ok",
        "dataset_length": len(dataset),
        "chunk_size": dataset_config.chunk_size,
        "samples": samples,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test the official VLA loader")
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=os.environ.get("LINGBOT_UPSTREAM_ROOT"),
        required=os.environ.get("LINGBOT_UPSTREAM_ROOT") is None,
    )
    parser.add_argument("--training-config", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = smoke_loader(args.upstream_root, args.training_config)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
