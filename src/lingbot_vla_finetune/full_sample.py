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

from .norm_stats import validate_norm_stats


def _mapping_list(values: list[Any]) -> list[str]:
    if not all(isinstance(value, dict) for value in values):
        raise ValueError("Expected mapping lists for joints and norm_type")
    return [repr(value) for value in values]


def _shape(value: Any) -> list[int]:
    return list(value.shape)


def _finite(value: Any) -> bool:
    import torch

    return bool(torch.isfinite(value).all())


def smoke_full_sample(
    upstream_root: Path,
    training_config: Path,
    norm_stats_path: Path,
    *,
    index: int = 0,
    allow_unconfirmed: bool = False,
) -> dict[str, Any]:
    upstream = upstream_root.expanduser().resolve()
    sys.path.insert(0, str(upstream))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    norm_manifest = validate_norm_stats(norm_stats_path)
    if not norm_manifest.get("layout_confirmed") and not allow_unconfirmed:
        raise ValueError("Normalization stats were computed with an unconfirmed layout")

    from transformers import AutoProcessor

    from lingbotvla.data import build_vla_dataset

    payload = yaml.safe_load(training_config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Training config must be a YAML mapping")
    data = payload["data"]
    train = payload["train"]
    model = payload["model"]
    processor = AutoProcessor.from_pretrained(
        model["tokenizer_path"],
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
    )
    dataset_config = SimpleNamespace(
        data_name=data["data_name"],
        train_path=data["train_path"],
        robot_config_root=data["robot_config_root"],
        joints=_mapping_list(data["joints"]),
        cameras=list(data["cameras"]),
        norm_type=_mapping_list(data["norm_type"]),
        prompt_type=data.get("prompt_type", "global"),
        chunk_size=int(train["chunk_size"]),
        img_size=int(data.get("img_size", 256)),
        image_augment=False,
        use_future_image=bool(data.get("use_future_image", False)),
    )
    model_shape = SimpleNamespace(
        max_state_dim=int(train["max_state_dim"]),
        max_action_dim=int(train["max_action_dim"]),
        tokenizer_max_length=int(train["tokenizer_max_length"]),
        return_image_grid_thw=True,
        qwen3vl_use_vision_boundaries=True,
        use_qwen3_chat_template=True,
    )
    dataset = build_vla_dataset(
        dataset_config=dataset_config,
        model_config=SimpleNamespace(tokenizer_path=model["tokenizer_path"]),
        config=model_shape,
        processor=processor,
        do_nomalize=True,
        return_item=False,
        disabled_image_features=False,
        use_depth_align=True,
    )
    if index < 0 or index >= len(dataset):
        raise ValueError(f"Sample index out of range: {index}")
    item = dataset[index]
    tensor_keys = [
        "images",
        "img_masks",
        "state",
        "lang_tokens",
        "lang_masks",
        "actions",
        "action_is_pad",
        "joint_mask",
        "state_joint_mask",
        "action_joint_mask",
        "pil_images",
        "future_pil_images",
        "image_grid_thw",
    ]
    missing = [key for key in tensor_keys if key not in item]
    if missing:
        raise ValueError(f"Full sample is missing fields: {missing}")
    non_finite = [
        key
        for key in tensor_keys
        if key not in {"img_masks", "lang_masks", "action_is_pad"}
        and not _finite(item[key])
    ]
    if non_finite:
        raise ValueError(f"Full sample has non-finite tensors: {non_finite}")

    expected_shapes = {
        "state": (int(train["max_state_dim"]),),
        "actions": (int(train["chunk_size"]), int(train["max_action_dim"])),
        "action_is_pad": (int(train["chunk_size"]),),
        "state_joint_mask": (int(train["max_state_dim"]),),
        "action_joint_mask": (int(train["max_action_dim"]),),
    }
    shape_errors = {
        key: {"actual": tuple(item[key].shape), "expected": expected}
        for key, expected in expected_shapes.items()
        if tuple(item[key].shape) != expected
    }
    if shape_errors:
        raise ValueError(f"Unexpected full sample shapes: {shape_errors}")
    if int(item["img_masks"].sum()) != len(data["cameras"]):
        raise ValueError("Not all configured cameras decoded")
    if int(item["state_joint_mask"].sum()) != 8:
        raise ValueError("Expected exactly 8 active state dimensions")
    if int(item["action_joint_mask"].sum()) != 8:
        raise ValueError("Expected exactly 8 active action dimensions")
    if float(item["pil_images"].float().std()) <= 0.0:
        raise ValueError("Decoded current images are blank")
    if float(item["future_pil_images"].float().std()) <= 0.0:
        raise ValueError("Decoded future images are blank")

    return {
        "status": "ok",
        "dataset_length": len(dataset),
        "index": index,
        "shapes": {key: _shape(item[key]) for key in tensor_keys},
        "active_state_dimensions": int(item["state_joint_mask"].sum()),
        "active_action_dimensions": int(item["action_joint_mask"].sum()),
        "current_image_std": float(item["pil_images"].float().std()),
        "future_image_std": float(item["future_pil_images"].float().std()),
        "normalized_state_abs_max": float(item["state"].abs().max()),
        "normalized_action_abs_max": float(item["actions"].abs().max()),
        "language_token_count": int(item["lang_masks"].sum()),
        "all_finite": all(
            math.isfinite(float(value))
            for value in (
                item["state"].abs().max(),
                item["actions"].abs().max(),
            )
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode and transform one complete LingBot training sample"
    )
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=os.environ.get("LINGBOT_UPSTREAM_ROOT"),
    )
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--allow-unconfirmed", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.upstream_root is None:
        raise SystemExit("--upstream-root or LINGBOT_UPSTREAM_ROOT is required")
    try:
        result = smoke_full_sample(
            args.upstream_root,
            args.training_config,
            args.norm_stats,
            index=args.index,
            allow_unconfirmed=args.allow_unconfirmed,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
