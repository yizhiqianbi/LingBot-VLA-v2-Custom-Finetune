from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from .contract import DEFAULT_CONTRACT_PATH, contract_sha256, load_contract


MANIFEST_SCHEMA = "lingbot-norm-stats-v1"


def norm_manifest_path(norm_path: Path) -> Path:
    return norm_path.with_suffix(norm_path.suffix + ".manifest.json")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping_list(values: list[Any]) -> list[str]:
    result = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError(f"Expected mapping in training config, got {value!r}")
        result.append(repr(value))
    return result


def _collate_numeric(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    return {
        key: torch.stack([item[key] for item in batch])
        for key, value in batch[0].items()
        if isinstance(value, torch.Tensor)
    }


def _runtime_is_confirmed(
    runtime_manifest_path: Path,
    *,
    allow_unconfirmed: bool,
) -> dict[str, Any]:
    try:
        payload = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot read runtime manifest {runtime_manifest_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Runtime manifest must be a JSON object")
    if not payload.get("layout_confirmed") and not allow_unconfirmed:
        raise ValueError(
            "Runtime layout is unconfirmed; rerun the dataset audit with "
            "--accept-inferred-layout after checking the joint semantics"
        )
    return payload


def compute_norm_stats(
    upstream_root: Path,
    training_config: Path,
    runtime_manifest_path: Path,
    output_path: Path,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    *,
    batch_size: int = 128,
    num_workers: int = 4,
    allow_unconfirmed: bool = False,
) -> dict[str, Any]:
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    upstream = upstream_root.expanduser().resolve()
    sys.path.insert(0, str(upstream))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    runtime = _runtime_is_confirmed(
        runtime_manifest_path,
        allow_unconfirmed=allow_unconfirmed,
    )
    contract = load_contract(contract_path)
    if runtime.get("contract_sha256") != contract_sha256(contract_path):
        raise ValueError("Runtime manifest has a stale contract hash")

    from torch.utils.data import DataLoader

    from lingbotvla.data import build_vla_dataset
    from lingbotvla.utils import normalize

    payload = yaml.safe_load(training_config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Training config must be a YAML mapping")
    data = payload["data"]
    train = payload["train"]
    chunk_size = int(train["chunk_size"])
    dataset_config = SimpleNamespace(
        data_name=data["data_name"],
        train_path=data["train_path"],
        robot_config_root=data["robot_config_root"],
        joints=_mapping_list(data["joints"]),
        cameras=list(data["cameras"]),
        norm_type=_mapping_list(data["norm_type"]),
        prompt_type=data.get("prompt_type", "global"),
        chunk_size=chunk_size,
        img_size=int(data.get("img_size", 256)),
        image_augment=False,
        use_future_image=False,
    )
    dataset = build_vla_dataset(
        dataset_config=dataset_config,
        model_config=None,
        config=None,
        processor=None,
        do_nomalize=False,
        return_item=True,
        disabled_image_features=True,
        use_depth_align=False,
    )
    state_features = list(dataset.state_features)
    action_features = list(dataset.action_features)
    features = state_features + action_features
    stats = {feature: normalize.RunningStats() for feature in features}
    counts = {feature: 0 for feature in features}

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_numeric,
        persistent_workers=num_workers > 0,
        pin_memory=False,
    )
    for batch in loader:
        for feature in state_features:
            values = batch[feature].numpy()
            matrix = values.reshape(-1, values.shape[-1])
            stats[feature].update(matrix)
            counts[feature] += matrix.shape[0]
        for feature in action_features:
            values = batch[feature].numpy()
            matrix = values.reshape(-1, values.shape[-1])
            stats[feature].update(matrix)
            counts[feature] += matrix.shape[0]

    norm_stats = {
        feature: running.get_statistics() for feature, running in stats.items()
    }
    for feature, values in norm_stats.items():
        for field in ("mean", "std", "min", "max", "q01", "q99"):
            array = getattr(values, field)
            if array is None or not all(math.isfinite(float(item)) for item in array):
                raise ValueError(
                    f"Non-finite normalization statistic: {feature}.{field}"
                )

    output = output_path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    normalize.save(temporary, norm_stats, counts[state_features[0]])
    temporary.replace(output)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": contract_sha256(contract_path),
        "dataset_repo_id": contract["dataset"]["repo_id"],
        "dataset_revision": contract["dataset"]["revision"],
        "prepared_dataset_root": runtime["prepared_dataset_root"],
        "upstream_revision": runtime["upstream_revision"],
        "layout_confirmed": bool(runtime["layout_confirmed"]),
        "chunk_size": chunk_size,
        "dataset_length": len(dataset),
        "state_features": state_features,
        "action_features": action_features,
        "feature_counts": counts,
        "norm_stats_path": str(output),
        "norm_stats_sha256": _sha256(output),
    }
    _atomic_write_json(norm_manifest_path(output), manifest)
    return manifest


def validate_norm_stats(
    norm_path: Path,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    path = norm_path.expanduser().resolve()
    manifest_path = norm_manifest_path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot read norm stats manifest {manifest_path}: {exc}"
        ) from exc
    expected = {
        "schema_version": MANIFEST_SCHEMA,
        "contract_sha256": contract_sha256(contract_path),
        "norm_stats_path": str(path),
        "norm_stats_sha256": _sha256(path),
    }
    mismatches = [
        f"{key}:{manifest.get(key)!r}:{wanted!r}"
        for key, wanted in expected.items()
        if manifest.get(key) != wanted
    ]
    if mismatches:
        raise ValueError(f"Normalization stats are stale: {mismatches}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute LingBot normalization stats without importing model kernels"
    )
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=os.environ.get("LINGBOT_UPSTREAM_ROOT"),
    )
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--allow-unconfirmed", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.upstream_root is None:
        raise SystemExit("--upstream-root or LINGBOT_UPSTREAM_ROOT is required")
    try:
        result = compute_norm_stats(
            args.upstream_root,
            args.training_config,
            args.runtime_manifest,
            args.output,
            args.contract,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            allow_unconfirmed=args.allow_unconfirmed,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
