from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audit import audit_dataset
from .contract import (
    DEFAULT_CONTRACT_PATH,
    PROJECT_ROOT,
    contract_sha256,
    load_contract,
)


RECEIPT_SCHEMA = "lingbot-prepared-dataset-v1"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _assert_outside_project(path: Path, label: str) -> None:
    project_root = PROJECT_ROOT.resolve()
    if path == project_root or project_root in path.parents:
        raise ValueError(f"{label} must be outside the code repository: {path}")


def _prepared_spec(contract: dict[str, Any]) -> dict[str, Any]:
    spec = contract.get("prepared_dataset")
    if not isinstance(spec, dict):
        raise ValueError("Contract is missing prepared_dataset")
    return spec


def _feature_width(info: dict[str, Any], key: str) -> int | None:
    feature = (info.get("features") or {}).get(key) or {}
    shape = feature.get("shape")
    return int(shape[-1]) if isinstance(shape, list) and shape else None


def validate_prepared_metadata(
    dataset_root: Path,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    root = dataset_root.expanduser().resolve()
    contract = load_contract(contract_path)
    expected = contract["dataset"]
    prepared = _prepared_spec(contract)
    info = _read_json(root / "meta" / "info.json")

    checks = {
        "codebase_version": prepared["lerobot_version"],
        "fps": expected["fps"],
        "total_episodes": expected["episodes"],
        "total_frames": expected["frames"],
    }
    mismatches = [
        f"{key}:{info.get(key)}:{wanted}"
        for key, wanted in checks.items()
        if info.get(key) != wanted
    ]
    features = info.get("features") or {}
    required_features = {
        contract["raw_schema"]["state_key"],
        contract["raw_schema"]["action_key"],
        *contract["raw_schema"]["cameras"],
    }
    missing_features = sorted(required_features - set(features))
    width_mismatches = []
    for key, expected_width in (
        (contract["raw_schema"]["state_key"], contract["raw_schema"]["state_width"]),
        (
            contract["raw_schema"]["action_key"],
            contract["raw_schema"]["action_width"],
        ),
    ):
        actual_width = _feature_width(info, key)
        if actual_width != int(expected_width):
            width_mismatches.append(f"{key}:{actual_width}:{expected_width}")
    required_globs = {
        "data": "data/**/*.parquet",
        "videos": "videos/**/*.mp4",
        "episodes": "meta/episodes/**/*.parquet",
    }
    file_counts = {
        name: sum(1 for path in root.glob(pattern) if path.is_file())
        for name, pattern in required_globs.items()
    }
    file_counts.update(
        {
            "stats": int((root / "meta" / "stats.json").is_file()),
            "tasks": int((root / "meta" / "tasks.parquet").is_file()),
        }
    )
    empty_groups = sorted(name for name, count in file_counts.items() if count == 0)
    if mismatches or missing_features or width_mismatches or empty_groups:
        raise ValueError(
            "Prepared dataset validation failed: "
            f"metadata={mismatches}, missing_features={missing_features}, "
            f"widths={width_mismatches}, empty_groups={empty_groups}"
        )
    return {
        "codebase_version": info["codebase_version"],
        "fps": info["fps"],
        "episodes": info["total_episodes"],
        "frames": info["total_frames"],
        "file_counts": file_counts,
    }


def read_prepare_receipt(
    dataset_root: Path,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    root = dataset_root.expanduser().resolve()
    contract = load_contract(contract_path)
    prepared = _prepared_spec(contract)
    receipt_path = root / str(prepared["receipt_name"])
    receipt = _read_json(receipt_path)
    expected = {
        "schema_version": RECEIPT_SCHEMA,
        "dataset_repo_id": contract["dataset"]["repo_id"],
        "dataset_revision": contract["dataset"]["revision"],
        "contract_sha256": contract_sha256(contract_path),
        "lerobot_package_version": prepared["lerobot_package_version"],
        "prepared_dataset_root": str(root),
    }
    mismatches = [
        f"{key}:{receipt.get(key)!r}:{wanted!r}"
        for key, wanted in expected.items()
        if receipt.get(key) != wanted
    ]
    if mismatches:
        raise ValueError(f"Prepared dataset receipt is stale: {mismatches}")
    validate_prepared_metadata(root, contract_path)
    return receipt


def _convert_v21_to_v30(
    source_root: Path,
    stage_root: Path,
    *,
    data_file_size_mb: int,
    video_file_size_mb: int,
) -> None:
    from lerobot.datasets.v30.convert_dataset_v21_to_v30 import (
        convert_data,
        convert_episodes_metadata,
        convert_info,
        convert_tasks,
        convert_videos,
    )

    convert_info(source_root, stage_root, data_file_size_mb, video_file_size_mb)
    convert_tasks(source_root, stage_root)
    episodes_metadata = convert_data(source_root, stage_root, data_file_size_mb)
    video_metadata = convert_videos(
        source_root,
        stage_root,
        video_file_size_mb,
    )
    convert_episodes_metadata(
        source_root,
        stage_root,
        episodes_metadata,
        video_metadata,
    )


def prepare_dataset(
    source_root: Path,
    output_root: Path,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    *,
    data_file_size_mb: int = 100,
    video_file_size_mb: int = 500,
) -> dict[str, Any]:
    source = source_root.expanduser().resolve()
    output = output_root.expanduser().resolve()
    if source == output:
        raise ValueError("Source and prepared dataset directories must differ")
    _assert_outside_project(source, "Source dataset")
    _assert_outside_project(output, "Prepared dataset")
    if source in output.parents or output in source.parents:
        raise ValueError("Source and prepared dataset directories cannot be nested")

    contract = load_contract(contract_path)
    prepared = _prepared_spec(contract)
    wanted_lerobot = str(prepared["lerobot_package_version"])
    actual_lerobot = importlib.metadata.version("lerobot")
    if actual_lerobot != wanted_lerobot:
        raise ValueError(
            f"LeRobot version mismatch: {actual_lerobot} != {wanted_lerobot}"
        )

    if output.exists():
        try:
            receipt = read_prepare_receipt(output, contract_path)
            return {"status": "reused", **receipt}
        except ValueError as stale_error:
            prepared_metrics = validate_prepared_metadata(output, contract_path)
            receipt_path = output / str(prepared["receipt_name"])
            receipt = _read_json(receipt_path)
            refresh_identity = {
                "schema_version": RECEIPT_SCHEMA,
                "dataset_repo_id": contract["dataset"]["repo_id"],
                "dataset_revision": contract["dataset"]["revision"],
                "source_dataset_root": str(source),
                "prepared_dataset_root": str(output),
                "prepared_lerobot_version": prepared["lerobot_version"],
                "lerobot_package_version": wanted_lerobot,
            }
            identity_mismatches = [
                key
                for key, wanted in refresh_identity.items()
                if receipt.get(key) != wanted
            ]
            if identity_mismatches:
                raise stale_error
            refresh_audit = audit_dataset(source, contract_path)
            if refresh_audit["status"] != "passed":
                raise ValueError(
                    "Cannot refresh a prepared receipt after source audit failure: "
                    + "; ".join(refresh_audit["errors"])
                ) from stale_error
            receipt.update(
                {
                    "contract_sha256": contract_sha256(contract_path),
                    "receipt_refreshed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_audit": {
                        "status": refresh_audit["status"],
                        "row_count": refresh_audit["metrics"]["row_count"],
                        "video_file_count": refresh_audit["metrics"][
                            "video_file_count"
                        ],
                        "passive_next_state_max_error": refresh_audit["metrics"][
                            "passive_next_state_max_error"
                        ],
                    },
                    "prepared_metrics": prepared_metrics,
                }
            )
            _atomic_write_json(receipt_path, receipt)
            read_prepare_receipt(output, contract_path)
            return {"status": "receipt_refreshed", **receipt}

    audit = audit_dataset(source, contract_path)
    if audit["status"] != "passed":
        raise ValueError(
            "Source audit failed; inspect the audit report before conversion: "
            + "; ".join(audit["errors"])
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    if stage.exists():
        raise ValueError(f"Unexpected conversion staging path exists: {stage}")

    try:
        _convert_v21_to_v30(
            source,
            stage,
            data_file_size_mb=data_file_size_mb,
            video_file_size_mb=video_file_size_mb,
        )
        metrics = validate_prepared_metadata(stage, contract_path)
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_repo_id": contract["dataset"]["repo_id"],
            "dataset_revision": contract["dataset"]["revision"],
            "contract_sha256": contract_sha256(contract_path),
            "source_dataset_root": str(source),
            "prepared_dataset_root": str(output),
            "source_lerobot_version": contract["dataset"]["lerobot_version"],
            "prepared_lerobot_version": prepared["lerobot_version"],
            "lerobot_package_version": actual_lerobot,
            "converter": "lerobot.datasets.v30.convert_dataset_v21_to_v30",
            "conversion_parameters": {
                "data_file_size_mb": data_file_size_mb,
                "video_file_size_mb": video_file_size_mb,
            },
            "source_audit": {
                "status": audit["status"],
                "row_count": audit["metrics"]["row_count"],
                "video_file_count": audit["metrics"]["video_file_count"],
                "passive_next_state_max_error": audit["metrics"][
                    "passive_next_state_max_error"
                ],
            },
            "prepared_metrics": metrics,
        }
        _atomic_write_json(
            stage / str(prepared["receipt_name"]),
            receipt,
        )
        stage.replace(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    read_prepare_receipt(output, contract_path)
    return {"status": "converted", **receipt}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert the audited LeRobot v2.1 source to a v3.0 training copy"
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=os.environ.get("LINGBOT_SOURCE_DATASET_ROOT"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=os.environ.get("LINGBOT_TRAIN_DATASET_ROOT"),
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--data-file-size-mb", type=int, default=100)
    parser.add_argument("--video-file-size-mb", type=int, default=500)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.source_root is None or args.output_root is None:
        raise SystemExit(
            "--source-root/LINGBOT_SOURCE_DATASET_ROOT and "
            "--output-root/LINGBOT_TRAIN_DATASET_ROOT are required"
        )
    if args.data_file_size_mb <= 0 or args.video_file_size_mb <= 0:
        raise SystemExit("Conversion file-size limits must be positive")
    try:
        result = prepare_dataset(
            args.source_root,
            args.output_root,
            args.contract,
            data_file_size_mb=args.data_file_size_mb,
            video_file_size_mb=args.video_file_size_mb,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
