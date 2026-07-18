from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .contract import (
    DEFAULT_ACCEPTANCE_PATH,
    DEFAULT_CONTRACT_PATH,
    contract_sha256,
    format_lerobot_path,
    load_contract,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield payload


def _feature_width(info: dict[str, Any], key: str) -> int | None:
    feature = (info.get("features") or {}).get(key) or {}
    shape = feature.get("shape")
    return int(shape[-1]) if isinstance(shape, list) and shape else None


def _local_hf_revision(root: Path) -> str | None:
    metadata = (
        root / ".cache" / "huggingface" / "download" / "meta" / "info.json.metadata"
    )
    if not metadata.is_file():
        return None
    lines = metadata.read_text(encoding="utf-8").splitlines()
    return lines[0].strip() if lines else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _decode_video_samples(
    root: Path,
    info: dict[str, Any],
    cameras: list[str],
    episode_lengths: dict[int, int],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    try:
        import cv2
    except ImportError:
        return [], [], ["opencv_not_installed_video_decode_failed"]

    chunk_size = int(info.get("chunks_size") or 1000)
    template = str(info["video_path"])
    results: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []
    episode_indices = sorted(episode_lengths)
    sampled_indices = sorted(
        {
            episode_indices[0],
            episode_indices[len(episode_indices) // 2],
            episode_indices[-1],
        }
    )
    for episode_index in sampled_indices:
        for camera in cameras:
            path = root / format_lerobot_path(
                template,
                episode_index,
                chunk_size,
                video_key=camera,
            )
            capture = cv2.VideoCapture(str(path))
            opened = bool(capture.isOpened())
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
            ok, frame = capture.read() if opened else (False, None)
            capture.release()
            result = {
                "episode_index": episode_index,
                "camera": camera,
                "path": str(path),
                "opened": opened,
                "first_frame_decoded": bool(ok),
                "frame_count": frame_count,
                "first_frame_shape": list(frame.shape) if ok else None,
            }
            results.append(result)
            if not opened or not ok:
                errors.append(f"video_decode_failed:{episode_index}:{camera}:{path}")
            expected = episode_lengths[episode_index]
            if frame_count and abs(frame_count - expected) > 2:
                errors.append(
                    "video_frame_count_mismatch:"
                    f"{episode_index}:{camera}:{frame_count}:{expected}"
                )
    return results, warnings, errors


def audit_dataset(
    dataset_root: Path,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    *,
    decode_videos: bool = False,
) -> dict[str, Any]:
    import numpy as np
    import pyarrow.parquet as pq

    root = dataset_root.expanduser().resolve()
    contract = load_contract(contract_path)
    expected = contract["dataset"]
    raw = contract["raw_schema"]
    thresholds = contract["audit_thresholds"]
    errors: list[str] = []
    warnings: list[str] = []

    required_metadata = [
        root / "meta" / "info.json",
        root / "meta" / "episodes.jsonl",
        root / "meta" / "tasks.jsonl",
    ]
    missing_metadata = [str(path) for path in required_metadata if not path.is_file()]
    if missing_metadata:
        return {
            "schema_version": "lingbot-custom-audit-v1",
            "status": "failed",
            "dataset_root": str(root),
            "contract_sha256": contract_sha256(contract_path),
            "errors": [f"missing_metadata:{path}" for path in missing_metadata],
            "warnings": [],
            "metrics": {},
        }

    info = _read_json(root / "meta" / "info.json")
    episodes = list(_iter_jsonl(root / "meta" / "episodes.jsonl"))
    tasks = list(_iter_jsonl(root / "meta" / "tasks.jsonl"))
    local_revision = _local_hf_revision(root)
    if local_revision is None:
        warnings.append("huggingface_revision_receipt_missing")
    elif local_revision != str(expected["revision"]):
        errors.append(
            f"dataset_revision_mismatch:{local_revision}:{expected['revision']}"
        )

    scalar_checks = {
        "codebase_version": (info.get("codebase_version"), expected["lerobot_version"]),
        "fps": (info.get("fps"), expected["fps"]),
        "total_episodes": (info.get("total_episodes"), expected["episodes"]),
        "total_frames": (info.get("total_frames"), expected["frames"]),
    }
    for name, (actual, wanted) in scalar_checks.items():
        if actual != wanted:
            errors.append(f"metadata_mismatch:{name}:{actual}:{wanted}")

    state_key = str(raw["state_key"])
    action_key = str(raw["action_key"])
    widths = {
        "state": _feature_width(info, state_key),
        "action": _feature_width(info, action_key),
    }
    if widths["state"] != int(raw["state_width"]):
        errors.append(f"state_width_mismatch:{widths['state']}:{raw['state_width']}")
    if widths["action"] != int(raw["action_width"]):
        errors.append(f"action_width_mismatch:{widths['action']}:{raw['action_width']}")

    available_features = set(info.get("features") or {})
    cameras = [str(camera) for camera in raw["cameras"]]
    for camera in cameras:
        if camera not in available_features:
            errors.append(f"camera_feature_missing:{camera}")
    observed_tasks = {
        str(task.get("task")) for task in tasks if task.get("task") is not None
    }
    if observed_tasks != {str(expected["task"])}:
        errors.append(f"task_mismatch:{sorted(observed_tasks)}")

    task_by_index: dict[int, str] = {}
    for task in tasks:
        try:
            task_by_index[int(task["task_index"])] = str(task["task"])
        except (KeyError, TypeError, ValueError):
            errors.append("invalid_task_metadata")

    episode_rows: dict[int, dict[str, Any]] = {}
    for row in episodes:
        try:
            episode_index = int(row["episode_index"])
            if episode_index in episode_rows:
                errors.append(f"duplicate_episode_index:{episode_index}")
            episode_rows[episode_index] = row
        except (KeyError, TypeError, ValueError):
            errors.append("invalid_episode_index")
    expected_indices = set(range(int(expected["episodes"])))
    if set(episode_rows) != expected_indices:
        errors.append("episode_index_set_mismatch")

    chunk_size = int(info.get("chunks_size") or 1000)
    data_template = str(info.get("data_path") or "")
    video_template = str(info.get("video_path") or "")
    all_states: list[Any] = []
    all_actions: list[Any] = []
    passive_errors: list[Any] = []
    active_errors: list[Any] = []
    timestamp_fps: list[float] = []
    timestamp_fps_by_episode: dict[int, float] = {}
    timestamp_warning_episodes: list[int] = []
    row_total = 0
    episode_lengths: dict[int, int] = {}

    for episode_index in sorted(episode_rows):
        episode = episode_rows[episode_index]
        expected_length = int(episode.get("length") or 0)
        episode_tasks = episode.get("tasks")
        if episode_tasks != [str(expected["task"])]:
            errors.append(f"episode_task_mismatch:{episode_index}:{episode_tasks}")
        episode_lengths[episode_index] = expected_length
        data_path = root / format_lerobot_path(data_template, episode_index, chunk_size)
        if not data_path.is_file():
            errors.append(f"parquet_missing:{data_path}")
            continue
        try:
            table = pq.read_table(
                data_path,
                columns=[
                    state_key,
                    action_key,
                    "timestamp",
                    "frame_index",
                    "episode_index",
                    "task_index",
                ],
            )
            states = np.asarray(table[state_key].to_pylist(), dtype=np.float64)
            actions = np.asarray(table[action_key].to_pylist(), dtype=np.float64)
            timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
            frame_indices = np.asarray(table["frame_index"].to_pylist())
            episode_indices = np.asarray(table["episode_index"].to_pylist())
            task_indices = np.asarray(table["task_index"].to_pylist())
        except Exception as exc:
            errors.append(f"parquet_read_failed:{data_path}:{exc}")
            continue

        if table.num_rows != expected_length:
            errors.append(
                f"episode_length_mismatch:{episode_index}:{table.num_rows}:{expected_length}"
            )
        if states.shape != (table.num_rows, int(raw["state_width"])):
            errors.append(f"state_shape_mismatch:{episode_index}:{states.shape}")
            continue
        if actions.shape != (table.num_rows, int(raw["action_width"])):
            errors.append(f"action_shape_mismatch:{episode_index}:{actions.shape}")
            continue
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            errors.append(f"non_finite_state_or_action:{episode_index}")
        if not np.array_equal(frame_indices, np.arange(table.num_rows)):
            errors.append(f"non_contiguous_frame_index:{episode_index}")
        if not np.all(episode_indices == episode_index):
            errors.append(f"parquet_episode_index_mismatch:{episode_index}")
        observed_episode_tasks = {
            task_by_index.get(int(task_index)) for task_index in task_indices
        }
        if observed_episode_tasks != {str(expected["task"])}:
            errors.append(
                f"parquet_task_mismatch:{episode_index}:"
                f"{sorted(str(value) for value in observed_episode_tasks)}"
            )
        delta_t = np.diff(timestamps)
        if len(delta_t) and (delta_t <= 0).any():
            errors.append(f"non_monotonic_timestamp:{episode_index}")
        if len(delta_t):
            measured_fps = 1.0 / float(np.median(delta_t))
            timestamp_fps.append(measured_fps)
            timestamp_fps_by_episode[episode_index] = measured_fps
            relative_error = abs(measured_fps - float(expected["fps"])) / float(
                expected["fps"]
            )
            if relative_error > float(
                thresholds["timestamp_fps_error_relative_tolerance"]
            ):
                errors.append(
                    f"timestamp_fps_mismatch:{episode_index}:{measured_fps:.6f}"
                )
            elif relative_error > float(
                thresholds["timestamp_fps_warning_relative_tolerance"]
            ):
                timestamp_warning_episodes.append(episode_index)
                warnings.append(
                    f"timestamp_fps_deviation:{episode_index}:{measured_fps:.6f}"
                )

        if table.num_rows > 1:
            passive_errors.append(np.abs(actions[:-1, 7:15] - states[1:, 7:15]))
            active_errors.append(np.abs(actions[:-1, 0:7] - states[1:, 0:7]))
        all_states.append(states)
        all_actions.append(actions)
        row_total += table.num_rows

    if row_total != int(expected["frames"]):
        errors.append(f"total_rows_mismatch:{row_total}:{expected['frames']}")

    state_matrix = np.concatenate(all_states) if all_states else np.empty((0, 15))
    action_matrix = np.concatenate(all_actions) if all_actions else np.empty((0, 15))
    passive = np.concatenate(passive_errors) if passive_errors else np.empty((0, 8))
    active = np.concatenate(active_errors) if active_errors else np.empty((0, 7))
    passive_max = float(np.max(passive)) if passive.size else math.inf
    active_mean = float(np.mean(active)) if active.size else 0.0
    if passive_max > float(thresholds["next_state_max_error"]):
        errors.append(f"passive_next_state_relation_failed:{passive_max}")
    if active_mean < float(thresholds["independent_action_min_mean_error"]):
        errors.append(f"active_action_not_independent:{active_mean}")

    video_missing: list[str] = []
    video_bytes = 0
    video_count = 0
    minimum_video_bytes = int(thresholds["minimum_video_bytes"])
    for episode_index in sorted(episode_rows):
        for camera in cameras:
            video_path = root / format_lerobot_path(
                video_template,
                episode_index,
                chunk_size,
                video_key=camera,
            )
            try:
                size = video_path.stat().st_size
            except OSError:
                video_missing.append(str(video_path))
                continue
            if size < minimum_video_bytes:
                video_missing.append(str(video_path))
                continue
            video_count += 1
            video_bytes += size
    if video_missing:
        errors.append(f"missing_or_empty_videos:{len(video_missing)}")

    decode_results: list[dict[str, Any]] = []
    if decode_videos and episode_lengths and not video_missing:
        decode_results, decode_warnings, decode_errors = _decode_video_samples(
            root, info, cameras, episode_lengths
        )
        warnings.extend(decode_warnings)
        errors.extend(decode_errors)

    if bool(contract["training_mapping"].get("owner_confirmation_required")):
        warnings.append("semantic_layout_requires_dataset_owner_confirmation")

    metrics = {
        "local_hf_revision": local_revision,
        "episode_count": len(episode_rows),
        "row_count": row_total,
        "fps_median": float(np.median(timestamp_fps)) if timestamp_fps else None,
        "fps_min": float(np.min(timestamp_fps)) if timestamp_fps else None,
        "fps_max": float(np.max(timestamp_fps)) if timestamp_fps else None,
        "fps_by_episode": {
            str(key): value for key, value in timestamp_fps_by_episode.items()
        },
        "fps_warning_episodes": timestamp_warning_episodes,
        "state_width": widths["state"],
        "action_width": widths["action"],
        "passive_next_state_max_error": passive_max,
        "active_action_to_next_state_mean_error": active_mean,
        "active_action_to_next_state_max_error": (
            float(np.max(active)) if active.size else None
        ),
        "gripper_state_min": (
            float(np.min(state_matrix[:, 14])) if state_matrix.size else None
        ),
        "gripper_state_max": (
            float(np.max(state_matrix[:, 14])) if state_matrix.size else None
        ),
        "camera_count": len(cameras),
        "video_file_count": video_count,
        "video_total_bytes": video_bytes,
        "video_decode_samples": decode_results,
        "state_std": np.std(state_matrix, axis=0).tolist() if state_matrix.size else [],
        "action_std": (
            np.std(action_matrix, axis=0).tolist() if action_matrix.size else []
        ),
    }
    return {
        "schema_version": "lingbot-custom-audit-v1",
        "status": "failed" if errors else "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "dataset_repo_id": expected["repo_id"],
        "dataset_revision": expected["revision"],
        "contract_sha256": contract_sha256(contract_path),
        "errors": errors,
        "warnings": sorted(set(warnings)),
        "metrics": metrics,
    }


def write_layout_acceptance(
    output_path: Path,
    report: dict[str, Any],
    contract_path: Path,
) -> dict[str, Any]:
    if report.get("status") != "passed":
        raise ValueError("Cannot accept a layout whose audit did not pass")
    contract = load_contract(contract_path)
    payload = {
        "schema_version": "lingbot-layout-acceptance-v1",
        "accepted_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_repo_id": contract["dataset"]["repo_id"],
        "dataset_revision": contract["dataset"]["revision"],
        "dataset_root": report["dataset_root"],
        "contract_sha256": contract_sha256(contract_path),
        "assumptions": contract["acceptance_assumptions"],
    }
    _write_json(output_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the custom LeRobot dataset")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=os.environ.get("LINGBOT_SOURCE_DATASET_ROOT"),
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CONTRACT_PATH.parents[1] / "work" / "audit_report.json",
    )
    parser.add_argument("--decode-videos", action="store_true")
    parser.add_argument("--accept-inferred-layout", action="store_true")
    parser.add_argument(
        "--acceptance-output", type=Path, default=DEFAULT_ACCEPTANCE_PATH
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dataset_root is None:
        raise SystemExit("--dataset-root or LINGBOT_SOURCE_DATASET_ROOT is required")
    report = audit_dataset(
        args.dataset_root,
        args.contract,
        decode_videos=args.decode_videos,
    )
    _write_json(args.output, report)
    if args.accept_inferred_layout:
        write_layout_acceptance(args.acceptance_output, report, args.contract)
    summary = {
        "status": report["status"],
        "errors": len(report["errors"]),
        "warnings": report["warnings"],
        "report": str(args.output),
        "acceptance": (
            str(args.acceptance_output) if args.accept_inferred_layout else None
        ),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
