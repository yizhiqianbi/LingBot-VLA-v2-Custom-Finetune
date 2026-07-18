from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .contract import (
    DEFAULT_ACCEPTANCE_PATH,
    DEFAULT_CONTRACT_PATH,
    PROJECT_ROOT,
    contract_sha256,
    load_contract,
    load_yaml,
)
from .prepare import read_prepare_receipt
from .norm_stats import validate_norm_stats


TOKEN_PATTERN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _render_text(template: Path, values: dict[str, str]) -> str:
    source = template.read_text(encoding="utf-8")

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values or not values[name]:
            raise ValueError(f"Missing template value: {name}")
        return values[name]

    rendered = TOKEN_PATTERN.sub(replace, source)
    unresolved = sorted(set(TOKEN_PATTERN.findall(rendered)))
    if unresolved:
        raise ValueError(f"Unresolved template values: {unresolved}")
    payload = yaml.safe_load(rendered)
    if not isinstance(payload, dict):
        raise ValueError(f"Rendered template is not a YAML mapping: {template}")
    return rendered


def _read_acceptance(path: Path, contract_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Layout acceptance is missing or invalid. Run audit with "
            "--accept-inferred-layout after confirming the robot indices."
        ) from exc
    contract = load_contract(contract_path)
    expected = {
        "contract_sha256": contract_sha256(contract_path),
        "dataset_revision": contract["dataset"]["revision"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Stale layout acceptance: {key} mismatch")
    return payload


def _verify_upstream(upstream_root: Path, lock_path: Path) -> str:
    lock = load_yaml(lock_path)
    expected = str(lock["revision"])
    training_entry = upstream_root / "tasks" / "vla" / "train_lingbotvla.py"
    if not training_entry.is_file():
        raise ValueError(f"LingBot-VLA-v2 training entry is missing: {training_entry}")
    try:
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=upstream_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            f"Upstream checkout is not a Git repository: {upstream_root}"
        ) from exc
    if actual != expected:
        raise ValueError(f"Upstream revision mismatch: {actual} != {expected}")
    return actual


def render_configs(
    *,
    output_root: Path,
    contract_path: Path,
    acceptance_path: Path,
    upstream_root: Path,
    dataset_root: Path,
    model_path: Path,
    tokenizer_path: Path,
    moge_path: Path,
    depth_path: Path,
    video_ckpt_path: Path,
    video_config_path: Path,
    run_output_dir: Path,
    norm_stats_path: Path,
    allow_unconfirmed: bool = False,
    require_norm_stats: bool = False,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    prepare_receipt = read_prepare_receipt(dataset_root, contract_path)
    if not allow_unconfirmed:
        acceptance = _read_acceptance(acceptance_path, contract_path)
        accepted_root = Path(str(acceptance.get("dataset_root", ""))).resolve()
        source_root = Path(prepare_receipt["source_dataset_root"]).resolve()
        if accepted_root != source_root:
            raise ValueError(
                "Layout acceptance was generated for a different source dataset: "
                f"{accepted_root} != {source_root}"
            )
    upstream_revision = _verify_upstream(
        upstream_root.resolve(), PROJECT_ROOT / "upstream.lock"
    )

    required_paths = {
        "dataset_root": dataset_root / "meta" / "info.json",
        "model_path": model_path / "model.safetensors.index.json",
        "tokenizer_path": tokenizer_path / "tokenizer_config.json",
        "moge_path": moge_path,
        "depth_path": depth_path,
        "video_ckpt_path": video_ckpt_path,
        "video_config_path": video_config_path,
    }
    if require_norm_stats:
        required_paths["norm_stats_path"] = norm_stats_path
    missing = [
        f"{name}:{path}" for name, path in required_paths.items() if not path.exists()
    ]
    if missing:
        raise ValueError(f"Required paths are missing: {missing}")
    norm_manifest = None
    if require_norm_stats:
        norm_manifest = validate_norm_stats(norm_stats_path, contract_path)
        if not norm_manifest.get("layout_confirmed"):
            raise ValueError(
                "Normalization stats were computed before layout confirmation"
            )
        if (
            Path(norm_manifest["prepared_dataset_root"]).resolve()
            != dataset_root.resolve()
        ):
            raise ValueError("Normalization stats target a different prepared dataset")
        if norm_manifest.get("upstream_revision") != upstream_revision:
            raise ValueError("Normalization stats target a different upstream revision")

    runtime_root = output_root.expanduser().resolve()
    robot_root = runtime_root / "configs" / "robot_configs"
    robot_path = robot_root / "take_wrong_item_right_arm.yaml"
    train_path = runtime_root / "take_wrong_item_right_arm.yaml"
    values = {
        "DATASET_ROOT": str(dataset_root.resolve()),
        "ROBOT_CONFIG_ROOT": str(robot_root),
        "MODEL_PATH": str(model_path.resolve()),
        "TOKENIZER_PATH": str(tokenizer_path.resolve()),
        "MOGE_PATH": str(moge_path.resolve()),
        "DEPTH_PATH": str(depth_path.resolve()),
        "VIDEO_CKPT_PATH": str(video_ckpt_path.resolve()),
        "VIDEO_CONFIG_PATH": str(video_config_path.resolve()),
        "RUN_OUTPUT_DIR": str(run_output_dir.expanduser().resolve()),
        "NORM_STATS_PATH": str(norm_stats_path.expanduser().resolve()),
    }
    robot_rendered = _render_text(
        PROJECT_ROOT
        / "configs"
        / "robot_configs"
        / "take_wrong_item_right_arm.yaml.in",
        values,
    )
    train_rendered = _render_text(
        PROJECT_ROOT / "configs" / "vla" / "take_wrong_item_right_arm.yaml.in",
        values,
    )
    _atomic_write(robot_path, robot_rendered)
    _atomic_write(train_path, train_rendered)

    manifest = {
        "schema_version": "lingbot-runtime-config-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": contract_sha256(contract_path),
        "dataset_repo_id": contract["dataset"]["repo_id"],
        "dataset_revision": contract["dataset"]["revision"],
        "source_dataset_root": prepare_receipt["source_dataset_root"],
        "prepared_dataset_root": str(dataset_root.resolve()),
        "prepared_lerobot_version": prepare_receipt["prepared_lerobot_version"],
        "lerobot_package_version": prepare_receipt["lerobot_package_version"],
        "upstream_revision": upstream_revision,
        "layout_confirmed": not allow_unconfirmed,
        "robot_config": str(robot_path),
        "training_config": str(train_path),
        "norm_stats": str(norm_stats_path.resolve()),
        "run_output_dir": str(run_output_dir.resolve()),
    }
    _atomic_write(
        runtime_root / "runtime_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def _path_argument(
    parser: argparse.ArgumentParser, flag: str, environment: str, **kwargs: Any
) -> None:
    parser.add_argument(flag, type=Path, default=os.environ.get(environment), **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render portable runtime configs")
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "work" / "runtime"
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE_PATH)
    parser.add_argument("--allow-unconfirmed", action="store_true")
    parser.add_argument("--require-norm-stats", action="store_true")
    _path_argument(parser, "--upstream-root", "LINGBOT_UPSTREAM_ROOT")
    _path_argument(parser, "--dataset-root", "LINGBOT_TRAIN_DATASET_ROOT")
    _path_argument(parser, "--model-path", "LINGBOT_MODEL_PATH")
    _path_argument(parser, "--tokenizer-path", "LINGBOT_TOKENIZER_PATH")
    _path_argument(parser, "--moge-path", "LINGBOT_MOGE_PATH")
    _path_argument(parser, "--depth-path", "LINGBOT_DEPTH_PATH")
    _path_argument(parser, "--video-ckpt-path", "LINGBOT_VIDEO_CKPT_PATH")
    _path_argument(parser, "--video-config-path", "LINGBOT_VIDEO_CONFIG_PATH")
    _path_argument(parser, "--run-output-dir", "LINGBOT_RUN_OUTPUT")
    _path_argument(parser, "--norm-stats-path", "LINGBOT_NORM_STATS")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    defaults = {
        "upstream_root": PROJECT_ROOT / ".upstream" / "lingbot-vla-v2",
        "run_output_dir": PROJECT_ROOT / "work" / "training",
        "norm_stats_path": (
            PROJECT_ROOT / "work" / "norm_stats" / "take_wrong_item_right_arm.json"
        ),
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    required = [
        "dataset_root",
        "model_path",
        "tokenizer_path",
        "moge_path",
        "depth_path",
        "video_ckpt_path",
        "video_config_path",
    ]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit(
            "Missing path arguments/environment variables: " + ", ".join(missing)
        )
    try:
        manifest = render_configs(
            output_root=args.output_root,
            contract_path=args.contract,
            acceptance_path=args.acceptance,
            upstream_root=args.upstream_root,
            dataset_root=args.dataset_root,
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            moge_path=args.moge_path,
            depth_path=args.depth_path,
            video_ckpt_path=args.video_ckpt_path,
            video_config_path=args.video_config_path,
            run_output_dir=args.run_output_dir,
            norm_stats_path=args.norm_stats_path,
            allow_unconfirmed=args.allow_unconfirmed,
            require_norm_stats=args.require_norm_stats,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
