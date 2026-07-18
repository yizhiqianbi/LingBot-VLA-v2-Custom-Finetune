from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import yaml

from .contract import DEFAULT_CONTRACT_PATH, load_contract


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_environment(
    upstream_root: Path,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    *,
    require_cuda: bool = False,
) -> dict[str, Any]:
    import torch

    upstream = upstream_root.expanduser().resolve()
    lock = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "upstream.lock").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(lock, dict):
        raise ValueError("upstream.lock must be a YAML mapping")
    import subprocess

    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=upstream,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"Invalid upstream checkout: {upstream}") from exc
    if revision != lock["revision"]:
        raise ValueError(
            f"Upstream revision mismatch: {revision} != {lock['revision']}"
        )

    contract = load_contract(contract_path)
    lerobot_version = _package_version("lerobot")
    wanted_lerobot = str(contract["prepared_dataset"]["lerobot_package_version"])
    if lerobot_version != wanted_lerobot:
        raise ValueError(
            f"LeRobot version mismatch: {lerobot_version} != {wanted_lerobot}"
        )

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    if require_cuda and not cuda_available:
        raise ValueError("CUDA is required for LingBot-VLA-v2 training")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if require_cuda and not visible:
        raise ValueError("CUDA_VISIBLE_DEVICES must list the training GPUs")
    devices = []
    if cuda_available:
        for index in range(device_count):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "memory_bytes": properties.total_memory,
                    "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                }
            )

    return {
        "status": "ok",
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": {
            name: _package_version(name)
            for name in (
                "torch",
                "torchvision",
                "transformers",
                "datasets",
                "huggingface-hub",
                "lerobot",
            )
        },
        "upstream_revision": revision,
        "cuda_available": cuda_available,
        "cuda_visible_devices": visible,
        "device_count": device_count,
        "devices": devices,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check the LingBot runtime")
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=os.environ.get("LINGBOT_UPSTREAM_ROOT"),
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--require-cuda", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.upstream_root is None:
        raise SystemExit("--upstream-root or LINGBOT_UPSTREAM_ROOT is required")
    try:
        result = check_environment(
            args.upstream_root,
            args.contract,
            require_cuda=args.require_cuda,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
