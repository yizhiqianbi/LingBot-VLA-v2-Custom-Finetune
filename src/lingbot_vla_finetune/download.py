from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

from .contract import DEFAULT_CONTRACT_PATH, PROJECT_ROOT, load_contract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the contract-pinned custom dataset"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=os.environ.get("LINGBOT_SOURCE_DATASET_ROOT"),
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument(
        "--token-file", type=Path, default=os.environ.get("HF_TOKEN_FILE")
    )
    parser.add_argument("--max-workers", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dataset_root is None:
        raise SystemExit("--dataset-root or LINGBOT_SOURCE_DATASET_ROOT is required")
    destination = args.dataset_root.expanduser().resolve()
    project_root = PROJECT_ROOT.resolve()
    if destination == project_root or project_root in destination.parents:
        raise SystemExit("Dataset destination must be outside the code repository")

    token = None
    if args.token_file is not None:
        token = args.token_file.expanduser().read_text(encoding="utf-8").strip()
        if not token:
            raise SystemExit(f"Token file is empty: {args.token_file}")

    contract = load_contract(args.contract)
    dataset = contract["dataset"]
    resolved = snapshot_download(
        repo_id=str(dataset["repo_id"]),
        repo_type="dataset",
        revision=str(dataset["revision"]),
        token=token,
        local_dir=destination,
        max_workers=max(1, args.max_workers),
    )
    print(
        json.dumps(
            {
                "repo_id": dataset["repo_id"],
                "revision": dataset["revision"],
                "dataset_root": str(Path(resolved).resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
