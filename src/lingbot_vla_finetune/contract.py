from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT_PATH = PROJECT_ROOT / "configs" / "dataset_contract.yaml"
DEFAULT_ACCEPTANCE_PATH = PROJECT_ROOT / "work" / "layout_acceptance.json"


class ContractError(ValueError):
    pass


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ContractError(f"Cannot read YAML {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"YAML root must be a mapping: {path}")
    return payload


def contract_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    payload = load_yaml(path)
    required = {
        "schema_version",
        "dataset",
        "prepared_dataset",
        "raw_schema",
        "training_mapping",
        "camera_mapping",
        "acceptance_assumptions",
        "audit_thresholds",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ContractError(f"Contract is missing sections: {missing}")

    dataset = payload["dataset"]
    prepared = payload["prepared_dataset"]
    raw = payload["raw_schema"]
    mapping = payload["training_mapping"]
    if not all(isinstance(value, dict) for value in (dataset, prepared, raw)):
        raise ContractError(
            "dataset, prepared_dataset, and raw_schema must be mappings"
        )
    if not isinstance(mapping, dict):
        raise ContractError("training_mapping must be a mapping")
    if len(str(dataset.get("revision") or "")) != 40:
        raise ContractError("dataset.revision must be a 40-character commit SHA")
    for key in ("lerobot_version", "lerobot_package_version", "receipt_name"):
        if not str(prepared.get(key) or ""):
            raise ContractError(f"prepared_dataset.{key} must be set")
    for key in ("state_width", "action_width"):
        if int(raw.get(key, 0)) <= 0:
            raise ContractError(f"raw_schema.{key} must be positive")
    return payload


def format_lerobot_path(
    template: str, episode_index: int, chunk_size: int, **values: Any
) -> Path:
    episode_chunk = episode_index // chunk_size
    return Path(
        template.format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
            **values,
        )
    )
