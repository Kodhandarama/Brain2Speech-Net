"""YAML config loading with ``key=value`` command-line overrides (dotted keys, YAML-typed values)."""
from __future__ import annotations

import copy
from typing import Any, Dict, Sequence

import yaml


def load_config(path: str, overrides: Sequence[str] = ()) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def dump_config(cfg: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(copy.deepcopy(cfg), f, sort_keys=False)
