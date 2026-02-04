import copy
import os
from typing import Any, Dict

import yaml


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")

    # Load the main file
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Check for includes
    includes = cfg.pop("include", [])
    if not isinstance(includes, list):
        includes = [includes]

    # Resolve base dir for relative includes
    base_dir = os.path.dirname(path)

    # Load and merge includes in order
    merged_cfg = {}
    for inc in includes:
        inc_path = inc if os.path.isabs(inc) else os.path.join(base_dir, inc)
        inc_cfg = load_config(inc_path)  # Recursive load
        deep_update(merged_cfg, inc_cfg)

    # Finally update with the main file's content
    deep_update(merged_cfg, cfg)
    return merged_cfg


def set_by_dotted_key(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    keys = dotted.split(".")
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
