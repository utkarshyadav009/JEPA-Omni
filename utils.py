"""Small shared helpers: attribute-accessible config loading + distributed glue.

Kept intentionally tiny and dependency-light so the training / eval entry
points stay plain-PyTorch scripts rather than a framework.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import yaml


class AttrDict(dict):
    """A ``dict`` that also supports attribute access (``cfg.model.lr``).

    Because it subclasses ``dict``, ``SpineConfig(**cfg.model)`` and
    ``cfg["model"]`` both keep working.
    """

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:  # pragma: no cover - defensive
        try:
            del self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return AttrDict({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def load_config(path: str) -> AttrDict:
    """Load a YAML file into an :class:`AttrDict` (nested dicts wrapped too)."""
    with open(path, "r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}
    cfg = _wrap(raw)
    assert isinstance(cfg, AttrDict)
    return cfg


_MISSING = object()


def cfg_get(cfg: Any, *paths: str, default: Any = None) -> Any:
    """Look up the first present dotted ``path`` in ``cfg``.

    Lets the entry points tolerate small differences in the authoritative
    config layout, e.g. ``cfg_get(cfg, "optim.lr", "lr", default=1e-4)`` finds
    the learning rate whether it lives under ``optim:`` or at the top level.
    """
    for path in paths:
        cur: Any = cfg
        for key in path.split("."):
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                cur = _MISSING
                break
        if cur is not _MISSING:
            return cur
    return default


# --------------------------------------------------------------------------- #
# Distributed helpers
# --------------------------------------------------------------------------- #
def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return get_rank() == 0


def maybe_int(value: Optional[Any]) -> Optional[int]:
    return None if value is None else int(value)
