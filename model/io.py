"""Shared lightweight IO helpers for model persistence modules."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip()).strip("-").lower()
    return s or "item"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def append_index_entry(
    *,
    index_path: str | Path,
    list_key: str,
    entry: Mapping[str, Any],
) -> str:
    path = Path(index_path)
    if path.exists():
        index = read_json(path)
    else:
        index = {list_key: []}
    index.setdefault(list_key, []).append(dict(entry))
    write_json(path, index)
    return str(path.resolve())
