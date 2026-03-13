"""Shared persistence helpers for policy evaluation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from model.io import append_index_entry, now_utc_iso, slugify, write_json


def save_evaluation_artifacts(
    *,
    output_dir: str | Path,
    raw_steps,
    kpi_summary,
    metadata: Mapping[str, Any],
    evaluation_id: Optional[str] = None,
) -> dict[str, Any]:
    """Save raw steps/KPI summary + metadata under output_dir."""
    out_dir = Path(output_dir)
    eid = evaluation_id or f"eval-{now_utc_iso().replace(':', '').replace('-', '')}"
    run_dir = out_dir / slugify(eid)
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_path = run_dir / "raw_steps.csv"
    kpi_path = run_dir / "kpi_summary.csv"
    meta_path = run_dir / "evaluation.json"

    raw_steps.to_csv(raw_path, index=False)
    kpi_summary.to_csv(kpi_path, index=False)

    created_at = now_utc_iso()
    payload = {
        "evaluation_id": eid,
        "created_at": created_at,
        "raw_steps_path": str(raw_path.resolve()),
        "kpi_summary_path": str(kpi_path.resolve()),
        **dict(metadata),
    }
    write_json(meta_path, payload)

    return {
        "evaluation_id": eid,
        "created_at": created_at,
        "evaluation_dir": str(run_dir.resolve()),
        "metadata_path": str(meta_path.resolve()),
        "raw_steps_path": str(raw_path.resolve()),
        "kpi_summary_path": str(kpi_path.resolve()),
    }


def append_evaluation_index(
    *,
    index_path: str | Path,
    evaluation_id: str,
    created_at: str,
    metadata_path: str | Path,
) -> str:
    """Append one evaluation entry to index_path and return index path."""
    return append_index_entry(
        index_path=index_path,
        list_key="evaluations",
        entry={
            "evaluation_id": str(evaluation_id),
            "created_at": str(created_at),
            "evaluation_path": str(Path(metadata_path).resolve()),
        },
    )
