"""Local experiment run history repository and state tracking."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    """Record representing a submitted remote Kaggle experiment."""

    slug: str
    kernel_id: str
    title: str
    command: str
    accelerator: str
    submitted_at: str
    status: str = "queued"
    duration_sec: float | None = None
    output_dir: str | None = None
    url: str | None = None
    error_message: str | None = None
    project_path: str | None = None
    account: str | None = None


def kernel_owner(kernel_id: str) -> str:
    """Kaggle username that owns a kernel id ('user/slug' -> 'user')."""
    return kernel_id.split("/", 1)[0] if "/" in kernel_id else ""


def get_history_file(history_path: Path | None = None) -> Path:
    """Get path to global user-level run history file (~/.kagglex/runs.json)."""
    if history_path:
        if history_path.is_dir():
            target_file = history_path / "runs.json"
        else:
            target_file = history_path
        target_file.parent.mkdir(parents=True, exist_ok=True)
        return target_file

    hist_dir = Path.home() / ".kagglex"
    legacy_file = Path.home() / ".kagglerun" / "runs.json"
    hist_dir.mkdir(parents=True, exist_ok=True)
    target_file = hist_dir / "runs.json"
    if not target_file.exists() and legacy_file.exists():
        import shutil

        shutil.copy2(legacy_file, target_file)
    return target_file


def load_all_records(history_path: Path | None = None) -> list[dict[str, Any]]:
    """Read all recorded runs from history file."""
    hist_file = get_history_file(history_path)
    if not hist_file.exists():
        return []

    try:
        with open(hist_file, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception as e:
        logger.warning("Could not read history file %s: %s", hist_file, e)
    return []


def save_all_records(
    records: list[dict[str, Any]], history_path: Path | None = None
) -> None:
    """Write all records to history file."""
    hist_file = get_history_file(history_path)
    try:
        with open(hist_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    except Exception as e:
        logger.warning("Could not write history file %s: %s", hist_file, e)


def record_run(record: RunRecord, history_path: Path | None = None) -> None:
    """Prepend a run to global history (newest first).

    Every submission is kept as its own record, even when the slug repeats:
    re-running a kernel (or the same slug on another account) must not erase
    the earlier run's duration from quota accounting. Lookups by slug or
    kernel id resolve to the newest matching record.
    """
    records = load_all_records(history_path)
    records.insert(0, asdict(record))

    save_all_records(records, history_path)
    logger.debug(
        "Recorded run '%s' in %s", record.kernel_id, get_history_file(history_path)
    )


def update_run(
    kernel_id_or_slug: str,
    updates: dict[str, Any],
    history_path: Path | None = None,
) -> RunRecord | None:
    """Update attributes of an existing run record."""
    records = load_all_records(history_path)
    target_idx = -1

    for i, r in enumerate(records):
        if (
            r.get("kernel_id") == kernel_id_or_slug
            or r.get("slug") == kernel_id_or_slug
            or r.get("kernel_id", "").endswith(f"/{kernel_id_or_slug}")
        ):
            target_idx = i
            break

    if target_idx == -1:
        return None

    records[target_idx].update(updates)
    save_all_records(records, history_path)
    rec_dict = records[target_idx]
    return RunRecord(**rec_dict)


def get_run(
    kernel_id_or_slug: str, history_path: Path | None = None
) -> RunRecord | None:
    """Find a run record by kernel id or slug."""
    records = load_all_records(history_path)
    for r in records:
        if (
            r.get("kernel_id") == kernel_id_or_slug
            or r.get("slug") == kernel_id_or_slug
            or r.get("kernel_id", "").endswith(f"/{kernel_id_or_slug}")
        ):
            return RunRecord(**r)
    return None


def list_runs(limit: int = 20, history_path: Path | None = None) -> list[RunRecord]:
    """List recent run records ordered by submission time."""
    records = load_all_records(history_path)
    results: list[RunRecord] = []
    for r in records[:limit]:
        try:
            results.append(RunRecord(**r))
        except Exception:
            continue
    return results


def parse_timestamp(timestamp_str: str) -> float | None:
    """Parse various timestamp string formats into Unix epoch seconds."""
    import datetime

    if not timestamp_str or not isinstance(timestamp_str, str):
        return None
    cleaned = timestamp_str.strip()
    if cleaned.endswith(" UTC"):
        cleaned = cleaned[:-4].strip()
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.datetime.strptime(cleaned, fmt)
            return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def get_quota_usage(
    history_path: Path | None = None,
    window_days: int = 7,
    gpu_limit_hours: float = 30.0,
    tpu_limit_hours: float = 20.0,
    current_time: float | None = None,
    username: str | None = None,
) -> dict[str, Any]:
    """Calculate GPU and TPU accelerator quota usage across a rolling time window.

    Args:
        history_path: Optional path to history file/directory.
        window_days: Number of days in rolling window (default: 7).
        gpu_limit_hours: Weekly GPU quota limit in hours (default: 30.0).
        tpu_limit_hours: Weekly TPU quota limit in hours (default: 20.0).
        current_time: Optional reference epoch timestamp (default: current time).
        username: Only count runs owned by this Kaggle username (the kernel id
            prefix), so each account's quota is tracked separately.

    Returns:
        Dictionary containing aggregated hours, limits, percentages, and run items.
    """
    import time

    now = current_time if current_time is not None else time.time()
    cutoff_epoch = now - (window_days * 86400.0)

    records = load_all_records(history_path)

    gpu_seconds = 0.0
    tpu_seconds = 0.0
    runs_in_window: list[dict[str, Any]] = []

    for r in records:
        sub_time = parse_timestamp(r.get("submitted_at", ""))
        if sub_time is None or sub_time < cutoff_epoch:
            continue
        if (
            username is not None
            and kernel_owner(str(r.get("kernel_id", ""))).casefold()
            != username.casefold()
        ):
            continue

        accel = str(r.get("accelerator", "none")).lower()
        dur = float(r.get("duration_sec") or 0.0)

        if accel in {"t4-2x", "p100", "t4", "p100-1x"}:
            gpu_seconds += dur
            accel_type = "GPU"
        elif accel in {"v3-8", "tpu", "v2-8"}:
            tpu_seconds += dur
            accel_type = "TPU"
        else:
            accel_type = "CPU"

        runs_in_window.append(
            {
                "kernel_id": r.get("kernel_id", ""),
                "title": r.get("title", ""),
                "accelerator": accel,
                "accelerator_type": accel_type,
                "submitted_at": r.get("submitted_at", ""),
                "duration_sec": dur,
                "status": r.get("status", "unknown"),
            }
        )

    gpu_hours = gpu_seconds / 3600.0
    tpu_hours = tpu_seconds / 3600.0

    gpu_remaining = max(0.0, gpu_limit_hours - gpu_hours)
    tpu_remaining = max(0.0, tpu_limit_hours - tpu_hours)

    gpu_pct = (
        min(100.0, (gpu_hours / gpu_limit_hours * 100.0))
        if gpu_limit_hours > 0
        else 0.0
    )
    tpu_pct = (
        min(100.0, (tpu_hours / tpu_limit_hours * 100.0))
        if tpu_limit_hours > 0
        else 0.0
    )

    return {
        "window_days": window_days,
        "gpu": {
            "used_hours": round(gpu_hours, 2),
            "limit_hours": gpu_limit_hours,
            "remaining_hours": round(gpu_remaining, 2),
            "used_pct": round(gpu_pct, 1),
            "run_count": sum(
                1 for item in runs_in_window if item["accelerator_type"] == "GPU"
            ),
        },
        "tpu": {
            "used_hours": round(tpu_hours, 2),
            "limit_hours": tpu_limit_hours,
            "remaining_hours": round(tpu_remaining, 2),
            "used_pct": round(tpu_pct, 1),
            "run_count": sum(
                1 for item in runs_in_window if item["accelerator_type"] == "TPU"
            ),
        },
        "runs": runs_in_window,
    }
