"""Programmatic Python API for kagglex."""

import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kagglex.accounts import (
    Account,
    resolve_account,
    resolve_account_for_kernel,
    set_active_account,
)
from kagglex.bootstrap import generate_bootstrap_script
from kagglex.client import (
    cancel_kernel,
    get_authenticated_username,
    get_kernel_status,
    poll_kernel,
    pull_kernel_output,
    push_kernel,
    stream_kernel_logs,
)
from kagglex.config import DatasetConfig, RunConfig
from kagglex.dataset import create_dataset_metadata, push_dataset
from kagglex.history import RunRecord, list_runs, record_run, update_run
from kagglex.packager import (
    create_kernel_metadata,
    package_local_data,
    package_project,
)
from kagglex.workspace import (
    extract_script_path_from_command,
    find_repo_root,
    validate_python_syntax,
)

logger = logging.getLogger(__name__)


class Job:
    """Handle for a submitted Kaggle remote job."""

    def __init__(
        self, kernel_id: str, url: str, config: RunConfig, repo_root: Path
    ) -> None:
        self.kernel_id = kernel_id
        self.url = url
        self.config = config
        self.repo_root = repo_root
        self._last_status_info: dict[str, Any] | None = None

    @property
    def status(self) -> str:
        """Fetch current remote execution status."""
        status_info = get_kernel_status(self.kernel_id)
        self._last_status_info = status_info
        return str(status_info.get("status", "unknown"))

    @property
    def last_status_info(self) -> dict[str, Any] | None:
        """Cached or last fetched status dictionary."""
        return self._last_status_info

    def wait(
        self,
        poll_interval_sec: int | None = None,
        timeout_sec: int | None = None,
        on_status_change: Callable[[str, float], None] | None = None,
    ) -> dict[str, Any]:
        """Block until kernel terminates (complete, error, cancel).

        Args:
            poll_interval_sec: Seconds between polling requests.
            timeout_sec: Maximum seconds before TimeoutError.
            on_status_change: Optional callback for status changes.

        Returns:
            Final status info dictionary.
        """
        interval = poll_interval_sec or self.config.poll_interval
        timeout = timeout_sec or self.config.timeout_sec
        info = poll_kernel(
            self.kernel_id,
            poll_interval_sec=interval,
            timeout_sec=timeout,
            on_status_change=on_status_change,
        )
        self._last_status_info = info
        update_run(
            self.kernel_id,
            {
                "status": info.get("status", "unknown"),
                "duration_sec": info.get("duration_sec"),
                "error_message": info.get("failure_message"),
            },
        )
        return info

    def stream_logs(self, on_line: Callable[[str], None] | None = None) -> None:
        """Stream live execution logs to stdout or callback."""
        stream_kernel_logs(self.kernel_id, on_line=on_line)

    def cancel(self) -> bool:
        """Cancel this job on Kaggle."""
        ok = cancel_kernel(self.kernel_id)
        if ok:
            update_run(self.kernel_id, {"status": "cancelled"})
        return ok

    def pull_outputs(
        self,
        destination_dir: Path | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> list[Path]:
        """Download output files from completed job."""
        dest = destination_dir or self.config.output_dir or (self.repo_root / "outputs")
        inc = include_patterns or self.config.include_outputs
        exc = exclude_patterns or self.config.exclude_outputs
        saved = pull_kernel_output(
            self.kernel_id,
            destination_dir=dest,
            include_patterns=inc,
            exclude_patterns=exc,
        )
        update_run(self.kernel_id, {"output_dir": str(dest)})
        return saved


class KaggleRunner:
    """High-level runner orchestrating local packaging, remote execution, and output sync."""

    def __init__(
        self,
        repo_root: Path | None = None,
        staging_dir: Path | None = None,
    ) -> None:
        self.repo_root = (repo_root or find_repo_root()).resolve()
        self.staging_dir = staging_dir or (self.repo_root / ".kagglex" / "staging")

    def use_account(self, config: RunConfig) -> Account | None:
        """Resolve `config.account` ("Phong", "auto", or None for default) and activate it.

        After resolution `config.account` holds the concrete account name, so a
        later call (e.g. stage() then run()) never re-picks a different "auto" account.
        """
        acct = resolve_account(config.account, accelerator=config.gpu_type)
        if acct is not None or config.account is not None:
            set_active_account(acct)
        if acct is not None:
            config.account = acct.name
        return acct

    def stage(self, config: RunConfig) -> Path:
        """Stage project files, payloads, and bootstrap scripts locally."""
        self.use_account(config)
        slug = config.slug or "run"
        staging_base = self.staging_dir / slug
        staging_base.mkdir(parents=True, exist_ok=True)
        user = get_authenticated_username()

        # Target project directory
        proj_dir = config.project_dir or self.repo_root

        # Pre-flight syntax validation if target python script exists
        script_to_check = config.target_file or extract_script_path_from_command(
            config.command, proj_dir
        )
        if script_to_check and script_to_check.exists():
            validate_python_syntax(script_to_check)

        # 1. Package Python project
        pkg_zip = staging_base / "pkg_payload.zip"
        package_project(
            project_dir=proj_dir,
            output_zip=pkg_zip,
            target_file=config.target_file,
            allow_large_payload=config.auto_dataset,
        )

        # 2. Package local data if requested
        data_zip_path = None
        if config.include_data:
            data_paths = [Path(p) for p in config.include_data]
            data_zip = staging_base / "data_payload.zip"
            res = package_local_data(
                data_paths,
                base_dir=proj_dir,
                output_zip=data_zip,
                allow_large_payload=config.auto_dataset,
            )
            if res:
                data_zip_path = data_zip

        # 3. If auto_dataset is enabled, prepare payload dataset directory
        if config.auto_dataset:
            ds_slug = config.auto_dataset_slug or f"kagglex-payload-{config.slug}"
            ds_dir = staging_base / "payload_dataset"
            ds_dir.mkdir(parents=True, exist_ok=True)

            if pkg_zip.exists():
                shutil.copy2(pkg_zip, ds_dir / "pkg_payload.zip")
            if data_zip_path and data_zip_path.exists():
                shutil.copy2(data_zip_path, ds_dir / "data_payload.zip")

            ds_config = DatasetConfig(
                title=f"Payload for {config.title}",
                slug=ds_slug,
                data_dir=ds_dir,
                is_public=False,
            )
            create_dataset_metadata(ds_config, user, ds_dir)

            full_ds_ref = f"{user}/{ds_slug}"
            if (
                full_ds_ref not in config.dataset_slugs
                and ds_slug not in config.dataset_slugs
            ):
                config.dataset_slugs.append(full_ds_ref)

        # 4. Generate bootstrap script
        bootstrap_file = staging_base / "kaggle_bootstrap.py"
        generate_bootstrap_script(
            config=config,
            output_path=bootstrap_file,
            pkg_zip_path=pkg_zip,
            data_zip_path=data_zip_path,
            allow_large_payload=config.auto_dataset,
        )

        # 5. Generate kernel-metadata.json
        create_kernel_metadata(
            config=config,
            kaggle_username=user,
            staging_dir=staging_base,
            code_file="kaggle_bootstrap.py",
        )

        return staging_base

    def run(
        self,
        config: RunConfig | None = None,
        command: str | None = None,
        title: str | None = None,
        wait: bool = True,
        stream: bool = False,
        pull: bool = True,
        **kwargs: Any,
    ) -> Job:
        """Stage, submit, monitor, and optionally pull outputs for a Kaggle run."""
        if config is None:
            if not command or not title:
                raise ValueError(
                    "Must provide either a RunConfig or both command and title."
                )
            config = RunConfig(command=command, title=title, **kwargs)

        staging_base = self.stage(config)

        # If auto_dataset is configured, publish/version dataset before kernel push
        if config.auto_dataset:
            ds_slug = config.auto_dataset_slug or f"kagglex-payload-{config.slug}"
            ds_dir = staging_base / "payload_dataset"
            if ds_dir.exists():
                ds_config = DatasetConfig(
                    title=f"Payload for {config.title}",
                    slug=ds_slug,
                    data_dir=ds_dir,
                    is_public=False,
                )
                logger.info("Uploading auto-dataset payload '%s' to Kaggle...", ds_slug)
                push_dataset(
                    ds_config, version_notes=f"Auto-dataset payload for {config.slug}"
                )

        actual_kernel_id, kernel_url = push_kernel(staging_base)

        job = Job(
            kernel_id=actual_kernel_id,
            url=kernel_url,
            config=config,
            repo_root=self.repo_root,
        )

        # Record in global history
        rec = RunRecord(
            slug=config.slug or config.title,
            kernel_id=actual_kernel_id,
            title=config.title,
            command=config.command,
            accelerator=config.gpu_type,
            submitted_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            status="queued",
            url=kernel_url,
            project_path=str(self.repo_root),
            account=config.account,
        )
        record_run(rec)

        if not wait:
            return job

        if stream:
            job.stream_logs()

        status_info = job.wait()
        if pull and status_info.get("status") == "complete":
            job.pull_outputs()

        return job

    def list_runs(self, limit: int = 20) -> list[RunRecord]:
        """List recent runs from global history."""
        return list_runs(limit=limit)

    def cancel(self, kernel_id_or_slug: str) -> bool:
        """Cancel a run by kernel id or slug (using the account that owns it)."""
        acct, known_id = resolve_account_for_kernel(kernel_id_or_slug)
        set_active_account(acct)
        return cancel_kernel(known_id or kernel_id_or_slug)
